

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import random
from functools import partial

# -- Insert the path(s) to your evaluation_code so Python can find them
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
EVALUATION_CODE_PATH = os.path.join(REPO_ROOT, "evaluation_code")
EVALS_PATH = os.path.join(EVALUATION_CODE_PATH, "evals")
SRC_PATH = os.path.join(EVALUATION_CODE_PATH, "src")

for p in [EVALS_PATH, SRC_PATH, EVALUATION_CODE_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation modules
from evals.intuitive_physics.eval import init_model, extract_losses, compute_metrics
from evals.intuitive_physics.data_manager import init_data
from evals.intuitive_physics.utils import get_dataset_paths, get_time_masks
from src.utils.transforms import make_transforms
from src.masks.utils import apply_masks


def load_model_from_config(config_path, device=None):
    """
    Load model based on configuration file

    Args:
        config_path: Path to the yaml config file
        device: PyTorch device (will use CUDA if available if not specified)
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pretrain_cfg = config['pretrain']
    data_cfg = config['data']
    
    # Update paths to be absolute if needed
    pretrained_path = os.path.join(pretrain_cfg['folder'], pretrain_cfg['checkpoint'])

    # Initialize model
    encoder, target_encoder, predictor = init_model(
        device=device,
        pretrained=pretrained_path,
        model_name=pretrain_cfg["model_name"],
        patch_size=pretrain_cfg["patch_size"],
        crop_size=data_cfg["resolution"],
        frames_per_clip=pretrain_cfg["frames_per_clip"],
        tubelet_size=pretrain_cfg["tubelet_size"],
        use_sdpa=pretrain_cfg.get("use_sdpa", True),
        use_SiLU=pretrain_cfg.get("use_silu", False),
        wide_SiLU=pretrain_cfg.get("wide_silu", True),
        is_causal=pretrain_cfg.get("is_causal", False),
        pred_is_causal=pretrain_cfg.get("pred_is_causal", False),
        pred_depth=pretrain_cfg.get("pred_depth", 12),
        uniform_power=pretrain_cfg.get("uniform_power", False),
        enc_checkpoint_key=pretrain_cfg.get("enc_checkpoint_key", "encoder"),
        pred_checkpoint_key=pretrain_cfg.get("pred_checkpoint_key", "predictor"),
        is_mae=False
    )

    # Freeze
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in target_encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    return {
        "encoder": encoder, 
        "target_encoder": target_encoder, 
        "predictor": predictor, 
        "config": config
    }


class ContrastiveActivationPatcher:
    """
    Class for patching activations from one event type to another
    (possible -> impossible and vice versa)
    """
    def __init__(self, model):
        self.model = model
        self.encoder = model["encoder"]
        self.target_encoder = model["target_encoder"]
        self.predictor = model["predictor"]
        self.hooks = {}
        self.stored_activations = {}
        self.device = next(self.encoder.parameters()).device
        
        # Get encoder embedding dimension
        self.embed_dim = self.encoder.backbone.blocks[0].attn.qkv.in_features
        
        # Get number of heads for attention head patching
        try:
            # First try the most common attribute name
            self.num_heads = self.encoder.backbone.blocks[0].attn.num_heads
        except AttributeError:
            try:
                # Next try to get it from qkv projection shape
                qkv = self.encoder.backbone.blocks[0].attn.qkv
                if hasattr(qkv, "weight"):
                    out_dim = qkv.weight.shape[0]
                    self.num_heads = out_dim // 3 // (self.embed_dim // 16)
            except AttributeError:
                # Default to 16 heads for ViT models
                self.num_heads = 16
                print(f"Could not determine number of attention heads, defaulting to {self.num_heads}")
        
        # Calculate head dimension
        self.head_dim = self.embed_dim // self.num_heads
        
        print(f"Model has embedding dimension {self.embed_dim}")
        print(f"Model has {self.num_heads} attention heads per block with head dimension {self.head_dim}")
    
    def _get_block_module(self, block_idx, component="full", head_idx=None):
        """Get the specific module for a given block and component"""
        if component == "full":
            return self.encoder.backbone.blocks[block_idx]
        elif component == "att":
            return self.encoder.backbone.blocks[block_idx].attn
        elif component == "mlp":
            return self.encoder.backbone.blocks[block_idx].mlp
        elif component == "head":
            # We still return the attention module, but will handle head-specific patches separately
            if head_idx is None:
                raise ValueError("head_idx must be provided for component='head'")
            return self.encoder.backbone.blocks[block_idx].attn
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def _make_hook_fn(self, block_idx, component):
        """Create a hook function to store activations"""
        
        def hook_fn(module, input, output):
            # Store the activations for the component
            if isinstance(output, tuple) and len(output) == 2:
                actual_output = output[0]
                attention_weights = output[1]
            else:
                actual_output = output
                attention_weights = None
                
            self.stored_activations[(block_idx, component)] = {
                "input": [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input],
                "output": actual_output.detach().clone() if isinstance(actual_output, torch.Tensor) else actual_output
            }
            
            # If attention weights are available, store those too
            if attention_weights is not None:
                self.stored_activations[(block_idx, component)]["attention"] = attention_weights.detach().clone()
        
        return hook_fn
    
    def register_hooks(self, block_indices, components=["full", "att", "mlp"]):
        """
        Register hooks to capture activations for specified blocks and components
        
        Args:
            block_indices: List of block indices to hook
            components: List of components to hook ('full', 'att', 'mlp')
        """
        # Clear any existing hooks and activations
        self.remove_hooks()
        self.stored_activations = {}
        
        # Register new hooks
        for block_idx in block_indices:
            for component in components:
                # We only need to track standard components (full, att, mlp)
                # For head patching, we don't need special hooks since we'll 
                # directly modify the QKV weights during patching
                if component != "head":
                    module = self._get_block_module(block_idx, component)
                    hook = module.register_forward_hook(
                        self._make_hook_fn(block_idx, component)
                    )
                    self.hooks[(block_idx, component)] = hook
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for key, hook in list(self.hooks.items()):
            # Only call remove() on PyTorch hook objects, not on our custom callables
            if hasattr(hook, 'remove'):
                hook.remove()
        self.hooks = {}
    
    def run_forward_pass(self, x, context_len=8):
        """Run a forward pass to collect activations"""
        # Generate masks for specified context length
        m, m_, full_m = get_time_masks(
            n_timesteps=context_len,
            spatial_size=(16, 16),  # For ViT-16
            temporal_size=2,
            spatial_dim=(224, 224),
            temporal_dim=x.shape[2],
            as_bool=False
        )
        
        # Add batch dimension and move to device
        B = x.shape[0]
        m = m.unsqueeze(0).repeat(B, 1).to(self.device)
        m_ = m_.unsqueeze(0).repeat(B, 1).to(self.device)
        full_m = full_m.unsqueeze(0).repeat(B, 1).to(self.device)
        
        with torch.no_grad():
            # Step 1: Get context representation from partial observation
            context_out = self.encoder(x, [m])
            
            # Step 2: Get target representation from full observation
            target_out = self.target_encoder(x, [full_m])[0]
            target_out = F.layer_norm(target_out, (target_out.size(-1),))
            
            # Step 3: Apply masks to target
            masked_targets = apply_masks(target_out, [m_], concat=False)
            
            # Step 4: Predict target from context
            preds = self.predictor(context_out, masked_targets, [m], [m_])
            out = preds[0]
            targets = masked_targets[0]
            
            # Step 5: Compute L1 loss
            loss = F.l1_loss(out, targets, reduction="mean")
            
        return loss.item()
    
    def patch_activations(self, x, block_idx, component, source_activations, context_len=8, head_idx=None):
        """
        Run forward pass with patched activations
        
        Args:
            x: Input tensor
            block_idx: Block index to patch
            component: Component to patch ('full', 'att', 'mlp', or 'head')
            source_activations: Activations to patch in
            context_len: Context length for masks
            head_idx: Attention head index (only used when component='head')
        """
        # Generate masks for specified context length
        m, m_, full_m = get_time_masks(
            n_timesteps=context_len,
            spatial_size=(16, 16),  # For ViT-16
            temporal_size=2,
            spatial_dim=(224, 224),
            temporal_dim=x.shape[2],
            as_bool=False
        )
        
        # Add batch dimension and move to device
        B = x.shape[0]
        m = m.unsqueeze(0).repeat(B, 1).to(self.device)
        m_ = m_.unsqueeze(0).repeat(B, 1).to(self.device)
        full_m = full_m.unsqueeze(0).repeat(B, 1).to(self.device)
        
        # Create a standard patching hook function for regular components (full, att, mlp)
        def patch_hook_fn_standard(module, input, output):
            try:
                # Replace the activation with the one from source
                source_output = source_activations["output"]
                
                # Make sure we match the output type (tuple vs tensor)
                if isinstance(output, tuple) and len(output) == 2:
                    if isinstance(source_output, tuple) and len(source_output) == 2:
                        # Both are tuples, return as is
                        return source_output
                    else:
                        # Source is tensor but output is tuple, wrap it
                        return (source_output, output[1])
                else:
                    # Output is tensor
                    if isinstance(source_output, tuple) and len(source_output) == 2:
                        # Source is tuple but output is tensor, extract first element
                        return source_output[0]
                    else:
                        # Both are tensors
                        return source_output
            except Exception as e:
                print(f"Error in patching hook: {e}")
                # If anything goes wrong, just return the original output
                return output
        
        # Register temporary patching hook(s)
        hooks = []
        original_states = {}
        
        if component == "head":
            # For head patching, we'll patch the entire attention module
            # This is true contrastive patching, replacing the output of the module
            
            # Get the attention module
            attn_module = self._get_block_module(block_idx, "att")
            
            # Store the original forward method
            original_states['attn_forward'] = attn_module.forward
            
            # Create a patched forward method for the attention module
            def patched_attn_forward(*args, **kwargs):
                try:
                    # Run the original forward to maintain the same output structure
                    real_output = original_states['attn_forward'](*args, **kwargs)
                    
                    # Get output from source activations
                    source_output = source_activations["output"]
                    
                    # Handle tuple outputs (output, attention_weights)
                    if isinstance(real_output, tuple) and len(real_output) == 2:
                        # If attention weights are available in source, use them
                        if "attention" in source_activations:
                            return (source_output, source_activations["attention"])
                        # Otherwise keep original attention weights
                        return (source_output, real_output[1])
                    else:
                        # Direct output replacement
                        return source_output
                        
                except Exception as e:
                    print(f"Error in patched_attn_forward: {e}")
                    # If anything goes wrong, just return the source output without trying to unpack
                    return source_activations["output"]
            
            # Replace the forward method
            attn_module.forward = patched_attn_forward
                
        else:
            # For standard component patching (full, att, mlp)
            module = self._get_block_module(block_idx, component)
            hook = module.register_forward_hook(patch_hook_fn_standard)
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                # Step 1: Get context representation from partial observation
                context_out = self.encoder(x, [m])
                
                # Step 2: Get target representation from full observation
                target_out = self.target_encoder(x, [full_m])[0]
                target_out = F.layer_norm(target_out, (target_out.size(-1),))
                
                # Step 3: Apply masks to target
                masked_targets = apply_masks(target_out, [m_], concat=False)
                
                # Step 4: Predict target from context
                preds = self.predictor(context_out, masked_targets, [m], [m_])
                out = preds[0]
                targets = masked_targets[0]
                
                # Step 5: Compute L1 loss
                loss = F.l1_loss(out, targets, reduction="mean")
        finally:
            # Ensure we always remove all hooks
            for hook in hooks:
                hook.remove()
            
            # Restore original methods if we modified them
            if component == "head":
                attn_module = self._get_block_module(block_idx, "att")
                
                # Restore original forward method
                if 'attn_forward' in original_states:
                    attn_module.forward = original_states['attn_forward']
            
        return loss.item()


class ContrastivePatching:
    """
    Class for running contrastive activation patching experiments
    """
    def __init__(self, config_path, output_dir="contrastive_patching_results", device=None):
        """
        Initialize the contrastive patching experiment
        
        Args:
            config_path: Path to the yaml config file
            output_dir: Directory to save results
            device: PyTorch device
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        
        # Load model and config
        print(f"Loading model from config: {config_path}")
        self.model = load_model_from_config(config_path, self.device)
        self.config = self.model["config"]
        
        # Initialize activation patcher
        self.patcher = ContrastiveActivationPatcher(self.model)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results
        self.results = {}
    
    def load_pair(self, pair_path, pair_prefix):
        """
        Load a pair of possible/impossible events
        
        Args:
            pair_path: Path to directory containing the pairs
            pair_prefix: Prefix for the pair files
        """
        possible_path = os.path.join(pair_path, f"{pair_prefix}_possible.pt")
        impossible_path = os.path.join(pair_path, f"{pair_prefix}_impossible.pt")
        
        # Check if files exist
        if not os.path.exists(possible_path) or not os.path.exists(impossible_path):
            raise FileNotFoundError(f"Could not find pair files: {possible_path}, {impossible_path}")
        
        # Load tensors
        x_possible = torch.load(possible_path, map_location=self.device)
        x_impossible = torch.load(impossible_path, map_location=self.device)
        
        # Ensure correct shape [B, C, T, H, W]
        if len(x_possible.shape) == 4:  # [C, T, H, W]
            x_possible = x_possible.unsqueeze(0)
        if len(x_impossible.shape) == 4:  # [C, T, H, W] 
            x_impossible = x_impossible.unsqueeze(0)
        
        # Convert to float tensors
        if x_possible.dtype != torch.float32:
            print(f"Converting possible tensor from {x_possible.dtype} to float32")
            x_possible = x_possible.float()
        if x_impossible.dtype != torch.float32:
            print(f"Converting impossible tensor from {x_impossible.dtype} to float32")
            x_impossible = x_impossible.float()
        
        # Scale if needed (if converted from uint8)
        if x_possible.max() > 1.0:
            print("Scaling possible tensor values to [0,1] range")
            x_possible = x_possible / 255.0
            
        if x_impossible.max() > 1.0:
            print("Scaling impossible tensor values to [0,1] range")
            x_impossible = x_impossible / 255.0
        
        print(f"Loaded pair {pair_prefix}:")
        print(f"  Possible shape: {x_possible.shape}, dtype: {x_possible.dtype}")
        print(f"  Impossible shape: {x_impossible.shape}, dtype: {x_impossible.dtype}")
        
        return x_possible, x_impossible
    
    def run_contrastive_patching(self, x_possible, x_impossible, pair_name, pair_dir="O1_pairs",
                                block_indices=None, components=["full", "att", "mlp", "head"],
                                context_len=8):
        """
        Run contrastive activation patching between possible and impossible events
        
        Args:
            x_possible: Tensor for possible event
            x_impossible: Tensor for impossible event
            pair_name: Name of the pair for results
            pair_dir: Directory containing the pair (O1_pairs, O2_pairs, etc.)
            block_indices: List of block indices to patch
            components: List of components to patch ('full', 'att', 'mlp', 'head')
            context_len: Context length for masks
        """
        # Get total blocks if not specified
        if block_indices is None:
            num_blocks = len(self.model["encoder"].backbone.blocks)
            block_indices = list(range(num_blocks))
        
        # Setup for results
        results = {
            "pair_name": pair_name,
            "pair_dir": pair_dir,
            "context_len": context_len,
            "block_indices": block_indices,
            "components": components,
            "patching_results": {},
            "baseline": {}
        }
        
        # Get number of attention heads if patching heads
        if "head" in components:
            num_heads = self.patcher.num_heads
            print(f"Will patch {num_heads} attention heads per block")
        else:
            num_heads = 0
        
        # Run baseline forward passes (without patching)
        print("Running baseline forward passes...")
        with torch.no_grad():
            possible_loss = self.patcher.run_forward_pass(x_possible, context_len)
            impossible_loss = self.patcher.run_forward_pass(x_impossible, context_len)
        
        results["baseline"] = {
            "possible": possible_loss,
            "impossible": impossible_loss
        }
        
        print(f"Baseline losses: Possible = {possible_loss:.6f}, Impossible = {impossible_loss:.6f}")
        print(f"Expected effect of patching: {abs(impossible_loss - possible_loss):.6f}")
        
        # Run contrastive patching experiments
        for component in components:
            print(f"\nRunning contrastive patching for component: {component}")
            
            # Initialize results for this component
            results["patching_results"][component] = {
                "possible_to_impossible": [],
                "impossible_to_possible": []
            }
            
            # First pass: collect activations from both scenarios
            print("Collecting activations from both scenarios...")
            
            # For head patching, we only need attention component
            components_to_hook = [component] if component != "head" else ["att"]
            
            # Register hooks for all blocks
            self.patcher.register_hooks(block_indices, components_to_hook)
            
            # Run forward pass on possible scenario to collect activations
            _ = self.patcher.run_forward_pass(x_possible, context_len)
            possible_activations = copy.deepcopy(self.patcher.stored_activations)
            
            # Run forward pass on impossible scenario to collect activations
            _ = self.patcher.run_forward_pass(x_impossible, context_len)
            impossible_activations = copy.deepcopy(self.patcher.stored_activations)
            
            # Clean up hooks
            self.patcher.remove_hooks()
            
            # Second pass: patch activations one block at a time
            print("Patching activations one block at a time...")
            
            # Check if we're patching attention heads or regular components
            if component == "head":
                # For head patching, we need to loop through blocks and heads
                # Initialize head-specific results
                results["patching_results"]["head"] = {
                    "possible_to_impossible": [],
                    "impossible_to_possible": []
                }
                
                print(f"Using forward method patching for attention heads")
                print(f"This implements true contrastive patching by replacing the attention module's output")
                
                # Loop through each block and head
                for block_idx in tqdm(block_indices, desc=f"Patching attention heads"):
                    # First check if we have the attention activations for this block
                    if (block_idx, "att") not in possible_activations or (block_idx, "att") not in impossible_activations:
                        print(f"  Missing attention activations for block {block_idx}")
                        continue
                        
                    for head_idx in range(num_heads):
                        # For possible → impossible patching
                        try:
                            # Use attention activations for the block (we don't need head-specific activations)
                            source_activations = impossible_activations[(block_idx, "att")]
                                
                            # Run head patching using QKV weight modification
                            poss_to_imposs = self.patcher.patch_activations(
                                x_possible, block_idx, "head", 
                                source_activations,
                                context_len, head_idx
                            )
                            
                            # Calculate effect
                            effect_poss_to_imposs = poss_to_imposs - possible_loss
                            # Use absolute difference for proper normalization
                            abs_diff = abs(impossible_loss - possible_loss)
                            explanation_ratio = effect_poss_to_imposs / abs_diff if abs_diff > 0 else 0.0
                            
                            results["patching_results"]["head"]["possible_to_impossible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "patched_loss": poss_to_imposs,
                                "effect": effect_poss_to_imposs,
                                "explanation_ratio": explanation_ratio.item() if isinstance(explanation_ratio, torch.Tensor) else explanation_ratio
                            })
                            
                            # Only print for significant effects to reduce output
                            if abs(explanation_ratio) > 0.05:
                                print(f"  Block {block_idx}, Head {head_idx} possible→impossible: "
                                    f"loss={poss_to_imposs:.6f}, effect={effect_poss_to_imposs:.6f}, "
                                    f"ratio={explanation_ratio:.2f}")
                        except Exception as e:
                            print(f"  Error in possible→impossible patching block {block_idx}, head {head_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # For impossible → possible patching
                        try:
                            # Use attention activations for the block (we don't need head-specific activations)
                            source_activations = possible_activations[(block_idx, "att")]
                                
                            # Run head patching using QKV weight modification
                            imposs_to_poss = self.patcher.patch_activations(
                                x_impossible, block_idx, "head", 
                                source_activations,
                                context_len, head_idx
                            )
                            
                            # Calculate effect
                            effect_imposs_to_poss = imposs_to_poss - impossible_loss
                            # Use absolute difference for proper normalization
                            abs_diff = abs(impossible_loss - possible_loss)
                            explanation_ratio = -effect_imposs_to_poss / abs_diff if abs_diff > 0 else 0.0
                            
                            results["patching_results"]["head"]["impossible_to_possible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "patched_loss": imposs_to_poss,
                                "effect": effect_imposs_to_poss,
                                "explanation_ratio": explanation_ratio.item() if isinstance(explanation_ratio, torch.Tensor) else explanation_ratio
                            })
                            
                            # Only print for significant effects to reduce output
                            if abs(explanation_ratio) > 0.05:
                                print(f"  Block {block_idx}, Head {head_idx} impossible→possible: "
                                    f"loss={imposs_to_poss:.6f}, effect={effect_imposs_to_poss:.6f}, "
                                    f"ratio={explanation_ratio:.2f}")
                        except Exception as e:
                            print(f"  Error in impossible→possible patching block {block_idx}, head {head_idx}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # Standard component patching (not attention heads)
                # Loop through each block
                for block_idx in tqdm(block_indices, desc=f"Patching {component}"):
                    # Patch possible → impossible (what makes the model detect impossibility)
                    try:
                        key = (block_idx, component)
                        if key not in impossible_activations:
                            print(f"  Missing activations for block {block_idx}, component {component}")
                            continue
                            
                        poss_to_imposs = self.patcher.patch_activations(
                            x_possible, block_idx, component, 
                            impossible_activations[key],
                            context_len
                        )
                        
                        # Calculate effect (positive means moving toward impossible)
                        effect_poss_to_imposs = poss_to_imposs - possible_loss
                        abs_diff = abs(impossible_loss - possible_loss)
                        explanation_ratio = effect_poss_to_imposs / abs_diff if abs_diff > 0 else 0.0
                        
                        results["patching_results"][component]["possible_to_impossible"].append({
                            "block_idx": block_idx,
                            "patched_loss": poss_to_imposs,
                            "effect": effect_poss_to_imposs,
                            "explanation_ratio": explanation_ratio.item() if isinstance(explanation_ratio, torch.Tensor) else explanation_ratio
                        })
                        
                        print(f"  Block {block_idx} possible→impossible: loss={poss_to_imposs:.6f}, effect={effect_poss_to_imposs:.6f}, ratio={explanation_ratio:.2f}")
                    except Exception as e:
                        print(f"  Error in possible→impossible patching block {block_idx}: {e}")
                        # Add an empty result to avoid "No data for component" error
                        results["patching_results"][component]["possible_to_impossible"].append({
                            "block_idx": block_idx,
                            "patched_loss": possible_loss,  # Just use baseline
                            "effect": 0.0,  # No effect
                            "explanation_ratio": 0.0
                        })
                    
                    # Patch impossible → possible (what makes the model detect possibility)
                    try:
                        key = (block_idx, component)
                        if key not in possible_activations:
                            continue
                            
                        imposs_to_poss = self.patcher.patch_activations(
                            x_impossible, block_idx, component, 
                            possible_activations[key],
                            context_len
                        )
                        
                        # Calculate effect (negative means moving toward possible)
                        effect_imposs_to_poss = imposs_to_poss - impossible_loss
                        abs_diff = abs(impossible_loss - possible_loss)
                        explanation_ratio = -effect_imposs_to_poss / abs_diff if abs_diff > 0 else 0.0
                        
                        results["patching_results"][component]["impossible_to_possible"].append({
                            "block_idx": block_idx,
                            "patched_loss": imposs_to_poss,
                            "effect": effect_imposs_to_poss,
                            "explanation_ratio": explanation_ratio.item() if isinstance(explanation_ratio, torch.Tensor) else explanation_ratio
                        })
                        
                        print(f"  Block {block_idx} impossible→possible: loss={imposs_to_poss:.6f}, effect={effect_imposs_to_poss:.6f}, ratio={explanation_ratio:.2f}")
                    except Exception as e:
                        print(f"  Error in impossible→possible patching block {block_idx}: {e}")
                        # Add an empty result to avoid "No data for component" error
                        results["patching_results"][component]["impossible_to_possible"].append({
                            "block_idx": block_idx,
                            "patched_loss": impossible_loss,  # Just use baseline
                            "effect": 0.0,  # No effect
                            "explanation_ratio": 0.0
                        })
        
        # Store results
        self.results[f"{pair_name}_{context_len}"] = results
        
        return results
    
    def plot_contrastive_results(self, pair_name, context_len=8, save_prefix=None):
        """
        Plot contrastive patching results
        
        Args:
            pair_name: Name of the pair to plot
            context_len: Context length used in the experiment
            save_prefix: Prefix for saving plots
        """
        # Check if results exist
        key = f"{pair_name}_{context_len}"
        if key not in self.results:
            print(f"No results found for {key}")
            return
        
        results = self.results[key]
        components = results["patching_results"].keys()
        pair_dir_source = results.get("pair_dir", "O1_pairs")  # Default to O1_pairs if not specified
        
        # Create output directory for this pair
        pair_output_dir = os.path.join(self.output_dir, f"{pair_dir_source}/{pair_name}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        # Plot for each component
        for component in components:
            # Extract data
            p2i_data = results["patching_results"][component]["possible_to_impossible"]
            i2p_data = results["patching_results"][component]["impossible_to_possible"]
            
            # Check if we have data
            if len(p2i_data) == 0 or len(i2p_data) == 0:
                print(f"No data for component {component}")
                continue
            
            # Extract block indices and effects
            block_indices = [d["block_idx"] for d in p2i_data]
            p2i_effects = [d["effect"] for d in p2i_data]
            i2p_effects = [d["effect"] for d in i2p_data]
            p2i_explanation = [d["explanation_ratio"] for d in p2i_data]
            i2p_explanation = [d["explanation_ratio"] for d in i2p_data]
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Create three subplots
            plt.subplot(2, 1, 1)
            
            # Plot effects (raw change in loss)
            plt.plot(block_indices, p2i_effects, 'r-', label='Possible → Impossible', alpha=0.8, marker='o')
            plt.plot(block_indices, i2p_effects, 'b-', label='Impossible → Possible', alpha=0.8, marker='o')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Find max effect blocks
            if len(p2i_effects) > 0:
                max_p2i_idx = p2i_effects.index(max(p2i_effects))
                max_p2i_block = block_indices[max_p2i_idx]
                plt.scatter([max_p2i_block], [p2i_effects[max_p2i_idx]], color='darkred', s=150, zorder=5)
                plt.annotate(f'Block {max_p2i_block}: {p2i_effects[max_p2i_idx]:.4f}', 
                            (max_p2i_block, p2i_effects[max_p2i_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
            
            if len(i2p_effects) > 0:
                min_i2p_idx = i2p_effects.index(min(i2p_effects))
                min_i2p_block = block_indices[min_i2p_idx]
                plt.scatter([min_i2p_block], [i2p_effects[min_i2p_idx]], color='darkblue', s=150, zorder=5)
                plt.annotate(f'Block {min_i2p_block}: {i2p_effects[min_i2p_idx]:.4f}', 
                            (min_i2p_block, i2p_effects[min_i2p_idx]),
                            xytext=(10, -25), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
            
            # Add labels and title
            plt.xlabel('Block Index', fontsize=12)
            plt.ylabel('Effect on Loss', fontsize=12)
            plt.title(f'Contrastive Patching: Raw Effect on Loss\nPair: {pair_name}, Component: {component}, Context: {context_len}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
            # Second subplot for explanation ratio
            plt.subplot(2, 1, 2)
            
            # Plot explanation ratio
            plt.bar(
                [i - 0.2 for i in block_indices],
                p2i_explanation,
                width=0.4,
                alpha=0.7,
                color='r',
                label='Possible → Impossible'
            )
            
            plt.bar(
                [i + 0.2 for i in block_indices],
                i2p_explanation,
                width=0.4,
                alpha=0.7,
                color='b',
                label='Impossible → Possible'
            )
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='100% Explanation')
            
            # Find max explanation blocks
            if len(p2i_explanation) > 0:
                max_p2i_exp_idx = p2i_explanation.index(max(p2i_explanation))
                max_p2i_exp_block = block_indices[max_p2i_exp_idx]
                plt.scatter([max_p2i_exp_block - 0.2], [p2i_explanation[max_p2i_exp_idx]], color='darkred', s=150, zorder=5)
                plt.annotate(f'Block {max_p2i_exp_block}: {p2i_explanation[max_p2i_exp_idx]:.2f}x', 
                            (max_p2i_exp_block - 0.2, p2i_explanation[max_p2i_exp_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
            
            if len(i2p_explanation) > 0:
                max_i2p_exp_idx = i2p_explanation.index(max(i2p_explanation))
                max_i2p_exp_block = block_indices[max_i2p_exp_idx]
                plt.scatter([max_i2p_exp_block + 0.2], [i2p_explanation[max_i2p_exp_idx]], color='darkblue', s=150, zorder=5)
                plt.annotate(f'Block {max_i2p_exp_block}: {i2p_explanation[max_i2p_exp_idx]:.2f}x', 
                            (max_i2p_exp_block + 0.2, i2p_explanation[max_i2p_exp_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
            
            # Add labels and title
            plt.xlabel('Block Index', fontsize=12)
            plt.ylabel('Explanation Ratio', fontsize=12)
            plt.title('Contrastive Patching: Explanation Ratio (how much of the difference is explained)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
            # Add tight layout
            plt.tight_layout()
            
            # Create save filename
            if save_prefix:
                save_name = f"{save_prefix}_{component}_ctx{context_len}.png"
            else:
                save_name = f"contrastive_patching_{component}_ctx{context_len}.png"
            
            save_path = os.path.join(pair_output_dir, save_name)
            
            # Save figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
            plt.close()
    
    def plot_attention_heads(self, pair_name, context_len=8, save_prefix=None):
        """
        Plot attention head patching results as a heatmap
        
        Args:
            pair_name: Name of the pair to plot
            context_len: Context length used in the experiment
            save_prefix: Prefix for saving plots
        """
        # Check if results exist
        key = f"{pair_name}_{context_len}"
        if key not in self.results:
            print(f"No results found for {key}")
            return
        
        results = self.results[key]
        pair_dir_source = results.get("pair_dir", "O1_pairs")  # Default to O1_pairs if not specified
        
        # Check if head results exist
        if "head" not in results["patching_results"]:
            print(f"No attention head patching results found for {key}")
            return
        
        # Create output directory for this pair
        pair_output_dir = os.path.join(self.output_dir, f"{pair_dir_source}/{pair_name}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        # Extract head patching data
        head_results = results["patching_results"]["head"]
        
        # Extract data
        p2i_data = head_results["possible_to_impossible"]
        i2p_data = head_results["impossible_to_possible"]
        
        if len(p2i_data) == 0 or len(i2p_data) == 0:
            print(f"No attention head patching data found for {key}")
            return
        
        # Get unique block and head indices
        block_indices = sorted(list(set([d["block_idx"] for d in p2i_data])))
        head_indices = sorted(list(set([d["head_idx"] for d in p2i_data])))
        num_blocks = len(block_indices)
        num_heads = len(head_indices)
        
        # Create matrices for heatmaps
        p2i_matrix = np.zeros((num_blocks, num_heads))
        i2p_matrix = np.zeros((num_blocks, num_heads))
        
        # Fill matrices with explanation ratios
        for data in p2i_data:
            block_idx = data["block_idx"]
            head_idx = data["head_idx"]
            block_pos = block_indices.index(block_idx)
            head_pos = head_indices.index(head_idx)
            p2i_matrix[block_pos, head_pos] = data["explanation_ratio"]
        
        for data in i2p_data:
            block_idx = data["block_idx"]
            head_idx = data["head_idx"]
            block_pos = block_indices.index(block_idx)
            head_pos = head_indices.index(head_idx)
            i2p_matrix[block_pos, head_pos] = data["explanation_ratio"]
        
        # Plot heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Possible to Impossible heatmap
        im0 = axes[0].imshow(p2i_matrix, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[0].set_title('Attention Heads: Possible → Impossible\n(higher values = more important for detecting impossibility)', fontsize=14)
        axes[0].set_xlabel('Head Index', fontsize=12)
        axes[0].set_ylabel('Block Index', fontsize=12)
        axes[0].set_xticks(np.arange(num_heads))
        axes[0].set_yticks(np.arange(num_blocks))
        axes[0].set_xticklabels(head_indices)
        axes[0].set_yticklabels(block_indices)
        plt.colorbar(im0, ax=axes[0], label='Explanation Ratio')
        
        # Impossible to Possible heatmap
        im1 = axes[1].imshow(i2p_matrix, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[1].set_title('Attention Heads: Impossible → Possible\n(higher values = more important for detecting possibility)', fontsize=14)
        axes[1].set_xlabel('Head Index', fontsize=12)
        axes[1].set_ylabel('Block Index', fontsize=12)
        axes[1].set_xticks(np.arange(num_heads))
        axes[1].set_yticks(np.arange(num_blocks))
        axes[1].set_xticklabels(head_indices)
        axes[1].set_yticklabels(block_indices)
        plt.colorbar(im1, ax=axes[1], label='Explanation Ratio')
        
        # Add text annotations to the heatmap cells
        threshold = 0.2  # Only label cells with significant values
        for i in range(num_blocks):
            for j in range(num_heads):
                if abs(p2i_matrix[i, j]) > threshold:
                    axes[0].text(j, i, f"{p2i_matrix[i, j]:.2f}", 
                                ha="center", va="center", 
                                color="white" if abs(p2i_matrix[i, j]) > 0.3 else "black",
                                fontsize=8)
                if abs(i2p_matrix[i, j]) > threshold:
                    axes[1].text(j, i, f"{i2p_matrix[i, j]:.2f}", 
                                ha="center", va="center", 
                                color="white" if abs(i2p_matrix[i, j]) > 0.3 else "black",
                                fontsize=8)
        
        # Add overall title
        fig.suptitle(f'Attention Head Patching Results: {pair_name}, Context: {context_len}', fontsize=16)
        
        # Add tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create save filename
        if save_prefix:
            save_name = f"{save_prefix}_head_heatmap_ctx{context_len}.png"
        else:
            save_name = f"contrastive_patching_head_heatmap_ctx{context_len}.png"
        
        save_path = os.path.join(pair_output_dir, save_name)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention head heatmap to {save_path}")
        
        plt.close()
    
    def save_results(self, pair_name, context_len=8, save_prefix=None):
        """Save patching results to disk"""
        # Check if results exist
        key = f"{pair_name}_{context_len}"
        if key not in self.results:
            print(f"No results found for {key}")
            return
        
        results = self.results[key]
        pair_dir_source = results.get("pair_dir", "O1_pairs")  # Default to O1_pairs if not specified
        
        # Create output directory for this pair
        pair_output_dir = os.path.join(self.output_dir, f"{pair_dir_source}/{pair_name}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        # Create save filename
        if save_prefix:
            save_name = f"{save_prefix}_ctx{context_len}_results.pth"
        else:
            save_name = f"contrastive_patching_ctx{context_len}_results.pth"
        
        save_path = os.path.join(pair_output_dir, save_name)
        
        # Save results
        torch.save(results, save_path)
        print(f"Saved results to {save_path}")


def main():
    """
    Main function to run contrastive activation patching
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Contrastive activation patching for intuitive physics tasks')
    parser.add_argument('--config', type=str, 
                        default='evaluation_code/evals/intuitive_physics/configs/default_intphys.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, 
                        default='contrastive_patching_results',
                        help='Output directory')
    parser.add_argument('--pair_path', type=str, 
                        default='O1_pairs',
                        help='Path to directory containing pairs (e.g., O1_pairs, O2_pairs)')
    parser.add_argument('--pair_prefix', type=str, 
                        default='pair_1_01',
                        help='Prefix for pair files')
    parser.add_argument('--components', type=str, 
                        nargs='+',
                        default=['full', 'att', 'mlp'],
                        choices=['full', 'att', 'mlp', 'head'],
                        help='Components to patch (full=entire block, att=attention, mlp=MLP, head=individual attention heads)')
    parser.add_argument('--context_lengths', type=int, 
                        nargs='+',
                        default=[8],
                        help='Context lengths to evaluate')
    # Alias for context_lengths for backward compatibility
    parser.add_argument('--context', type=int, 
                        help='Alias for --context_lengths (accepts a single value)')
    parser.add_argument('--block_subset', type=str,
                        default='all',
                        choices=['all', 'early', 'middle', 'late', 'spread'],
                        help='Which subset of blocks to patch')
    parser.add_argument('--max_blocks', type=int,
                        default=32,
                        help='Maximum number of blocks to patch')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Initialize contrastive patching
    patcher = ContrastivePatching(
        config_path=args.config,
        output_dir=args.output_dir,
        device=device
    )
    
    # Get total number of blocks
    num_blocks = len(patcher.model["encoder"].backbone.blocks)
    print(f"Model has {num_blocks} transformer blocks")
    
    # Select blocks to patch based on argument
    if args.block_subset == 'all':
        block_indices = list(range(num_blocks))
    elif args.block_subset == 'early':
        end_idx = num_blocks // 3
        block_indices = list(range(0, end_idx))
    elif args.block_subset == 'middle':
        start_idx = num_blocks // 3
        end_idx = 2 * (num_blocks // 3)
        block_indices = list(range(start_idx, end_idx))
    elif args.block_subset == 'late':
        start_idx = 2 * (num_blocks // 3)
        block_indices = list(range(start_idx, num_blocks))
    elif args.block_subset == 'spread':
        # Evenly spread blocks throughout model
        step = max(1, num_blocks // args.max_blocks)
        block_indices = list(range(0, num_blocks, step))
    
    # Limit to max_blocks
    if args.max_blocks > 0 and len(block_indices) > args.max_blocks:
        # Ensure we include first and last blocks in selection
        if 0 not in block_indices[:args.max_blocks]:
            block_indices = [0] + block_indices[1:args.max_blocks]
        if (num_blocks - 1) not in block_indices[:args.max_blocks]:
            block_indices = block_indices[:args.max_blocks-1] + [num_blocks - 1]
        else:
            block_indices = block_indices[:args.max_blocks]
    
    print(f"Selected {len(block_indices)} blocks to patch: {block_indices}")
    
    # Load pair
    x_possible, x_impossible = patcher.load_pair(args.pair_path, args.pair_prefix)
    
    # Handle both --context and --context_lengths options
    if args.context is not None:
        # If --context was used, make a list with just that value
        context_lengths = [args.context]
    else:
        # Otherwise use the context_lengths list
        context_lengths = args.context_lengths

    # Run contrastive patching for each context length
    for context_len in context_lengths:
        print(f"\nRunning contrastive patching with context length {context_len}")
        
        # Run patching
        results = patcher.run_contrastive_patching(
            x_possible=x_possible,
            x_impossible=x_impossible,
            pair_name=args.pair_prefix,
            pair_dir=os.path.basename(args.pair_path),  # Extract just the directory name (O1_pairs, O2_pairs)
            block_indices=block_indices,
            components=args.components,
            context_len=context_len
        )
        
        # Plot standard results
        patcher.plot_contrastive_results(
            pair_name=args.pair_prefix,
            context_len=context_len
        )
        
        # Plot attention head results if we patched heads
        if "head" in args.components:
            patcher.plot_attention_heads(
                pair_name=args.pair_prefix,
                context_len=context_len
            )
        
        # Save results
        patcher.save_results(
            pair_name=args.pair_prefix,
            context_len=context_len
        )
    
    print("\nContrastive patching completed!")


def process_all_pairs():
    """
    Process all pairs in the specified directory
    """
    import glob
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process all pairs for contrastive activation patching')
    parser.add_argument('--config', type=str, 
                        default='evaluation_code/evals/intuitive_physics/configs/default_intphys.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, 
                        default='contrastive_patching_results',
                        help='Output directory')
    parser.add_argument('--pairs_dir', type=str, 
                        default='O1_pairs',
                        help='Directory containing pairs')
    parser.add_argument('--components', type=str, 
                        nargs='+',
                        default=['full'],
                        choices=['full', 'att', 'mlp', 'head'],
                        help='Components to patch (full=entire block, att=attention, mlp=MLP, head=individual attention heads)')
    parser.add_argument('--context_length', type=int, 
                        default=8,
                        help='Context length to use')
    parser.add_argument('--max_blocks', type=int,
                        default=32,
                        help='Maximum number of blocks to patch')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Find all possible pairs
    possible_files = glob.glob(os.path.join(args.pairs_dir, "*_possible.pt"))
    pair_prefixes = [os.path.basename(f).replace("_possible.pt", "") for f in possible_files]
    pair_prefixes.sort()
    
    print(f"Found {len(pair_prefixes)} pairs:")
    for prefix in pair_prefixes:
        print(f"  - {prefix}")
    
    # Process each pair
    for prefix in pair_prefixes:
        print(f"\n{'='*50}")
        print(f"Processing pair: {prefix}")
        print(f"{'='*50}\n")
        
        # Set up sys.argv for main
        sys.argv = [
            sys.argv[0],
            "--config", args.config,
            "--output_dir", args.output_dir,
            "--pair_path", args.pairs_dir,
            "--pair_prefix", prefix,
            "--components", *args.components,
            "--context_lengths", str(args.context_length),
            "--max_blocks", str(args.max_blocks),
            "--device", args.device
        ]
        
        # Run main for this pair
        main()


if __name__ == "__main__":
    main()
    
    # To process all pairs, uncomment this line
    # process_all_pairs()