#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Add parent directory to path to import modules
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


class AblationPatcher:
    """
    Class for ablating model components to analyze their importance
    """
    def __init__(self, model):
        self.model = model
        self.encoder = model["encoder"]
        self.target_encoder = model["target_encoder"]
        self.predictor = model["predictor"]
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
            # For head, we return the attention module, but will handle head-specific logic separately
            if head_idx is None:
                raise ValueError("head_idx must be provided for component='head'")
            return self.encoder.backbone.blocks[block_idx].attn
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def run_forward_pass(self, x, context_len=8):
        """Run a forward pass and return the prediction loss"""
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
    
    def ablate_component(self, x, block_idx, component, context_len=8, head_idx=None):
        """
        Run forward pass with a component ablated (zeroed out)
        
        Args:
            x: Input tensor
            block_idx: Block index to ablate
            component: Component to ablate ('full', 'att', 'mlp', or 'head')
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
        
        # Store original state to restore later
        original_states = {}
        
        if component == "head":
            # For head ablation, we'll modify the QKV weights to zero out the specific head
            attn_module = self._get_block_module(block_idx, "att")
            
            # Save original QKV weight and bias
            original_states['qkv_weight'] = attn_module.qkv.weight.data.clone()
            if hasattr(attn_module.qkv, 'bias') and attn_module.qkv.bias is not None:
                original_states['qkv_bias'] = attn_module.qkv.bias.data.clone()
                
            # Create copies of weights that we'll modify
            modified_qkv_weight = original_states['qkv_weight'].clone()
            
            # If we have bias, also create a copy
            modified_qkv_bias = None
            if 'qkv_bias' in original_states:
                modified_qkv_bias = original_states['qkv_bias'].clone()
            
            # Ablate the specific head by zeroing out its parameters
            for qkv_idx in range(3):  # q, k, v
                # Calculate slices for this head's parameters
                start_idx = qkv_idx * self.embed_dim + head_idx * self.head_dim
                end_idx = start_idx + self.head_dim
                
                # Zero out this head's parameters
                modified_qkv_weight[start_idx:end_idx, :] = 0.0
                
                # Also zero out bias if it exists
                if modified_qkv_bias is not None:
                    modified_qkv_bias[start_idx:end_idx] = 0.0
            
            # Apply modified weights
            attn_module.qkv.weight.data = modified_qkv_weight
            if modified_qkv_bias is not None:
                attn_module.qkv.bias.data = modified_qkv_bias
                
        else:
            # For other components, we'll patch the forward method
            module = self._get_block_module(block_idx, component)
            
            # Store the original forward method
            original_states['forward'] = module.forward
            
            # Create a patched forward that returns zeros
            def zero_forward(*args, **kwargs):
                """Return zeros in the same shape as the original output"""
                # Run original to get the output shape
                with torch.no_grad():
                    original_output = original_states['forward'](*args, **kwargs)
                
                # If output is a tuple, zero out each element
                if isinstance(original_output, tuple):
                    result = []
                    for x in original_output:
                        if x is not None:
                            result.append(torch.zeros_like(x))
                        else:
                            result.append(None)
                    return tuple(result)
                elif original_output is None:
                    # If original output is None, return None
                    return None
                else:
                    # Otherwise just return zeros in the same shape
                    return torch.zeros_like(original_output)
            
            # Replace the forward method
            module.forward = zero_forward
        
        try:
            with torch.no_grad():
                # Run forward pass with the component ablated
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
            # Restore original state
            if component == "head":
                attn_module = self._get_block_module(block_idx, "att")
                
                # Restore original QKV weight and bias
                if 'qkv_weight' in original_states:
                    attn_module.qkv.weight.data = original_states['qkv_weight']
                if 'qkv_bias' in original_states:
                    attn_module.qkv.bias.data = original_states['qkv_bias']
            else:
                module = self._get_block_module(block_idx, component)
                
                # Restore original forward method
                if 'forward' in original_states:
                    module.forward = original_states['forward']
            
        return loss.item()


class AblationPatching:
    """
    Class for running activation ablation patching experiments
    """
    def __init__(self, config_path, output_dir="ablation_patching_results", device=None):
        """
        Initialize the ablation patching experiment
        
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
        
        # Initialize ablation patcher
        self.patcher = AblationPatcher(self.model)
        
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
    
    def run_ablation_patching(self, x_possible, x_impossible, pair_name, pair_dir="O1_pairs",
                            block_indices=None, components=["full", "att", "mlp", "head"],
                            context_len=8):
        """
        Run ablation patching on both possible and impossible events
        
        Args:
            x_possible: Tensor for possible event
            x_impossible: Tensor for impossible event
            pair_name: Name of the pair for results
            pair_dir: Directory containing the pair (O1_pairs, O2_pairs, etc.)
            block_indices: List of block indices to ablate
            components: List of components to ablate ('full', 'att', 'mlp', 'head')
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
            "ablation_results": {},
            "baseline": {}
        }
        
        # Get number of attention heads if patching heads
        if "head" in components:
            num_heads = self.patcher.num_heads
            print(f"Will ablate {num_heads} attention heads per block")
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
        
        # Run ablation experiments for each component
        for component in components:
            print(f"\nRunning ablation for component: {component}")
            
            # Initialize results for this component
            results["ablation_results"][component] = {
                "possible": [],
                "impossible": []
            }
            
            # Check if we're ablating attention heads or regular components
            if component == "head":
                print(f"Using weight modification for head ablation")
                
                # Loop through each block and head
                for block_idx in tqdm(block_indices, desc=f"Ablating attention heads"):
                    for head_idx in range(num_heads):
                        # Ablate head in possible scenario
                        try:
                            try:
                                ablated_poss = self.patcher.ablate_component(
                                    x_possible, block_idx, "head", 
                                    context_len, head_idx
                                )
                                
                                # Calculate effect
                                effect_poss = ablated_poss - possible_loss
                                
                                # Only print for significant effects
                                if abs(effect_poss) > 0.001:
                                    print(f"  Block {block_idx}, Head {head_idx} possible: "
                                        f"loss={ablated_poss:.6f}, effect={effect_poss:.6f}")
                            except Exception as e:
                                print(f"  Error in ablating possible, block {block_idx}, head {head_idx}: {e}")
                                # Fallback to baseline on error
                                ablated_poss = possible_loss
                                effect_poss = 0.0
                                
                            # Add results regardless of whether the ablation succeeded
                            results["ablation_results"]["head"]["possible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "ablated_loss": ablated_poss,
                                "effect": effect_poss
                            })
                        except Exception as e:
                            print(f"  Critical error recording head results: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Add placeholder as a last resort
                            results["ablation_results"]["head"]["possible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "ablated_loss": possible_loss,
                                "effect": 0.0
                            })
                        
                        # Ablate head in impossible scenario
                        try:
                            try:
                                ablated_imposs = self.patcher.ablate_component(
                                    x_impossible, block_idx, "head", 
                                    context_len, head_idx
                                )
                                
                                # Calculate effect
                                effect_imposs = ablated_imposs - impossible_loss
                                
                                # Only print for significant effects
                                if abs(effect_imposs) > 0.001:
                                    print(f"  Block {block_idx}, Head {head_idx} impossible: "
                                        f"loss={ablated_imposs:.6f}, effect={effect_imposs:.6f}")
                            except Exception as e:
                                print(f"  Error in ablating impossible, block {block_idx}, head {head_idx}: {e}")
                                # Fallback to baseline on error
                                ablated_imposs = impossible_loss
                                effect_imposs = 0.0
                                
                            # Add results regardless of whether the ablation succeeded
                            results["ablation_results"]["head"]["impossible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "ablated_loss": ablated_imposs,
                                "effect": effect_imposs
                            })
                        except Exception as e:
                            print(f"  Critical error recording head results: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Add placeholder as a last resort
                            results["ablation_results"]["head"]["impossible"].append({
                                "block_idx": block_idx,
                                "head_idx": head_idx,
                                "ablated_loss": impossible_loss,
                                "effect": 0.0
                            })
            else:
                # Standard component ablation (full, att, mlp)
                for block_idx in tqdm(block_indices, desc=f"Ablating {component}"):
                    # Ablate component in possible scenario
                    try:
                        try:
                            ablated_poss = self.patcher.ablate_component(
                                x_possible, block_idx, component, 
                                context_len
                            )
                            
                            # Calculate effect
                            effect_poss = ablated_poss - possible_loss
                            
                            print(f"  Block {block_idx} possible: loss={ablated_poss:.6f}, effect={effect_poss:.6f}")
                        except Exception as e:
                            print(f"  Error in ablating possible, block {block_idx}: {e}")
                            # Use baseline value on error
                            ablated_poss = possible_loss
                            effect_poss = 0.0
                            
                        # Add results regardless of whether the ablation succeeded
                        results["ablation_results"][component]["possible"].append({
                            "block_idx": block_idx,
                            "ablated_loss": ablated_poss,
                            "effect": effect_poss
                        })
                    except Exception as e:
                        print(f"  Critical error in recording possible results, block {block_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Add placeholder for plotting as a fallback
                        results["ablation_results"][component]["possible"].append({
                            "block_idx": block_idx,
                            "ablated_loss": possible_loss,
                            "effect": 0.0
                        })
                    
                    # Ablate component in impossible scenario
                    try:
                        try:
                            ablated_imposs = self.patcher.ablate_component(
                                x_impossible, block_idx, component, 
                                context_len
                            )
                            
                            # Calculate effect
                            effect_imposs = ablated_imposs - impossible_loss
                            
                            print(f"  Block {block_idx} impossible: loss={ablated_imposs:.6f}, effect={effect_imposs:.6f}")
                        except Exception as e:
                            print(f"  Error in ablating impossible, block {block_idx}: {e}")
                            # Use baseline value on error
                            ablated_imposs = impossible_loss
                            effect_imposs = 0.0
                            
                        # Add results regardless of whether the ablation succeeded
                        results["ablation_results"][component]["impossible"].append({
                            "block_idx": block_idx,
                            "ablated_loss": ablated_imposs,
                            "effect": effect_imposs
                        })
                    except Exception as e:
                        print(f"  Critical error in recording impossible results, block {block_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Add placeholder for plotting as a fallback
                        results["ablation_results"][component]["impossible"].append({
                            "block_idx": block_idx,
                            "ablated_loss": impossible_loss,
                            "effect": 0.0
                        })
        
        # Store results
        self.results[f"{pair_name}_{context_len}"] = results
        
        return results
    
    def plot_ablation_results(self, pair_name, component, context_len=8, save_prefix=None):
        """
        Plot ablation results for a specific component
        
        Args:
            pair_name: Name of the pair to plot
            component: Component to plot ('full', 'att', 'mlp')
            context_len: Context length used in the experiment
            save_prefix: Prefix for saving plots
        """
        # Check if results exist
        key = f"{pair_name}_{context_len}"
        if key not in self.results:
            print(f"No results found for {key}")
            return
        
        results = self.results[key]
        pair_dir_source = results.get("pair_dir", "O1_pairs")
        
        # Create output directory
        pair_output_dir = os.path.join(self.output_dir, f"{pair_dir_source}/{pair_name}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        # Check if component exists
        if component not in results["ablation_results"]:
            print(f"No data for component {component}")
            return
        
        # Extract data
        poss_data = results["ablation_results"][component]["possible"]
        imposs_data = results["ablation_results"][component]["impossible"]
        
        # Check if we have data
        if len(poss_data) == 0 or len(imposs_data) == 0:
            print(f"No data for component {component}")
            return
        
        # Extract block indices and effects
        block_indices = [d["block_idx"] for d in poss_data]
        if component == "head":
            head_indices = sorted(list(set([d["head_idx"] for d in poss_data])))
            num_blocks = len(set(block_indices))
            num_heads = len(head_indices)
            
            # Create matrices for heatmaps
            poss_matrix = np.zeros((num_blocks, num_heads))
            imposs_matrix = np.zeros((num_blocks, num_heads))
            
            # Fill matrices with effects
            for i, data in enumerate(poss_data):
                block_idx = data["block_idx"]
                head_idx = data["head_idx"]
                block_pos = sorted(list(set(block_indices))).index(block_idx)
                head_pos = head_indices.index(head_idx)
                poss_matrix[block_pos, head_pos] = data["effect"]
            
            for i, data in enumerate(imposs_data):
                block_idx = data["block_idx"]
                head_idx = data["head_idx"]
                block_pos = sorted(list(set(block_indices))).index(block_idx)
                head_pos = head_indices.index(head_idx)
                imposs_matrix[block_pos, head_pos] = data["effect"]
            
            # Compute effect difference (importance for physical understanding)
            diff_matrix = imposs_matrix - poss_matrix
            
            # Calculate appropriate color scaling based on the data
            max_val = max(np.max(np.abs(poss_matrix)), np.max(np.abs(imposs_matrix)), np.max(np.abs(diff_matrix)))
            # Use a smaller minimum value to better visualize small effects
            vmin = -0.01
            vmax = 0.01
            
            if max_val > 0.005:  # If we have some significant effects
                vmin = -max_val * 1.2  # Give some margin
                vmax = max_val * 1.2
            
            print(f"Heatmap color range: {vmin:.4f} to {vmax:.4f}")
            
            # Plot heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # Possible scenario
            im0 = axes[0].imshow(poss_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[0].set_title('Attention Heads: Possible Event\n(effect of ablating each head)', fontsize=14)
            axes[0].set_xlabel('Head Index', fontsize=12)
            axes[0].set_ylabel('Block Index', fontsize=12)
            axes[0].set_xticks(np.arange(num_heads))
            axes[0].set_yticks(np.arange(num_blocks))
            axes[0].set_xticklabels(head_indices)
            axes[0].set_yticklabels(sorted(list(set(block_indices))))
            plt.colorbar(im0, ax=axes[0], label='Effect on Loss')
            
            # Impossible scenario
            im1 = axes[1].imshow(imposs_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[1].set_title('Attention Heads: Impossible Event\n(effect of ablating each head)', fontsize=14)
            axes[1].set_xlabel('Head Index', fontsize=12)
            axes[1].set_yticks([])  # Hide y-ticks for middle plot
            axes[1].set_xticks(np.arange(num_heads))
            axes[1].set_xticklabels(head_indices)
            plt.colorbar(im1, ax=axes[1], label='Effect on Loss')
            
            # Difference (indicates importance for physical understanding)
            im2 = axes[2].imshow(diff_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[2].set_title('Attention Heads: Diff (Impossible - Possible)\n(indicates importance for physical understanding)', fontsize=14)
            axes[2].set_xlabel('Head Index', fontsize=12)
            axes[2].set_yticks([])  # Hide y-ticks for right plot
            axes[2].set_xticks(np.arange(num_heads))
            axes[2].set_xticklabels(head_indices)
            plt.colorbar(im2, ax=axes[2], label='Loss Difference')
            
            # Find important heads (highest absolute difference)
            # Adjust threshold based on the data range
            threshold = max(0.001, max_val / 10)  # Use at least 0.001 or 10% of max value
            
            # Find top 5 most important cells in difference matrix
            flat_idx_diff = np.argsort(np.abs(diff_matrix).flatten())[-5:]  # Top 5 indices
            idx_diff = [(idx // num_heads, idx % num_heads) for idx in flat_idx_diff]
            
            # Annotate difference values
            for i in range(num_blocks):
                for j in range(num_heads):
                    if abs(diff_matrix[i, j]) > threshold:
                        color = "white" if abs(diff_matrix[i, j]) > threshold * 2 else "black"
                        axes[2].text(j, i, f"{diff_matrix[i, j]:.4f}", 
                                    ha="center", va="center", 
                                    color=color, fontsize=8)
            
            # Highlight the top cells with rectangles
            for i, j in idx_diff:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=2)
                axes[2].add_patch(rect)
                
            # Add text noting the most important heads
            sorted_block_indices = sorted(list(set(block_indices)))
            important_heads_text = "Most important heads:\n"
            for idx, (i, j) in enumerate(idx_diff):
                block = sorted_block_indices[i]
                important_heads_text += f"Block {block}, Head {j}: {diff_matrix[i, j]:.4f}\n"
                
            plt.figtext(0.5, 0.02, important_heads_text, 
                       ha='center', va='bottom', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            # Add overall title
            fig.suptitle(f'Ablation Results for Attention Heads: {pair_name}, Context: {context_len}', fontsize=16)
            
            # Create save filename
            if save_prefix:
                save_name = f"{save_prefix}_{component}_heatmap_ctx{context_len}.png"
            else:
                save_name = f"ablation_{component}_heatmap_ctx{context_len}.png"
                
        else:
            # Standard components (full, att, mlp)
            poss_effects = [d["effect"] for d in poss_data]
            imposs_effects = [d["effect"] for d in imposs_data]
            
            # Compute effect difference (importance for physical understanding)
            diff_effects = [i - p for i, p in zip(imposs_effects, poss_effects)]
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 18))
            
            # Plot effects for possible scenario
            axes[0].plot(block_indices, poss_effects, 'b-', linewidth=2.5, label='Effect on Possible', alpha=0.8)
            axes[0].scatter(block_indices, poss_effects, color='blue', s=50, zorder=5)
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Highlight important blocks
            max_poss_idx = poss_effects.index(max(poss_effects))
            max_poss_block = block_indices[max_poss_idx]
            axes[0].scatter([max_poss_block], [poss_effects[max_poss_idx]], color='darkblue', s=150, zorder=6)
            axes[0].annotate(f'Block {max_poss_block}: {poss_effects[max_poss_idx]:.4f}', 
                        (max_poss_block, poss_effects[max_poss_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
            
            axes[0].set_title(f'Ablation Effect on Possible Event\nComponent: {component}, Context: {context_len}', fontsize=14)
            axes[0].set_xlabel('Block Index', fontsize=12)
            axes[0].set_ylabel('Effect on Loss (Ablated - Baseline)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Plot effects for impossible scenario
            axes[1].plot(block_indices, imposs_effects, 'r-', linewidth=2.5, label='Effect on Impossible', alpha=0.8)
            axes[1].scatter(block_indices, imposs_effects, color='red', s=50, zorder=5)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Highlight important blocks
            max_imposs_idx = imposs_effects.index(max(imposs_effects))
            max_imposs_block = block_indices[max_imposs_idx]
            axes[1].scatter([max_imposs_block], [imposs_effects[max_imposs_idx]], color='darkred', s=150, zorder=6)
            axes[1].annotate(f'Block {max_imposs_block}: {imposs_effects[max_imposs_idx]:.4f}', 
                        (max_imposs_block, imposs_effects[max_imposs_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
            
            axes[1].set_title('Ablation Effect on Impossible Event', fontsize=14)
            axes[1].set_xlabel('Block Index', fontsize=12)
            axes[1].set_ylabel('Effect on Loss (Ablated - Baseline)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Plot effect difference (importance for physical understanding)
            axes[2].plot(block_indices, diff_effects, 'g-', linewidth=2.5, label='Difference (Impossible - Possible)', alpha=0.8)
            axes[2].scatter(block_indices, diff_effects, color='green', s=50, zorder=5)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Highlight important blocks
            max_diff_idx = diff_effects.index(max(diff_effects))
            max_diff_block = block_indices[max_diff_idx]
            axes[2].scatter([max_diff_block], [diff_effects[max_diff_idx]], color='darkgreen', s=150, zorder=6)
            axes[2].annotate(f'Block {max_diff_block}: {diff_effects[max_diff_idx]:.4f}', 
                        (max_diff_block, diff_effects[max_diff_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
            
            axes[2].set_title('Difference in Ablation Effect (Impossible - Possible)\nIndicates importance for physical understanding', fontsize=14)
            axes[2].set_xlabel('Block Index', fontsize=12)
            axes[2].set_ylabel('Effect Difference', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Create save filename
            if save_prefix:
                save_name = f"{save_prefix}_{component}_ctx{context_len}.png"
            else:
                save_name = f"ablation_{component}_ctx{context_len}.png"
        
        # Add tight layout, allowing space for the summary text at the bottom when needed
        if component == "head":
            # Give more room at the bottom for the summary text
            plt.tight_layout(rect=[0, 0.10, 1, 0.92])
        else:
            # Standard spacing for other components
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        save_path = os.path.join(pair_output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def save_results(self, pair_name, context_len=8, save_prefix=None):
        """Save ablation results to disk"""
        # Check if results exist
        key = f"{pair_name}_{context_len}"
        if key not in self.results:
            print(f"No results found for {key}")
            return
        
        results = self.results[key]
        pair_dir_source = results.get("pair_dir", "O1_pairs")
        
        # Create output directory
        pair_output_dir = os.path.join(self.output_dir, f"{pair_dir_source}/{pair_name}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        # Create save filename
        if save_prefix:
            save_name = f"{save_prefix}_ctx{context_len}_results.pth"
        else:
            save_name = f"ablation_ctx{context_len}_results.pth"
        
        save_path = os.path.join(pair_output_dir, save_name)
        
        # Save results
        torch.save(results, save_path)
        print(f"Saved results to {save_path}")


def main():
    """
    Main function to run ablation patching
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Ablation patching for intuitive physics tasks')
    parser.add_argument('--config', type=str, 
                        default='evaluation_code/evals/intuitive_physics/configs/default_intphys.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, 
                        default='ablation_patching_results',
                        help='Output directory')
    parser.add_argument('--pair_path', type=str, 
                        default='O1_pairs',
                        help='Path to directory containing pairs (e.g., O1_pairs, O2_pairs)')
    parser.add_argument('--pair_prefix', type=str, 
                        default='pair_1_01',
                        help='Prefix for pair files')
    parser.add_argument('--components', type=str, 
                        nargs='+',
                        default=['full', 'att', 'mlp', 'head'],
                        choices=['full', 'att', 'mlp', 'head'],
                        help='Components to ablate (full=entire block, att=attention, mlp=MLP, head=individual attention heads)')
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
                        help='Which subset of blocks to ablate')
    parser.add_argument('--max_blocks', type=int,
                        default=32,
                        help='Maximum number of blocks to ablate')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Initialize ablation patching
    patcher = AblationPatching(
        config_path=args.config,
        output_dir=args.output_dir,
        device=device
    )
    
    # Get total number of blocks
    num_blocks = len(patcher.model["encoder"].backbone.blocks)
    print(f"Model has {num_blocks} transformer blocks")
    
    # Select blocks to ablate based on argument
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
    
    print(f"Selected {len(block_indices)} blocks to ablate: {block_indices}")
    
    # Load pair
    x_possible, x_impossible = patcher.load_pair(args.pair_path, args.pair_prefix)
    
    # Handle both --context and --context_lengths options
    if args.context is not None:
        # If --context was used, make a list with just that value
        context_lengths = [args.context]
    else:
        # Otherwise use the context_lengths list
        context_lengths = args.context_lengths

    # Run ablation patching for each context length
    for context_len in context_lengths:
        print(f"\nRunning ablation patching with context length {context_len}")
        
        # Run patching
        results = patcher.run_ablation_patching(
            x_possible=x_possible,
            x_impossible=x_impossible,
            pair_name=args.pair_prefix,
            pair_dir=os.path.basename(args.pair_path),  # Extract just the directory name
            block_indices=block_indices,
            components=args.components,
            context_len=context_len
        )
        
        # Plot results for each component
        for component in args.components:
            patcher.plot_ablation_results(
                pair_name=args.pair_prefix,
                component=component,
                context_len=context_len
            )
        
        # Save results
        patcher.save_results(
            pair_name=args.pair_prefix,
            context_len=context_len
        )
    
    print("\nAblation patching completed!")


if __name__ == "__main__":
    main()