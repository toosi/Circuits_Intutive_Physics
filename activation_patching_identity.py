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
from evals.intuitive_physics.utils import get_dataset_paths
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


class ComponentPatcher:
    """
    Class for patching different components of the model
    """
    def __init__(self, model):
        self.model = model
        self.encoder = model["encoder"]
        self.target_encoder = model["target_encoder"]
        self.predictor = model["predictor"]
        self.original_blocks = {}
        self.original_attns = {}
        self.original_mlps = {}
    
    def register_original_blocks(self):
        """Store original block implementations"""
        for i, block in enumerate(self.encoder.backbone.blocks):
            self.original_blocks[i] = copy.deepcopy(block)
            self.original_attns[i] = copy.deepcopy(block.attn)
            self.original_mlps[i] = copy.deepcopy(block.mlp)
    
    def patch_attention_block(self, block_idx, source_model):
        """Patch only the attention component of a block"""
        # Replace attention module with the one from source model
        source_attn = source_model["encoder"].backbone.blocks[block_idx].attn
        self.encoder.backbone.blocks[block_idx].attn = copy.deepcopy(source_attn)
    
    def patch_mlp_block(self, block_idx, source_model):
        """Patch only the MLP component of a block"""
        # Replace MLP module with the one from source model
        source_mlp = source_model["encoder"].backbone.blocks[block_idx].mlp
        self.encoder.backbone.blocks[block_idx].mlp = copy.deepcopy(source_mlp)
    
    def patch_full_block(self, block_idx, source_model):
        """Patch the entire block"""
        # Replace full block with the one from source model
        source_block = source_model["encoder"].backbone.blocks[block_idx]
        self.encoder.backbone.blocks[block_idx] = copy.deepcopy(source_block)
    
    def restore_original_blocks(self):
        """Restore all original blocks"""
        for i, block in self.original_blocks.items():
            self.encoder.backbone.blocks[i] = copy.deepcopy(block)
    
    def restore_original_attention(self, block_idx):
        """Restore original attention for a specific block"""
        if block_idx in self.original_attns:
            self.encoder.backbone.blocks[block_idx].attn = copy.deepcopy(self.original_attns[block_idx])
    
    def restore_original_mlp(self, block_idx):
        """Restore original MLP for a specific block"""
        if block_idx in self.original_mlps:
            self.encoder.backbone.blocks[block_idx].mlp = copy.deepcopy(self.original_mlps[block_idx])


class RelativeSurpriseEvaluator:
    """
    Class to evaluate the relative surprise of a model on intuitive physics tasks
    with component patching
    """
    def __init__(self, config_path, output_dir="activation_patching_results", device=None):
        """
        Initialize the evaluator
        
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
        
        # Initialize component patcher
        self.patcher = ComponentPatcher(self.model)
        self.patcher.register_original_blocks()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Cache for results
        self.baseline_results = {}
        self.activation_patching_results = {}
    
    def get_data_loader(self, block="O1", frame_step=2, transform=None):
        """
        Initialize data loader for evaluation
        
        Args:
            block: Block to evaluate (O1, O2, O3)
            frame_step: Frame step for sampling
            transform: Optional transform
        """
        # Extract configs
        data_cfg = self.config["data"]
        
        # Create transform if not provided
        if transform is None:
            transform = make_transforms(
                random_horizontal_flip=False,
                random_resize_aspect_ratio=[1/1, 1/1],
                random_resize_scale=[1.0, 1.0],
                reprob=0.,
                auto_augment=False,
                motion_shift=False,
                crop_size=data_cfg["resolution"])
        
        # Initialize data loader
        data_name = f"IntPhys-dev-{block}" 
        dataset, data_loader, sampler = init_data(
            batch_size=data_cfg["batch_size"],
            transform=transform,
            data=data_name,
            property=block,
            collator=None,
            pin_mem=True,
            num_workers=8, 
            world_size=1,
            rank=0,
            root_path=get_dataset_paths([data_name])[0],
            clip_len=99//frame_step,
            frame_sample_rate=frame_step,
            deterministic=True,
            log_dir=None)
        
        return data_loader
    
    def compute_baseline_surprise(self, block="O1", frame_step=2, context_lengths=None):
        """
        Compute baseline surprise scores without any patching
        
        Args:
            block: Block to evaluate (O1, O2, O3)
            frame_step: Frame step for sampling
            context_lengths: List of context lengths to evaluate
        """
        print(f"\nComputing baseline surprise scores for block {block}, frame_step {frame_step}")
        
        # Use default context lengths from config if not provided
        if context_lengths is None:
            context_lengths = self.config["data"]["context_lengths"]
        
        # Extract losses for the original model
        data_cfg = self.config["data"]
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=[1/1, 1/1],
            random_resize_scale=[1.0, 1.0],
            reprob=0.,
            auto_augment=False,
            motion_shift=False,
            crop_size=data_cfg["resolution"])
        
        # Extract losses
        all_losses, all_labels = extract_losses(
            device=self.device,
            encoder=self.model["encoder"],
            target_encoder=self.model["target_encoder"],
            predictor=self.model["predictor"],
            transform=transform,
            use_bfloat16=data_cfg["use_bfloat16"],
            block=block,
            frame_step=frame_step,
            context_lengths=context_lengths,
            batch_size=data_cfg["batch_size"],
            frames_per_clip=data_cfg["frames_per_clip"],
            stride=data_cfg["stride_sliding_window"],
            world_size=1,
            rank=0,
            normalize_enc=False,
            dataset="intphys",
            is_mae=False,
            normalize_targets=True
        )
        
        # Compute metrics for each context length
        baseline_metrics = {}
        for i, context in enumerate(context_lengths):
            losses = all_losses[:, i]
            metrics = compute_metrics(losses, all_labels)
            baseline_metrics[context] = metrics
            print(f"Context {context}: Relative Accuracy (avg): {metrics['Relative Accuracy (avg)']:.2f}%")
        
        # Store results
        self.baseline_results[f"{block}_{frame_step}"] = {
            "losses": all_losses,
            "labels": all_labels,
            "metrics": baseline_metrics
        }
        
        return baseline_metrics
    
    def patch_and_evaluate(self, block="O1", frame_step=2, block_indices=None, 
                          component="both", context_lengths=None):
        """
        Run patching experiments and evaluate relative surprise
        
        Args:
            block: Block to evaluate (O1, O2, O3)
            frame_step: Frame step for sampling
            block_indices: List of transformer block indices to patch
            component: Component to patch ('att', 'mlp', or 'both')
            context_lengths: List of context lengths to evaluate
        """
        print(f"\nRunning activation patching for block {block}, frame_step {frame_step}, component {component}")
        
        # Get number of blocks in model
        num_blocks = len(self.model["encoder"].backbone.blocks)
        
        # Default to all blocks if not specified
        if block_indices is None:
            block_indices = list(range(num_blocks))
        
        # Use default context lengths from config if not provided
        if context_lengths is None:
            context_lengths = self.config["data"]["context_lengths"]
        
        # Ensure we have baseline results
        baseline_key = f"{block}_{frame_step}"
        if baseline_key not in self.baseline_results:
            self.compute_baseline_surprise(block, frame_step, context_lengths)
        
        # Create a copy of the original model for patching
        print("Creating baseline model copy...")
        patched_model = copy.deepcopy(self.model)
        
        # For each transformer block, patch the component and evaluate
        results = {"att": [], "mlp": [], "full": []}
        
        # Extract baseline metrics (average over all context lengths)
        baseline_metrics = {}
        for context in context_lengths:
            baseline_metrics[context] = self.baseline_results[baseline_key]["metrics"][context]
        
        patched_results = {}
        
        # Check which components to patch
        components_to_patch = []
        if component == "att" or component == "both":
            components_to_patch.append("att")
        if component == "mlp" or component == "both":
            components_to_patch.append("mlp")
        if component == "full":
            components_to_patch.append("full")
        
        # Loop through blocks and patch
        for block_idx in tqdm(block_indices, desc="Patching blocks"):
            for comp in components_to_patch:
                print(f"Patching {comp} in block {block_idx}...")
                
                # Create a fresh copy of the model for this experiment
                patched_model = copy.deepcopy(self.model)
                patcher = ComponentPatcher(patched_model)
                
                # Apply patching
                if comp == "att":
                    # Only patch attention
                    patched_model["encoder"].backbone.blocks[block_idx].attn = nn.Identity()
                elif comp == "mlp":
                    # Only patch MLP
                    patched_model["encoder"].backbone.blocks[block_idx].mlp = nn.Identity()
                elif comp == "full":
                    # Patch entire block
                    def identity_forward(x, mask=None):
                        return x
                    patched_model["encoder"].backbone.blocks[block_idx].forward = identity_forward
                
                # Extract losses for patched model
                data_cfg = self.config["data"]
                transform = make_transforms(
                    random_horizontal_flip=False,
                    random_resize_aspect_ratio=[1/1, 1/1],
                    random_resize_scale=[1.0, 1.0],
                    reprob=0.,
                    auto_augment=False,
                    motion_shift=False,
                    crop_size=data_cfg["resolution"])
                
                # Extract losses
                try:
                    patched_losses, patched_labels = extract_losses(
                        device=self.device,
                        encoder=patched_model["encoder"],
                        target_encoder=patched_model["target_encoder"],
                        predictor=patched_model["predictor"],
                        transform=transform,
                        use_bfloat16=data_cfg["use_bfloat16"],
                        block=block,
                        frame_step=frame_step,
                        context_lengths=context_lengths,
                        batch_size=data_cfg["batch_size"],
                        frames_per_clip=data_cfg["frames_per_clip"],
                        stride=data_cfg["stride_sliding_window"],
                        world_size=1,
                        rank=0,
                        normalize_enc=False,
                        dataset="intphys",
                        is_mae=False,
                        normalize_targets=True
                    )
                    
                    # Compute metrics for each context length
                    patched_metrics = {}
                    for i, context in enumerate(context_lengths):
                        losses = patched_losses[:, i]
                        metrics = compute_metrics(losses, patched_labels)
                        patched_metrics[context] = metrics
                    
                    # Calculate delta for each metric
                    delta_metrics = {}
                    for context in context_lengths:
                        baseline = baseline_metrics[context]["Relative Accuracy (avg)"]
                        patched = patched_metrics[context]["Relative Accuracy (avg)"]
                        delta = baseline - patched  # Positive delta means importance (degradation after patching)
                        delta_metrics[context] = delta
                        print(f"  Context {context}: Delta = {delta:.2f}% (Baseline: {baseline:.2f}%, Patched: {patched:.2f}%)")
                    
                    # Store results
                    results[comp].append({
                        "block_idx": block_idx,
                        "patched_metrics": patched_metrics,
                        "delta_metrics": delta_metrics
                    })
                    
                except Exception as e:
                    print(f"Error in patching {comp} for block {block_idx}: {e}")
                    continue
                
                # Clear memory
                del patched_model
                torch.cuda.empty_cache()
        
        # Store activation patching results
        self.activation_patching_results[f"{block}_{frame_step}_{component}"] = results
        
        return results
    
    def plot_activation_patching_results(self, block="O1", frame_step=2, 
                                        component="both", context_length=8,
                                        save_path=None):
        """
        Plot activation patching results
        
        Args:
            block: Block that was evaluated (O1, O2, O3)
            frame_step: Frame step used
            component: Component that was patched ('att', 'mlp', or 'both')
            context_length: Context length to use for plotting
            save_path: Path to save the plot (if None, will use default)
        """
        # Get results key
        results_key = f"{block}_{frame_step}_{component}"
        if results_key not in self.activation_patching_results:
            print(f"No results found for {results_key}")
            return
        
        results = self.activation_patching_results[results_key]
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        components_to_plot = []
        if component == "att" or component == "both":
            components_to_plot.append("att")
        if component == "mlp" or component == "both":
            components_to_plot.append("mlp")
        if component == "full":
            components_to_plot.append("full")
        
        for comp in components_to_plot:
            block_indices = [r["block_idx"] for r in results[comp]]
            deltas = [r["delta_metrics"][context_length] for r in results[comp]]
            
            # Plot
            if comp == "att":
                plt.plot(block_indices, deltas, 'b-', label='Attention', alpha=0.8)
            elif comp == "mlp":
                plt.plot(block_indices, deltas, 'r-', label='MLP', alpha=0.8)
            elif comp == "full":
                plt.plot(block_indices, deltas, 'g-', label='Full Block', alpha=0.8)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Highlight max delta point
            if len(deltas) > 0:
                max_idx = deltas.index(max(deltas))
                max_block = block_indices[max_idx]
                plt.scatter([max_block], [deltas[max_idx]], color='green', s=100, zorder=5)
                plt.annotate(f'Block {max_block}: {deltas[max_idx]:.2f}%', 
                            (max_block, deltas[max_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
        # Add labels and title
        plt.xlabel('Block Index', fontsize=12)
        plt.ylabel('Relative Accuracy Delta (%)', fontsize=12)
        title = f'Activation Patching Results: Impact on Relative Surprise\nBlock: {block}, Context: {context_length}'
        plt.title(title, fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Set y-axis to be symmetric to better visualize positive/negative effects
        y_max = plt.ylim()[1]
        y_min = plt.ylim()[0]
        y_lim = max(abs(y_max), abs(y_min))
        plt.ylim(-y_lim, y_lim)
        
        # Add tight layout
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            os.makedirs(os.path.join(self.output_dir, f"{block}"), exist_ok=True)
            save_path = os.path.join(self.output_dir, f"{block}", f"activation_patching_{component}_ctx{context_length}.png")
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_all_context_lengths(self, block="O1", frame_step=2, component="both", save_path=None):
        """
        Plot activation patching results for all context lengths
        
        Args:
            block: Block that was evaluated (O1, O2, O3)
            frame_step: Frame step used
            component: Component that was patched ('att', 'mlp', or 'both')
            save_path: Path to save the plot (if None, will use default)
        """
        # Get results key
        results_key = f"{block}_{frame_step}_{component}"
        if results_key not in self.activation_patching_results:
            print(f"No results found for {results_key}")
            return
        
        results = self.activation_patching_results[results_key]
        
        # Get context lengths
        if len(results["att"]) > 0:
            context_lengths = list(results["att"][0]["delta_metrics"].keys())
        elif len(results["mlp"]) > 0:
            context_lengths = list(results["mlp"][0]["delta_metrics"].keys())
        else:
            context_lengths = list(results["full"][0]["delta_metrics"].keys())
        
        # Create subplots
        num_contexts = len(context_lengths)
        fig, axes = plt.subplots(num_contexts, 1, figsize=(12, 4*num_contexts), sharex=True)
        
        if num_contexts == 1:
            axes = [axes]  # Make it iterable
        
        # Extract data for plotting
        components_to_plot = []
        if component == "att" or component == "both":
            components_to_plot.append("att")
        if component == "mlp" or component == "both":
            components_to_plot.append("mlp")
        if component == "full":
            components_to_plot.append("full")
        
        # Plot each context length
        for i, context in enumerate(context_lengths):
            ax = axes[i]
            
            for comp in components_to_plot:
                block_indices = [r["block_idx"] for r in results[comp]]
                deltas = [r["delta_metrics"][context] for r in results[comp]]
                
                # Plot
                if comp == "att":
                    ax.plot(block_indices, deltas, 'b-', label='Attention', alpha=0.8)
                elif comp == "mlp":
                    ax.plot(block_indices, deltas, 'r-', label='MLP', alpha=0.8)
                elif comp == "full":
                    ax.plot(block_indices, deltas, 'g-', label='Full Block', alpha=0.8)
                
                # Add horizontal line at y=0
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Highlight max delta point
                if len(deltas) > 0:
                    max_idx = deltas.index(max(deltas))
                    max_block = block_indices[max_idx]
                    ax.scatter([max_block], [deltas[max_idx]], color='green', s=100, zorder=5)
                    ax.annotate(f'Block {max_block}: {deltas[max_idx]:.2f}%', 
                                (max_block, deltas[max_idx]),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
            
            # Add labels and title for each subplot
            ax.set_ylabel('Relative Accuracy Delta (%)', fontsize=12)
            ax.set_title(f'Context Length: {context}', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis to be symmetric
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            y_lim = max(abs(y_max), abs(y_min))
            ax.set_ylim(-y_lim, y_lim)
            
            if i == 0:  # Only add legend to the first subplot
                ax.legend(fontsize=11)
        
        # Add overall title and x-axis label
        fig.suptitle(f'Activation Patching Results: Impact on Relative Surprise\nBlock: {block}', fontsize=16)
        axes[-1].set_xlabel('Block Index', fontsize=12)
        
        # Add tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make space for suptitle
        
        # Default save path if not provided
        if save_path is None:
            os.makedirs(os.path.join(self.output_dir, f"{block}"), exist_ok=True)
            save_path = os.path.join(self.output_dir, f"{block}", f"activation_patching_{component}_all_contexts.png")
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-context plot to {save_path}")
        
        plt.close()
    
    def save_results(self, block="O1", frame_step=2, component="both"):
        """Save patching results to disk"""
        # Get results key
        results_key = f"{block}_{frame_step}_{component}"
        if results_key not in self.activation_patching_results:
            print(f"No results found for {results_key}")
            return
        
        results = self.activation_patching_results[results_key]
        
        # Create output directory
        os.makedirs(os.path.join(self.output_dir, f"{block}"), exist_ok=True)
        save_path = os.path.join(self.output_dir, f"{block}", f"activation_patching_{component}_results.pth")
        
        # Save results
        torch.save({
            "block": block,
            "frame_step": frame_step,
            "component": component,
            "results": results,
            "baseline": self.baseline_results[f"{block}_{frame_step}"]
        }, save_path)
        
        print(f"Saved results to {save_path}")


def main():
    """
    Main function to run activation patching evaluation
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Activation patching for intuitive physics tasks')
    parser.add_argument('--config', type=str, 
                        default='evaluation_code/evals/intuitive_physics/configs/default_intphys.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, 
                        default='activation_patching_results',
                        help='Output directory')
    parser.add_argument('--block', type=str, 
                        default='O1',
                        choices=['O1', 'O2', 'O3'],
                        help='Block to evaluate')
    parser.add_argument('--frame_step', type=int, 
                        default=2,
                        help='Frame step for sampling')
    parser.add_argument('--component', type=str, 
                        default='both',
                        choices=['att', 'mlp', 'both', 'full'],
                        help='Component to patch')
    parser.add_argument('--context_lengths', type=int, 
                        nargs='+',
                        default=None,
                        help='Context lengths to evaluate (default: use config)')
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
    
    # Initialize evaluator
    evaluator = RelativeSurpriseEvaluator(
        config_path=args.config,
        output_dir=args.output_dir,
        device=device
    )
    
    # Get total number of blocks
    num_blocks = len(evaluator.model["encoder"].backbone.blocks)
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
    
    # Compute baseline surprise
    evaluator.compute_baseline_surprise(
        block=args.block,
        frame_step=args.frame_step,
        context_lengths=args.context_lengths
    )
    
    # Run patching experiments
    results = evaluator.patch_and_evaluate(
        block=args.block,
        frame_step=args.frame_step,
        block_indices=block_indices,
        component=args.component,
        context_lengths=args.context_lengths
    )
    
    # Plot results
    evaluator.plot_all_context_lengths(
        block=args.block,
        frame_step=args.frame_step,
        component=args.component
    )
    
    # Plot individual context lengths
    if args.context_lengths:
        for context in args.context_lengths:
            evaluator.plot_activation_patching_results(
                block=args.block,
                frame_step=args.frame_step,
                component=args.component,
                context_length=context
            )
    else:
        # Use default context length of 8 if none provided
        evaluator.plot_activation_patching_results(
            block=args.block,
            frame_step=args.frame_step,
            component=args.component,
            context_length=8
        )
    
    # Save results
    evaluator.save_results(
        block=args.block,
        frame_step=args.frame_step,
        component=args.component
    )


if __name__ == "__main__":
    main()