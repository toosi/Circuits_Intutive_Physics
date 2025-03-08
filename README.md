# Circuits for Intuitive Physics Understanding in Vision models (Under Construction)

This repository contains code and data for investigating how JEPA (Joint Embedding Predictive Architecture) models represent intuitive physics concepts using mechanistic interpretabilty methods. It builds upon the work in the paper *Intuitive physics understanding emerges from self-supervised pretraining on natural videos* by Quentin Garrido et al and the methods developed in *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small by Kevin Wang et al.

## Overview

The project focuses on:
1. Evaluating how JEPA models represent physical concepts like object permanence
2. Analyzing internal model activations to understand physical reasoning
3. Visualizing differences between possible and impossible physics scenarios
4. Investigating the impact of context length on physical understanding

![Object Permanence Visualization](object_perm_results/surprise_barplot.png)

## Repository Contents

### Data
- **O1_pairs/**: Example possible/impossible physics scenarios for testing
- **activation_patching_results/**: Results from activation patching experiments
- **object_perm_results/**: Visualizations of object permanence tests

### Scripts
- **visualize_activations_diff.py**: Comprehensive script for analyzing model activations between possible/impossible events
- **test_object_permanence.py**: Test script for object permanence understanding
- **test_activation_patching_O1.py**: Activation patching for object permanence
- **activation_patching_ablation.py**: Ablation studies for patching technique
- **test_direct_prediction.py**: Direct comparison of predictions between scenarios

### Framework
- **evaluation_code/**: Original evaluation framework from JEPA paper



## Original Paper

This work extends the original paper by Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun.

