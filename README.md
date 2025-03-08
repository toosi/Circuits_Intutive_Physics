# Intuitive Physics Understanding with JEPA Models

This repository contains code and data for investigating how JEPA (Joint Embedding Predictive Architecture) models represent intuitive physics concepts. It builds upon the work in the paper *Intuitive physics understanding emerges from self-supervised pretraining on natural videos* by Quentin Garrido et al.

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

## Key Findings

- JEPA models show sensitivity to physically impossible events in their internal representations
- Different blocks, layers, and attention heads encode different aspects of physical understanding
- Context length significantly impacts how the model processes physical scenarios
- Early blocks (especially Block 0) show strong differences in activations between possible and impossible events

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jepa-intuitive-physics.git
cd jepa-intuitive-physics

# Install dependencies
pip install -r requirements.txt
```

### Running Visualization Script

```bash
# Run visualization script on a sample pair
python visualize_activations_diff.py --pair_name pair_4_02 --context_length 2

# Test multiple context lengths in sequence
for ctx in 1 2 4 8 16; do
  python visualize_activations_diff.py --pair_name pair_4_02 --context_length $ctx
done
```

### Analyzing Results

Generated visualizations will appear in the `activation_differences` directory, showing:

#### Block-level Analysis
Identifies which transformer blocks show strongest differentiation between possible and impossible events

#### Component-level Analysis
Determines whether attention mechanisms or MLPs are more important for physical reasoning

#### Attention Head Analysis
Shows which specific attention heads are sensitive to physically impossible events

#### Multi-Context Analysis
Compares how the model processes physical events at different context lengths

## Findings & Contributions

### Context Length Matters
Our analysis reveals that context length significantly impacts how the model processes physical events:
- Context length 2 shows strongest differentiation for some object pairs
- Different blocks become important at different context lengths
- The model uses different attention heads depending on context length

### Architectural Insights
The visualizations reveal important insights about how transformer architectures process physical events:
- Early blocks (0-3) often show strongest sensitivity to physical impossibility
- Attention mechanisms and MLPs contribute differently at various context lengths
- Specific attention heads in early blocks specialize in tracking object permanence

### Methodological Advances
We introduce several methodological improvements:
- Direct activation analysis instead of relying on patching
- Comprehensive visualization across model components
- Comparative analysis across context lengths

## Original Paper

This work extends the original paper by Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun.

## Raw surprises and performance data

After decompressing the data, e.g. by doing `tar -xzvf data_intphys.tar.gz`, you will be left with the following folder structure.
```
data_intphys/
├── model_1/
│   ├── dataset_1/
│   │   ├── performance.csv
│   │   └── raw_surprises/
│   │       ├── property1_XXX.pth
│   │      ...
│   │       └── propertyN_XXX.pth
│   ...
│   └── dataset_M
...
└── model_K/
```

Every model starting with `vit` is a V-JEPA model. Otherwise, the exact model is specified.

For each model and each dataset, we report processed performance in a `performance.csv` file. It can directly be processed by our notebook used to generate figures.

Performance is reported per property and context length, using various metrics (depending on whether the model is a LLM or prediction based model).

We also provide raw_surprises in the `raw_surprises/` folder. The `.pth` files contain all of the raw surprise metrics output by our evaluation code, which are used to determine performance. They are available per model, dataset, and property.

## Figures reproduction

The notebook `figures.ipynb` can be used to reproduce all of the figures in the paper.

After decompressing the data, it can directly be ran and will output all figures computed using the raw performance files.

## Evaluation code

### Running the code

For algorithmic clarity and reproducibility, we provide a version of our code which can be used to extract surprise metrics from models. It is compatible with V-JEPA models and VideoMAEv2. The code is based on [github.com/facebookresearch/jepa](https://github.com/facebookresearch/jepa).

For requirements to run the code, see `requirements.txt` .

We provide two evaluations:
- `intuitive_physics` is the regular evaluation, used for IntPhys, GRASP and InfLevel-lab.
- `intphys_test` is used to run an evaluation on the test set of IntPhys, that can then be submitted on the official leaderboard. It will not give any metrics by itself.

To run the evaluation code, the files `evaluation_code/evals/intuitive_physics/utils.py` and `evaluation_code/evals/intphys_test/utils.py` need to be adapted.

As the code is meant to be reusable on various clusters where data doesn't share a common path. You need to specify what is `CLUSTER` as well as what the paths of the datasets are.
If you intend on only using a singular cluster, the `get_cluster()` function can simply be replaced by:
```python
@lru_cache()
def get_cluster() -> str:
    return CLUSTER
```
Then, just update the dataset paths in `DATASET_PATHS_BY_CLUSTER`.

From the `evaluation_code` folder, evaluations can either be run locally, e.g:
```bash
python -m evals.main --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 --fname evals/intuitive_physics/configs/default_intphys.yaml
```

or through submitit, e.g.:

```bash
python -m evals.main_distributed --fname evals/intuitive_physics/configs/default_intphys.yaml --folder ./logs --partition PARTITION 
```

### Configurations

We provide default configurations in the evaluations folder that should be adapted depending on the model that you are using.

The *pretrain* section contains information to load the pretrained model. Most important are *folder* which is the folder where the checkpoint is located, and *checkpoint* which is the name of the checkpoint.

The parameters *tasks_per_node* and *nodes* are only used when using submitit to control the number of GPUs used. Due to the size of intphys, we recommend using 6 *tasks_per_node* and 1 *node* (as the world size must be a divisor of 30).


## License


All of the code in the present repository is released under CC-BY-NC, which does not allow commercial use. See [LICENSE](LICENSE) for details.
