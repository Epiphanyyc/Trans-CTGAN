# Trans-CTGAN: Improving Correlation Preservation in Tabular GANs

## Overview

This repository contains the supplementary code for our research paper that proposes Trans-CTGAN, an enhanced version of CTGAN that incorporates Transformer encoder modules and residual connections to improve the model's ability to capture latent correlations between different columns in tabular data.

## Repository Structure

### Baseline Model Reproduction

The following directories contain our initial experiments with existing models for reproduction and baseline comparison:

- **`CTAB-GAN-main/`**: Contains reproduction experiments of the CTAB-GAN model on various datasets
- **`CTAB-GAN-Plus-main/`**: Contains reproduction experiments of the CTAB-GAN+ model on various datasets  
- **`CTGAN-main/`**: Contains reproduction experiments of the original CTGAN model on various datasets

These folders include multiple experimental runs to evaluate the actual performance of these baseline models across different datasets.

### Main Implementation

The **`synthcity-main/`** directory contains the primary codebase used throughout our experimental process.

#### Core Model Implementation

- **`synthcity-main/trans-ctgan-model/trans-ctgan.py`**: Contains the main implementation of our proposed Trans-CTGAN model. This file includes the core `EnhancedCTGAN` class, which represents our Trans-CTGAN architecture built upon the original CTGAN framework with Transformer encoder, residual connection enhancements and other designs.

#### Comparative Analysis

The following notebooks demonstrate horizontal comparisons between different models using the synthcity framework:

- **`synthcity-main/tutorials/tutorial1_add_a_new_plugin.ipynb`**
- **`synthcity-main/tutorials/tutorial2_add_a_new_plugin.ipynb`**

#### Individual Model Experiments

The following notebooks showcase individual data generation experiments for each model:

- **`synthcity-main/tutorials/ctabgan.ipynb`**: CTAB-GAN model experiments
- **`synthcity-main/tutorials/ddpm.ipynb`**: DDPM model experiments
- **`synthcity-main/tutorials/OriginalCTGAN.ipynb`**: Original CTGAN model experiments
- **`synthcity-main/tutorials/tvae.ipynb`**: TVAE model experiments
- **`synthcity-main/tutorials/trans-ctgan.ipynb`**: Our proposed Trans-CTGAN model experiments

#### Evaluation Metrics

The following notebooks contain our evaluation methodology and metric calculations:

**Correlation Analysis:**

- **`synthcity-main/tutorials/L2.ipynb`**: Computation of L2 distance metrics based on Pearson correlation coefficient matrices between generated and real data
- **`synthcity-main/tutorials/dcor.ipynb`**: Computation of distance correlation (dcor) metrics between generated and real data

**Comprehensive Evaluation:**

- **`synthcity-main/ML_Utility_and_WDKLJSD.ipynb`**: Calculation of ML Utility, Wasserstein Distance (WD), Jensen-Shannon Divergence (JSD), and other evaluation metrics

### Datasets

All datasets used in the experiments are included in the **`datasets/`** folder. Please note that some code sections use absolute paths for dataset loading, but all referenced datasets are available within this directory.

## Important Notes

This research represents the author's first academic project and paper submission. Due to the three-day submission deadline for supplementary materials, the codebase retains its experimental development structure rather than being refactored into a clean, modular framework. 

The code contains numerous experimental iterations and local environment testing scripts that were part of the research development process. While this provides complete transparency into our experimental methodology, we acknowledge that the code organization could benefit from further systematization and modularization.

We appreciate your understanding regarding the current code structure and are happy to provide clarifications on any specific components upon request.

## Usage

To reproduce our experiments:

1. Ensure all required dependencies are installed
2. Update any absolute paths in the code to match your local environment
3. Run the desired experiment notebooks in the `synthcity-main/tutorials/` directory
4. For the core Trans-CTGAN implementation, refer to `synthcity-main/trans-ctgan-model/trans-ctgan.py`

## Contact

For questions regarding the implementation or experimental setup, please feel free to reach out to the authors.
