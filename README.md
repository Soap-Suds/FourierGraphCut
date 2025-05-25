# Fourier Graph Cut

## Overview
The Fourier Graph Cut project implements a Lazy Branch and Bound algorithm for graph partitioning. It provides tools for generating various datasets and a Jupyter notebook for easy execution of the algorithm. This project aims to facilitate research and experimentation in graph theory and optimization.

## Project Structure
```
FourierGraphCut
├── src
│   ├── __init__.py
│   ├── lazy_branch_and_bound.py
│   ├── dataset_generation.py
│   └── utils.py
├── notebooks
│   └── run_algorithm.ipynb
├── requirements.txt
└── README.md
└── Z_k-Graph-Cut.ipynb
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
1. **Dataset Generation**: Use the functions in `src/dataset_generation.py` to create unweighted and weighted datasets. The `randdataset` function allows you to specify parameters for different types of datasets.

2. **Running the Algorithm**: Open the Jupyter notebook located in the `notebooks` directory (`run_algorithm.ipynb`). This notebook provides an interactive interface to run the Lazy Branch and Bound algorithm with the generated datasets.

3. **Lazy Branch and Bound Class**: The core algorithm is implemented in the `LazyBranchAndBound` class found in `src/lazy_branch_and_bound.py`. This class includes methods for initializing the algorithm and executing the branch and bound process.

## Notes
This project is still currently in development, Z_k-Graph-Cut.ipynb is the Jupyter notebook where all methods are being tested.