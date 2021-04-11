## Introduction
This repository contains code for the experiments in the paper Statistical Estimation from Dependent Data [[arxiv]()].
The code base is derived from the code base of the following paper: Graph Random Neural Network for Semi-Supervised Learning on Graphs [[arxiv](https://arxiv.org/pdf/2005.11079.pdf)].
There are three graph datasets we work with:
* Cora
* Citeseer
* Pubmed

We study the improvements obtained by the maximum pseudo-likelihood estimator (MPLE) analyzed in our paper on inference from graph-dependent data. We also compare the performance of MPLE with a contemporary GNN based approach for the problem and observe competitive performance (as of February 2021).


## Requirements
* Python 3.7.3
* Please install other pakeages by 
``` pip install -r requirements.txt```

## Code Structure
The ```train_*.py``` scripts perform the training of different variants on any of the three datasets. The ```run_*.sh``` and ```run100_*.sh``` bash scripts have been set up to call the appropriate ```train_*.py``` based on the command line arguments passed. The ```run100_*.sh``` scripts perform multiple Monte-Carlo executions across a range of hyperparameter settings.
The output of the runs is set to be recorded in folders ```cora/, pubmed/, citeseer/```.
```result_*.py``` files compile the results of the Monte-Carlo runs generated above and generate plots.

<!--## Usage Example
* Running one trial on a dataset: -->
