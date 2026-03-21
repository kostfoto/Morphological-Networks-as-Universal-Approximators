# Morphological-Networks-as-Universal-Approximators
Pytorch implementation of the paper "Training Deep Morphological Neural Networks as Universal Approximators" by K. Fotopoulos and P. Maragos (arXiv:2505.09710). 

* `finalMNNs.ipynb`: Corresponds to the majority of the experiments, with results appearing in Tables 1,2,3,5,6 of the main text, and Table 13 and Figure 5 of the appendix. 

* `regressionMNNs.ipynb`: Corresponds to the regression and "mean-shift" results in the appendix.

* `resnet_20_experiments.ipynb`: Corresponds to any results involving ResNet-20 that appear in the paper. 

* `snip_experiments.ipynb`: Corresponds to the pruning experiments involving SNIP. 

* `conv2d_cuda.cu`: A Pytorch CUDA implementation for morphological convolutional layers. We provide only a GPU implementation (thus, a GPU is required to run the ResNet-20 experiments). 

Reproducing the experiments requires running the notebooks sequentially. Experiments used multiple runs of the above files to report averaged results. This repository includes only one run of each. 
