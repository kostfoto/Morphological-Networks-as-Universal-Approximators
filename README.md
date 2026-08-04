# Training Deep Morphological Neural Networks as Universal Approximators

Code of paper:

**Training Deep Morphological Neural Networks as Universal Approximators**  
arXiv preprint *arXiv:2505.09710*

We investigate deep morphological neural networks (DMNNs), studying how changes in algebraic structure affect the expressiveness and trainability of deep architectures. We show that despite the inherent non-linearity of morphological operations, existing deep morphological architectures fail to be universal approximators and exhibit optimization limitations. To address these issues, we introduce architectures incorporating constrained "linear" activations between morphological layers. In the first two architectures, only O(N) parameters (or learnable parameters) per layer of size N belong to the activations, with the remaining parameters constrained to morphological operations. We prove universal approximation results for the proposed architectures and show empirically that they can be successfully trained on standard image classification tasks. Residual connections and weight dropout further improve generalization. Our experiments show that our networks are trainable, without requiring substantially larger parameter counts than comparable linear networks despite the imposed architectural restrictions. Finally, we propose a hybrid linear/morphological architecture and observe accelerated convergence of gradient descent under large batches.

---

## Repository Structure

```text
.
├── morphConv2d_project/                # Directory for testing the CUDA modules
│   ├── README.md                       # readme for the cuda modules
│   ├── Makefile                        # makefile with compilation instructions
│   ├── build/                          # Folder where built modules are put
│   ├── include/                        
│   │   └── conv2d_thorough.hpp         # Header files for thorough module
│   ├── pytests/                        # Tests of modules 
│   │   ├── test_gpu*.cu                # Tests for gpu modules
│   │   └── test_cpu*.py                # Tests for the cpu fallback of the thorough module
│   └── src/                            # Source coda of the modules
│       ├── conv2d_cuda.cu              # src for the simple module
│       └── conv2d_thorough.cu          # src for the more thoroughly written module
├── conv2d_cuda.cu                      # The morphological convolution module
├── environment.yml                     # conda environment
├── mainMNN*.ipynb                      # The main experiments
├── regressionMNNs.ipynb                # Toy regression experiments of appendix
├── reporting.ipynb                     # Reporting of results (except snip)
├── reporting_snip.ipynb                # Reporting of snip results (corrected)
├── resnet_20_experiments*.ipynb        # MPM-ResNet-20 experiments
├── snip_experiments_corrected*.ipynb   # Snip experiments on RMPM (corrected)
├── snip_resnet_experiments_corrected*.ipynb    # Snip experiments on MPM-ResNet-20 (corrected)
└── README.md
```

The top-level directory contains the code required for replication of the experiments.

The `morphConv2d_project/` directory contains the development directory of the morphological convolution CUDA module, including its testing. See details below and the its corresponding `README.md` for usage.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kostfoto/Morphological-Networks-as-Universal-Approximators.git
cd Morphological-Networks-as-Universal-Approximators
```

The experiments were conducted using Python 3.8.18 with CUDA toolkit 11.1.

For reproduction of the experimental environment, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate myenv
```

The experiments were conducted using the top-level `conv2d_cuda.cu` CUDA module. You are meant to use this module with PyTorch's JIT compiler and loader. 

The code was designed strictly with the conducted experiments in mind. Please do not run low-level code outside of its development context without first reading it. (As an example, in larger networks the used `int` types may overflow.) 

The code is knowingly inefficient. This is not a concern for the conducted experiments.

If, for any reason, you are unable to run the code with JIT, you will have to compile the module yourself using `nvcc` from the CUDA toolkit. You can do this from the development directory `morphConv2d_project/`. The directory provides a Makefile with a compilation instruction. Note that, depending on your installed libraries, you may have to tinker with the compilation flags (in our case, we had to compile with std=c++17 to allow it to work). 

The development directory `morphConv2d_project/` also provides a series of tests for confirming the soundness of the code. We provide a more thoroughly written module (which is, however, more inefficient), which includes a CPU fallback kernel. The code that used to run the experiments passes all tests, and has also been tested with compute-sanitizer --tool {memcheck, racecheck} to ensure proper operation. To run the tests yourself, you have to compile the code, copy the .so file from `morphConv2d_project/build` to `morphConv2d_project/pytests`, and run the provided tests. 

In summary, without any problems, you should be able to ignore the developement directory `morphConv2d_project/`, use the top-level CUDA kernel, compile and load it with JIT, and everything should work just with this setup. 

Never run low-level code without first reading it: This software is provided "as is" without warranties of any kind. The authors make no guarantees regarding its accuracy, reliability, or fitness for any particular purpose outside of the conducted experiments. You use this software at your own risk. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising from the use of, or inability to use, the software.

---

## Important Note

An earlier version of this work had an implementation issue in the SNIP pruning experiments, affecting the presented results, which led to the corrections incorporated in this version.

---

## Running Experiments

Run the experiments by executing all cells of the provided netbooks top to bottom. 

## Mapping to Paper Results

| Paper result | File |
|---|---|
| Table 1 | `mainMNNs*.ipynb` |
| Table 2 | `mainMNNs*.ipynb`<br>`resnet_20_experiments*.ipynb` |
| Table 3 | `snip_experiments_corrected*.ipynb` <br> `snip_resnet_experiments_corrected*.ipynb` |
| Table 4 <br> Figure 5 | `mainMNNs*.ipynb` |

---

## Compute Requirements

All experiments were executed on:

- NVIDIA GeForce RTX 2080 Ti with 12 GB memory
- NVIDIA GeForce RTX 3060 with 12 GB memory

A single GPU with approximately 12 GB of memory is sufficient to reproduce the results.

---

## Citation

If you use this code, please cite:

```bibtex
@article{fotopoulos2025training,
  title={Training Deep Morphological Neural Networks as Universal Approximators},
  author={Fotopoulos, Konstantinos and Maragos, Petros},
  journal={arXiv preprint arXiv:2505.09710},
  year={2025}
}

```

---

## Acknowledgments

The authors thank Panagiotis Papanikolaou and Assistant Professor G. Tzimpragos for identifying an implementation issue in the SNIP pruning experiments, which led to the corrections incorporated in this version.
