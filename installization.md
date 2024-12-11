# Environment Install

## Basic environment requirement

Nvdiai Driver Version: 545.29.06

Cuda Version: 12.3

## Conda package install

```
`conda env create -f environment.yml`
```

## Pytorch and torch_ geometric install

### Torch 2.3.1

please find instructions from [here](https://pytorch.org/get-started/previous-versions/).

### Torch Geometric

Please find insturctions from [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

If you have the same cuda and torch version with above. you can cp the following commands

```
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/pyg_lib-0.4.0%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
pip3 install pyg_lib-0.4.0+pt23cu121-cp38-cp38-linux_x86_64.whl 
 pip3 install torch_scatter-2.1.2+pt23cu121-cp38-cp38-linux_x86_64.whl 
pip3 install torch_sparse-0.6.18+pt23cu121-cp38-cp38-linux_x86_64.whl 
pip3 install torch_spline_conv-1.2.2+pt23cu121-cp38-cp38-linux_x86_64.whl  
pip3 install torch_geometric




```
