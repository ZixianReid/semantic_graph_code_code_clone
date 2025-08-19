# Create and activate environment
conda env create -f environment.yml
conda activate codeclone

# Install PyTorch (example for CUDA 12.1, adjust if needed)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/pyg_lib-0.4.0%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt23cu121-cp38-cp38-linux_x86_64.whl

pip install pyg_lib-0.4.0+pt23cu121-cp38-cp38-linux_x86_64.whl 
pip install torch_scatter-2.1.2+pt23cu121-cp38-cp38-linux_x86_64.whl 
pip install torch_sparse-0.6.18+pt23cu121-cp38-cp38-linux_x86_64.whl 
pip install torch_spline_conv-1.2.2+pt23cu121-cp38-cp38-linux_x86_64.whl  

# Finally install PyG
pip install torch_geometric