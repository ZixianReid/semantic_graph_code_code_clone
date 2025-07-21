# 🔬 Experimental Platform for the Paper  
### **"AST-Enhanced or AST-Overloaded? The Surprising Impact of Hybrid Graph Representations on Code Clone Detection"**  
**Authors**: Zixian Zhang and Takfarinas Saber  
**arXiv**: [https://arxiv.org/abs/2506.14470](https://arxiv.org/abs/2506.14470)

---

## 📁 Dataset Description

The dataset is formatted as a CSV-like file (`clone_label.txt`) with the following structure:

| Column | Name              | Description                                                                 |
|--------|-------------------|-----------------------------------------------------------------------------|
| 1      | `code_file_1`     | Path to the first code snippet                                              |
| 2      | `code_file_2`     | Path to the second code snippet                                             |
| 3      | `clone_label`     | `0`: Non-clone, `1`: Clone                                                  |
| 4      | `split_label`     | `0`: Train, `1`: Test, `2`: Validation                                      |
| 5      | `dataset_label`   | `0`: BCB, `1`: GCJ, `2`: GPTCloneBench                                      |
| 6      | `clone_type`      | `0`: None, `1`: T1, `2`: T2, `3`: VST3, `4`: ST3, `5`: MT3, `6`: T4         |
| 7      | `similarity_score`| Numeric similarity score between code snippets                              |

### 📦 Dataset Sources

- **BCB Dataset**:  
  Sampled and merged with BigCloneBenchEval. Only pairs available in BigCloneBenchEval are retained to ensure clone type and similarity label availability.

- **GCJ Dataset**:  
  Preprocessed Google Code Jam dataset. All `import` and `package` statements are removed to ensure consistency with other datasets.

- **GPTCloneBench**:  
  Java-based dataset derived from SemanticCloneBench.  
  - Positive samples = semantic clones  
  - Negative samples = syntactic clones  
  - **Used for testing only**, due to lack of non-clone (false) pairs.

📌 **Important:**  
Current implementation supports **only BCB**.  

📥 **Download the dataset here**:  
[https://figshare.com/s/a7517be0234769b2fa5b](https://figshare.com/s/a7517be0234769b2fa5b)

### 📂 Setup the Dataset

```bash
mkdir -p /data/data_source
cd data
mv data_java.zip /data/data_source
cd /data/data_source
unzip data_java.zip
```

---

## 💻 Requirements

### Software & Hardware

- OS: Linux (Ubuntu 20.04 recommended)
- Python: 3.8+
- Conda
- NVIDIA Driver Version: **545.29.06**
- CUDA Version: **12.3**

---

## ⚙️ Installation

### Step 1: Create Environment

```bash
conda env create -f environment.yml
conda activate codeclone
```

### Step 2: Install PyTorch 2.3.1

Follow the instructions from [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/).

### Step 3: Install PyTorch Geometric (CUDA 12.1, Torch 2.3.0)

```bash
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

### ✅ Test Installation

To confirm the installation works, run:

```bash
cd run
sh run_gat.sh
```

✅ Expected Output (shortened):

```
Model: GAT
Test Accuracy: 0.85
Clone Type Detection Accuracy: ...
```

This confirms that the code runs correctly and produces meaningful results.

### 🧪 Tested Configuration

- OS: Ubuntu 20.04  
- GPU: NVIDIA RTX 3090  
- CUDA: 12.3  
- PyTorch: 2.3.1  
- Python: 3.8.18  
- torch-geometric: 2.5.0  

---

## ▶️ Running the Experiments

All execution scripts are located in the [`run`](./run) directory.

### 🔁 Reproduce Results

Each script runs a specific model:

- **GAT**:
  ```bash
  cd run
  sh run_gat.sh
  ```

- **GMN**:
  ```bash
  cd run
  sh run_gmn.sh
  ```

- **GGNN**:
  ```bash
  cd run
  sh run_ggnn.sh
  ```

📂 Output results will be stored in the `run/` directory.

---

## 📑 Artifact Contents

| Directory/File       | Description                                      |
|----------------------|--------------------------------------------------|
| `run/`               | Shell scripts to reproduce experiments           |
| `model/`             | Model implementation files (GAT, GGNN, GMN, etc.)|
| `data/`              | Contains `data_java.zip` and extraction script   |
| `clone_label.txt`    | Annotation file with clone labels and splits     |
| `installization.md`  | Environment setup instructions (also in README)  |
| `environment.yml`    | Conda environment definition                     |

---

## 📜 Citation

```bibtex
@misc{zhang2025astenhancedastoverloadedsurprisingimpact,
  title     = {AST-Enhanced or AST-Overloaded? The Surprising Impact of Hybrid Graph Representations on Code Clone Detection},
  author    = {Zixian Zhang and Takfarinas Saber},
  year      = {2025},
  eprint    = {2506.14470},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url       = {https://arxiv.org/abs/2506.14470}
}
```
