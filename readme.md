# **Experimental Platform for Paper:**

## *"AST-Enhanced or AST-Overloaded? The Surprising Impact of Hybrid Graph Representations on Code Clone Detection"*

---

## **Dataset**

The dataset includes a `clone_label.txt` file structured as follows, with columns separated by commas:


| Column Index    | Column 1      | Column 2      | Column 3                   | Column 4                               | Column 5                               | Column 6                                                            | Column 7           |
| --------------- | ------------- | ------------- | -------------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------------------------------------- | ------------------ |
| **Field**       | `code_file_1` | `code_file_2` | `clone_label`              | `split_label`                          | `dataset_label`                        | `clone_type`                                                        | `similarity_score` |
| **Description** | File 1        | File 2        | `0`: Non-clone, `1`: Clone | `0`: Train, `1`: Test, `2`: Validation | `0`: BCB, `1`: GCJ, `2`: GPTCloneBench | `0`: None, `1`: T1, `2`: T2, `3`: VST3, `4`: ST3, `5`: MT3, `6`: T4 | Similarity score   |

### **Dataset Details**

1. **BCB Dataset**

   - We utilize a sampled dataset from `()` and merge it with BigCloneBenchEval.
   - The merging rule retains code pairs present in BigCloneBenchEval, ensuring clone type and similarity score availability.
2. **GCJ Dataset**

   - The dataset is sourced from `()`.
   - To ensure fair comparison, we remove all import and package statements, maintaining a consistent structure with BCB and GPTCloneBench.
3. **GPTCloneBench**

   - This dataset, provided by `()`, focuses on Java code.
   - It is derived from SemanticCloneBench, where:
     - False samples are syntactic clone pairs.
     - True samples are semantic clone pairs.
   - Since it lacks false clone pairs, we use it **only for testing**, categorizing them into different clone types.

🚨 **Note:** The current experiments are implemented **only on the BCB dataset**. Dowloading dataset from [here](https://figshare.com/s/a7517be0234769b2fa5b).
📦 **Before running experiments, unzip** `data_java.zip`, following:

```bash
mkdir /data/data_source
cd data
mv data_java.zip /data/data_source
unzip data_java.zip
```

---

## **Installation**

Follow the instructions in [installation guide](installization.md).

---

## **Running Experiments**

All scripts are located in the [`run`](run) directory.

### **Reproduce Experiments**

Run the following scripts to replicate results for different models:

#### **GAT**

```bash
cd run
sh run_gat.sh
```

#### **GMN**

```bash
cd run
sh run_gmn.sh
```

#### **GGNN**

```bash
cd run
sh run_ggnn.sh
```

#### **GMN**

```bash
cd run
sh run_gmn.sh
```

After running, the experiment result can be found under the folod of [`run`](run) directory.
