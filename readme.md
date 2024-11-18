# Dataset

a clone_label.text file is kept with the data strcutre shown below: where columns are separated with comma


| Column index | Column 1    | Column 2    | Column 3                 | Column 4                 | Column 5                         | Column 6                                        | Column 7         |
| ------------ | ----------- | ----------- | ------------------------ | ------------------------ | -------------------------------- | ----------------------------------------------- | ---------------- |
| Fields       | code_file_1 | code_file_2 | clone_label              | split_label              | dataset_label                    | Clone_type                                      | similarity_score |
| Explaination | xx          | xx          | 0:non_lone,<br />1:clone | 0: train, 1:test, 2: val | 0: bcb, 1: gcj, 2: gptclonebench | 0:none, 1:T1, 2:T2, 3:VST3, 4:ST3, 5:MT3, 6: T4 |                  |

1. For **BCB dataset**, we utlize the sampled datasets provide by (), while merge with BigCloneBeneval, the rule is that keeping the code pair sample that is in BigCloneBeneval where code clone type and code similarity score can be found.
2. For **GCJ dataset**, we adpot the dataset provided by (). For fair comparsion, we remove all import and packages information from this datasets to make it keep same structure with BCB and GPT
3. For **GPTCloneBench**, we adpot the dataset proviede by (). We utlize the Java part of this. Since it is based on Semanticclonebench (), its false sample is syntatic clone pair and the true sample is semantic clone pair, while lacking false clone pair. We use it only as test dataset while classified them into different clone types.
