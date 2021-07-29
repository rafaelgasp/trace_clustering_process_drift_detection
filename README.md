# Concept Drift Detection and Localization in Process Mining

Repository of code from the experiments in the paper "Concept Drift Detection and Localization in Process Mining: An Integrated and Efficient Approach Enabled by Trace Clustering"

## Content

The **source** folder contains files with the code that are used as a package in the Jupyter notebooks, implementing the parsing of the dataset, vector representation, trace clustering, drift detection and localization.

- **Drift Detection and Localization.ipynb** presents the usage of the trace clustering and drift detection algorithm. Shows examples of execution for some logs and can be used to experiment with parameters and testing. The localization method can be used after a drift is localized.

- **Experiments Execution.ipynb** presents the pipeline used for running systematic experiments that detects, localize drifts and calculate metrics and results. 

```
📦trace_clustering_process_drift_detection
 ┣ 📂source
 ┃ ┣ 📜drift_detection.py
 ┃ ┣ 📜drift_localization.py
 ┃ ┣ 📜log_representation.py
 ┃ ┣ 📜offline_streaming_clustering.py
 ┃ ┣ 📜parse_mxml.py
 ┃ ┗ 📜plots.py
 ┣ 📜Drift Detection and Localization.ipynb
 ┣ 📜Experiments Execution.ipynb
 ┗ 📜requirements.txt
```


## Dataset

The dataset utilized in this work is presented in `Maaradji, A., Dumas, M., La Rosa, M. and Ostovar, A. 2015. Business Process Drift.  Dataset, Queensland Univ. of Technology, Australia`. Available at https://data.4tu.nl/articles/Business_Process_Drift/12712436.



## Citing this work

Rafael Gaspar de Sousa, Sarajane Marques Peres, Marcelo Fantinato, and Hajo Alexander Reijers. 2021. Concept drift detection and localization in process mining: an integrated and efficient approach enabled by trace clustering. In Proceedings of the 36th Annual ACM Symposium on Applied Computing (SAC '21). Association for Computing Machinery, New York, NY, USA, 364–373. DOI:https://doi.org/10.1145/3412841.3441918
