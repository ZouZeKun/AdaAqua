# AdaAqua: An adversarial domain adaptation neural network that adaptively transfers water-use patterns from data-rich source cities to the data-scarce target city by leveraging users’ Built-Environment and Social-Stratification indicators.

**Paper**： [Cross-City Water Usage Pattern Transferring: An Adversarial Domain Adaptation Deep Learning Method for Estimating Demand in Non-Metered Residential Users]<br>
**Authors**： [Zekun Zou], [Tingchao Yu]<br>

![Graphical Abstract](GraphicalAbstract.jpg)

## 📜 Abstract
> Uncertainty in water usage from the vast number of non-metered users is the major bottleneck to real-time urban water distribution analysis. Inferring water consumption of non-metered users using information from metered users offers a practical means to bypass this data limitation. However, the restricted sensor coverage in local city often fails to capture sufficiently diverse usage patterns, hindering intra-city transfer. To address this challenge, this study proposes AdaAqua, a real-time water usage estimation method for non-metered users based on cross-city water usage pattern transfer. Built upon adversarial domain-adaptive neural networks, AdaAqua transfers abundant water usage information from data-rich source cities while adapting them to data-scarce target city by conditioning on Built-Environmental and Socio-Stratification (BESS) indicators. Evaluation experiments conducted in a data-limited city using three-fold cross-validation show that AdaAqua achieves an average estimation accuracy of 1.5 m³/h and a coefficient of determination of 0.60, demonstrating its effectiveness and reliability for practical urban water management applications.<br>

## 🧠 Model Architecture
![Model Architecture](ModelArchitecture.jpg)  
*Overall architecture of the [AdaAqua] model proposed in the paper*

## 🚀 Qucik Start

### How to Use？

#### **Repository Contents:** This repository contains one data file and six experiment folders

#### **2. Dataset Folder: Stores source domain data (named ZZ) and target domain data (named ZB) separately**
- **h5 files**：Contain data with dimensions (number of residential areas, time steps, features).
- **CSV files**：保存了对应h5文件中小区的用水账单。

#### **3. Experiment Folders: Adjust the file paths in the "Main.ipynb" files within each folder to run the code.**
- **Exp AblationExperiment**：Contains code for ablation studies, demonstrating the necessity of the model architecture.
- **Exp ReplacementExperiment**：Contains code for replacement studies, demonstrating the superiority of the model architecture.
- **Exp DomainAdversairalIntensity**：Uses a domain adversarial strength of 0.1 as an example; adjust the strength based on migration difficulty.
- **Exp OnlyTargetDomain**：Contains experiments using only target domain data, proving the performance improvement from incorporating source domain data
- **Exp SourceSparse**：Contains experiments with 100 source domain samples, verifying the impact of source domain data richness on model performance.
- **Exp TargetSparse**：Contains experiments with 7 target domain samples, verifying the model's robustness under target domain data sparsity.


### Operation Environment
- **Python Version**： `3.12.3` 
- **CUDA Version**： `12.100` 
- **PyTorch Version**： `2.3.0+cu121` 
- **Numpy**： `1.26.4`
- **Pandas**： `2.2.2` 
###  If you have any questions, please contact us at: [*zekunzou@zju.edu.cn*]

