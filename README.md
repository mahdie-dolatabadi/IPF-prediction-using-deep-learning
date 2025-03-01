# IPF-prediction-using-deep-learning
### 📄 Related Paper: [![arXiv](https://img.shields.io/badge/arXiv-YourPaperID-red)](https://arxiv.org/abs/YourPaperID)
### 👋 Introduction 

Welcome to the IPF Prediction Using Deep Learning project! Idiopathic pulmonary fibrosis (IPF) is a progressive disease that irreversibly transforms lung tissue into rigid fibrotic structures, leading to debilitating symptoms such as shortness of breath and chronic fatigue. The heterogeneity and complexity of this disease, particularly regarding its severity and progression rate, have made predicting its future course a complex and challenging task. Besides, traditional diagnostic methods based on clinical evaluations and imaging have limitations in capturing the disease’s complexity.

This repository presents a deep learning approach to predict the progression of IPF using the Kaggle Pulmonary Fibrosis Progression dataset, which includes computed tomography images and clinical information. By leveraging a proposed context-aware sequential-parallel hybrid transformer model and enriching predictions with clinical information, the method achieved a Laplace Log-Likelihood score of −6.508, outperforming previous techniques. This work highlights the potential of advanced deep learning models in providing accurate predictions, aiding in the diagnosis and management of IPF. The overview of the proposed method is shown in figure below:
![Alt Text](images/overview.png)
### 📂 Repository Structure
```plaintext
IPF-prediction-using-deep-learning/
│── region_growing/ # Folder for the region growing technique
│   ├── matlab_code/ # MATLAB scripts for region growing
│   ├── python_code/ # Python implementation (if any)
│   ├── README.md # Explanation of the region growing method
│── model/ # Main model and configuration files
│   ├── config/ # Configuration files (hyperparameters, settings, etc.)
│   ├── main_code/ # Main scripts for training/testing
│   ├── utils/ # Utility functions and helper scripts
│   ├── results/ # Model outputs, logs, and performance reports
│── dataset/ # Dataset files or dataset links
│── diagrams/ # Workflow diagrams in PNG format
│── README.md # Main project documentation
│── .gitignore # Ignore unnecessary files



