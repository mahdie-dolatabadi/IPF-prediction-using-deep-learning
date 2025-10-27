# IPF-prediction-using-deep-learning
### 📄 Related Paper: [![BSOC](https://github.com/mahdie-dolatabadi/IPF-prediction-using-deep-learning/blob/main/images/BSPC.png =116x20)](https://doi.org/10.1016/j.bspc.2025.108958) [![arXiv](https://img.shields.io/badge/arXiv-2503.00386-red)](https://arxiv.org/abs/2503.00386)
### 👋 Introduction 

Welcome to the IPF Prediction Using Deep Learning project! Idiopathic pulmonary fibrosis (IPF) is a progressive disease that irreversibly transforms lung tissue into rigid fibrotic structures, leading to debilitating symptoms such as shortness of breath and chronic fatigue. The heterogeneity and complexity of this disease, particularly regarding its severity and progression rate, have made predicting its future course a complex and challenging task. Besides, traditional diagnostic methods based on clinical evaluations and imaging have limitations in capturing the disease’s complexity.

This repository presents a deep learning approach to predict the progression of IPF using the Kaggle Pulmonary Fibrosis Progression dataset, which includes computed tomography images and clinical information. By leveraging a proposed context-aware sequential-parallel hybrid transformer model and enriching predictions with clinical information, the method achieved a Laplace Log-Likelihood score of −6.508, outperforming previous techniques. This work highlights the potential of advanced deep learning models in providing accurate predictions, aiding in the diagnosis and management of IPF. The overview of the proposed method is shown in figure below:

![Alt Text](images/overview.png)
### 📂 Repository Structure
```plaintext
IPF-PREDICTION-USING-DEEP-LEARNING  
│── images/  
│   ├── overview.png  
│  
│── main/  
│   ├── results_slopes/  
│   │   ├── september8th/    
│   ├── vit_b_16simplestep2hybrid.txt  
│   ├── configs.py  
│   ├── dataset.py   
│   ├── main.ipynb  
│   ├── model.py  
│   ├── train_test.py  
│   ├── utils.py  
│  
│── RegionGrowing/  
│   ├── images/  
│   │   ├── bottom-image.png  
│   │   ├── bottom-mask.png  
│   │   ├── down.dcm  
│   │   ├── middle-image.png  
│   │   ├── middle-mask.png  
│   │   ├── middle.dcm  
│   │   ├── up-image.png  
│   │   ├── up-mask.png  
│   │   ├── up.dcm  
│   ├── load_display_dicom.m  
│   ├── main.m  
│   ├── regiongrowing.m  
│  
├── images.csv 
│── README.md 
│── requirements.txt
``` 
### 📦 Installation
Provide instructions to set up the environment:
```plaintext
git clone https://github.com/mahdie-dolatabadi/IPF-prediction-using-deep-learning.git
cd IPF-PREDICTION-USING-DEEP-LEARNING
pip install -r requirements.txt
```
### 🗂️ Dataset
To download the dataset, use the following instructions:
1. Install the Kaggle Python package:
```plaintext
pip install kaggle
```
2. You need a Kaggle API key to authenticate:
> - Go to [Kaggle](https://www.kaggle.com/).  
> - Click on your profile (top right) > **Account**.  
> - Scroll down to the **API** section.  
> - Click **Create New API Token** – this will download a `kaggle.json` file. 

3. Move the downloaded kaggle.json file to your Kaggle configuration directory:
```plaintext
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # Secure file permissions
```
4. Download a Kaggle Competition Dataset and extract it:
```plaintext
kaggle competitions download -c osic-pulmonary-fibrosis-progression
unzip osic-pulmonary-fibrosis-progression.zip
```

## Citation

If you find our code or paper useful, please cite as:

```bibtex
@article{dolatabadi2025prognostic,
  title={Prognostic Model for Idiopathic Pulmonary Fibrosis Using Context-Aware Sequential-Parallel Hybrid Transformer and Enriched Clinical Information},
  author={Dolatabadi, Mahdie and Nabavi, Shahabedin and Moghaddam, Mohsen Ebrahimi},
  journal={arXiv preprint arXiv:2503.00386},
  year={2025}
}





