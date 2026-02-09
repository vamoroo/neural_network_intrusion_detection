# neural_network_intrusion_detection
Develop a system capable of classifying 5 types of network attacks (DoS, Probe, R2L, U2R, Normal) with high accuracy using deep learning techniques on the NSL-KDD dataset.

# Network Attack Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-red)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Intrusion Detection System using Deep Learning techniques to classify network connections into 5 categories: Normal, DoS, Probe, R2L, and U2R.**

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Code Structure](#-code-structure)
- [Tools & Technologies](#-tools--technologies)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🎯 Project Overview

This project develops an intelligent Network Intrusion Detection System (IDS) based on Deep Learning. Using the NSL-KDD dataset (reduced version of KDD Cup 1999), the system learns to distinguish normal network traffic from various attack categories:

- **Normal**: Legitimate network connections
- **DoS (Denial of Service)**: Denial of service attacks
- **Probe**: Network reconnaissance attempts
- **R2L (Remote to Local)**: Unauthorized remote access
- **U2R (User to Root)**: Privilege escalation attacks

## ✨ Features

- **Advanced preprocessing** of network data
- **Intelligent feature engineering** (41 features)
- **Multi-class classification** with Deep Learning
- **Data visualization** and distribution analysis
- **Automated pipeline** for data preparation
- **Production-ready export** in CSV format

## 📊 Dataset

### NSL-KDD Dataset (KDD Cup 1999 - 10%)
The project uses the 10% version of the famous KDD Cup 1999 dataset, widely used in network security research.

**Key characteristics:**
- 494,021 network connection samples
- 41 features per sample
- 5 target classes (Normal + 4 attack types)
- Labeled data for supervised learning

**Feature structure:**
- **Basic**: duration, protocol, service, flag
- **Content**: root access attempts, file access counts
- **Traffic**: host/server connection counts
- **Time-based statistics**: traffic over 2s windows

# Network Attack Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-red)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Intrusion Detection System using Deep Learning techniques to classify network connections into 5 categories: Normal, DoS, Probe, R2L, and U2R.**

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Code Structure](#-code-structure)
- [Tools & Technologies](#-tools--technologies)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🎯 Project Overview

This project develops an intelligent Network Intrusion Detection System (IDS) based on Deep Learning. Using the NSL-KDD dataset (reduced version of KDD Cup 1999), the system learns to distinguish normal network traffic from various attack categories:

- **Normal**: Legitimate network connections
- **DoS (Denial of Service)**: Denial of service attacks
- **Probe**: Network reconnaissance attempts
- **R2L (Remote to Local)**: Unauthorized remote access
- **U2R (User to Root)**: Privilege escalation attacks

## ✨ Features

- **Advanced preprocessing** of network data
- **Intelligent feature engineering** (41 features)
- **Multi-class classification** with Deep Learning
- **Data visualization** and distribution analysis
- **Automated pipeline** for data preparation
- **Production-ready export** in CSV format

## 📊 Dataset

### NSL-KDD Dataset (KDD Cup 1999 - 10%)
The project uses the 10% version of the famous KDD Cup 1999 dataset, widely used in network security research.

**Key characteristics:**
- 494,021 network connection samples
- 41 features per sample
- 5 target classes (Normal + 4 attack types)
- Labeled data for supervised learning

**Feature structure:**
- **Basic**: duration, protocol, service, flag
- **Content**: root access attempts, file access counts
- **Traffic**: host/server connection counts
- **Time-based statistics**: traffic over 2s windows

1. Exploration and Preprocessing

# In Jupyter Notebook
from src.data_preprocessing import load_dataset_and_clean

# Load and clean dataset
features, labels = load_dataset_and_clean('data/dataset_finale.csv')

# Dataset information
from src.utils import information_about_dset
shape, num_classes = information_about_dset(features, labels)
print(f"Shape: {shape}, Classes: {num_classes}")

2. Run Complete Pipeline
# Execute main notebook
jupyter notebook notebooks/dl_project.ipynb

3. Main Steps
The pipeline follows this sequence:

Loading: Read compressed dataset

Mapping: Convert attacks to 5 categories

Cleaning: Handle missing values

Separation: Numeric vs categorical features

Visualization: Analyze distributions

Export: Save ready-to-use dataset

📈 Results
Exploratory Analysis
Class distribution: Visualization of Normal/Attack proportions

Correlations: Analysis of feature relationships

Important features: Identification of most discriminative features

Target Performance
Accuracy: > 95% on test data

Recall: > 90% for all classes

F1-Score: > 92% overall

Inference time: < 100ms per connection

🧩 Code Structure

Main Functions
# 1. Loading and cleaning
def load_dataset_and_clean(path):
    """
    Load and clean the dataset
    Returns: features (DataFrame), labels (Series)
    """

# 2. Dataset information
def information_about_dset(X, y):
    """
    Analyze dataset structure
    Returns: shape, number of classes
    """

# 3. Feature separation
def split_into_cat_and_num(X):
    """
    Separate numeric and categorical features
    Returns: df_numeric, df_categorical
    """

Main Notebook (dl_project.ipynb)
Import required libraries

Build dataset with feature names

Convert targets to attack types

Export to CSV for future use

Detailed analysis of data

Visualization of categorical features

🔧 Tools & Technologies
Languages & Libraries
Python - Main development language

Pandas - Data manipulation and analysis

NumPy - Numerical computations and matrix operations

Environment
Jupyter Notebook - Interactive development environment

Google Colab - Cloud execution alternative

Git - Version control

Data Processing
StandardScaler - Normalization of numerical features

LabelEncoder/OneHotEncoder - Encoding of categorical variables

train_test_split - Train/test/validation split

Matplotlib & Seaborn - Data visualization

Scikit-learn - Preprocessing and model evaluation

🤝 Contributing
Contributions are welcome! To contribute:

Fork the project

Create a branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Contribution Guidelines
Follow existing code style

Add tests for new features

Update documentation

Ensure all tests pass

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

👤 Author
Vamoro CAMARA

GitHub: @vamoroo

LinkedIn: vamoro-camara

Email: cvamorocamson@gmail.com

🙏 Acknowledgments
University of California, Irvine for the NSL-KDD dataset

Open source community for tools and libraries

All contributors and testers

📚 References
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set.

NSL-KDD Dataset. UCI Machine Learning Repository.

TensorFlow Documentation. Multi-class classification.

