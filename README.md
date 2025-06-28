# Comparing Ensemble Learning Methods for Imbalanced Data

This repository explores and compares various ensemble learning methods based on decision trees for handling imbalanced datasets. The project includes implementations, experiments, and evaluations of different techniques to address class imbalance problems in machine learning.

## Table of Contents
- [Introduction](#introduction)
- [Methods](#methods)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Imbalanced datasets are common in real-world applications, where the number of instances in one class significantly outweighs the others. This can lead to biased models that favor the majority class. Ensemble learning methods based on decision trees (e.g., Random Forest, Gradient Boosting) are particularly effective for such scenarios. This project compares the performance of these methods on imbalanced data, focusing on metrics like precision, recall, F1-score, and AUC-ROC.

## Methods
The following ensemble learning methods are compared:
1. **Random Forest**
2. **Gradient Boosting Machines (GBM)**
3. **XGBoost**
4. **LightGBM**
5. **CatBoost**
6. **Balanced Random Forest** (with built-in class weighting)
7. **RUSBoost** (Random UnderSampling + Boosting)

Each method is evaluated with and without techniques to handle imbalance, such as:
- Class weighting
- Oversampling (SMOTE)
- Undersampling

## Dataset
The experiments are performed on publicly available imbalanced datasets (e.g., from [Kaggle](https://www.kaggle.com/) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/)). You can replace or add your own datasets in the `data/` folder.

Example datasets:
- Credit Card Fraud Detection
- Medical Diagnosis (e.g., rare diseases)
- Customer Churn Prediction

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mahajialirezaei/cmp-learning-methods.git
   cd cmp-learning-methods
   ```


## Results
Preliminary results show that:
- **LightGBM** and **XGBoost** with class weighting perform well on highly imbalanced data.
- **Balanced Random Forest** is robust for moderate imbalance.
- **SMOTE + Gradient Boosting** improves recall for minority classes.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss the proposed changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


### Key Features:
1. Clear structure with sections for methods, datasets, and results.
2. Emphasis on imbalanced data handling (SMOTE, class weights, etc.).
3. Includes popular tree-based ensemble methods (XGBoost, LightGBM, etc.).
4. Ready for customization (add your datasets/metrics).


## 🛠 Developer

Developed by [Mohammad Amin Haji Alirezaei](https://github.com/mahajialirezaei)
Feel free to ⭐️ this repo or open an issue if you'd like to contribute or have questions!
