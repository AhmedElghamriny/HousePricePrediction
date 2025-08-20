# House Price Prediction Using Machine Learning

## Overview
This project implements a comprehensive house price prediction system using various machine learning algorithms. The analysis was conducted in Google Colab using a Kaggle dataset containing residential property information. Through extensive exploratory data analysis and model comparison, XGBoost emerged as the best-performing algorithm for accurate price predictions.

## Dataset
- **Source**: [Kaggle House Price Dataset](https://www.kaggle.com/datasets/shree1992/housedata)
- **Description**: The dataset contains various features of residential properties including structural characteristics, location details, and sale prices
- **Size**: Contains information about house sales with multiple features affecting price

## Project Structure
```
house-price-prediction/
│
├── house_price_prediction.ipynb   # Main notebook with all code
├── data/
│   └── housedata.csv              # Dataset from Kaggle
├── models/
│   └── best_model.pkl             # Saved XGBoost model
└── README.md                      # Project documentation
```

## Technologies Used
- **Platform**: Google Colab
- **Programming Language**: Python 3.x
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `tensorflow/keras`
  - Model Tuning: `GridSearchCV`

## Methodology

### 1. Exploratory Data Analysis (EDA)
Comprehensive analysis performed to understand the dataset:
- Distribution analysis of target variable (house prices)
- Feature correlation analysis using heatmaps
- Identification of outliers and anomalies
- Missing value assessment
- Statistical summaries of numerical and categorical features
- Visualization of relationships between features and target variable

### 2. Extract, Transform, Load (ETL)
Data preprocessing pipeline implemented:
- **Data Cleaning**: Handled missing values using appropriate imputation strategies
- **Encoding**: Converted categorical variables using appropriate encoding techniques
- **Scaling**: Normalized numerical features for neural network implementation
- **Outlier Treatment**: Applied statistical methods to handle extreme values
- **Feature Selection**: Identified and retained most relevant features

### 3. Model Development

#### Models Implemented:
1. **Random Forest Classifier**
   - Ensemble learning method using multiple decision trees
   - Implemented with various n_estimators and max_depth configurations
   - Feature importance analysis conducted

2. **XGBoost with GridSearchCV**
   - Gradient boosting framework optimized for performance
   - Hyperparameter tuning performed using GridSearchCV
   - Parameters tuned: learning_rate, max_depth, n_estimators, subsample
   - **Best performing model** with highest accuracy

3. **Neural Networks**
   - Deep learning approach using Pytorch
   - Activation functions: ReLU for hidden layers, linear for output
   - Optimizer: Adam with learning rate scheduling

### 4. Model Evaluation
Performance metrics used:
- **Mean Squared Error (MSE)**
- **R² Score**
