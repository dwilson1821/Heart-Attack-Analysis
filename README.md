# Predicting Heart Attack Risk using Machine Learning  

## Overview  

Heart attacks (myocardial infarctions) are a leading cause of mortality worldwide, often caused by lifestyle factors. By analyzing medical data, we can predict the likelihood of a heart attack and take preventive measures. In this project, I used **machine learning** to analyze patient data and determine key risk factors contributing to heart attacks.  

I compared multiple machine learning models to determine the best one for predicting heart attack risk, optimizing the model for better accuracy.  

The dataset was obtained from Kaggle: [Heart Attack Analysis and Prediction Dataset](https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset).  

---

## Repository Structure  

```  
Heart_Attack_Analysis/  
│  
├── Analysis.ipynb   			# Main notebook for data analysis & ML models
├── Neural Network Model.ipynb   	# Notebook for Neural Network model  
├── Resources/                     # Data files and supporting materials  
│   ├── heart_attack_data.csv  
└── README.md                      # Project documentation   
```  

---

## Dataset Description  

The dataset consists of medical attributes that help predict the likelihood of a heart attack.  

### **Key Features**  

#### **Numeric Variables:**  
- **Age**: Patient’s age  
- **trtbps**: Resting blood pressure  
- **chol**: Cholesterol level  
- **thalachh**: Maximum heart rate achieved  
- **oldpeak**: ST depression induced by exercise  

#### **Categorical Variables:**  
- **sex**: Gender (1 = Male, 0 = Female)  
- **cp**: Chest pain type  
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)  
- **restecg**: Resting electrocardiogram results  
- **exng**: Exercise-induced angina (1 = Yes, 0 = No)  
- **slp**: Slope of the ST segment  
- **caa**: Number of major vessels colored by fluoroscopy  
- **thall**: Thalassemia blood disorder type  
- **output**: Target variable (1 = Heart attack, 0 = No heart attack)  

---

## Steps in the Analysis  

### **Step 1: Data Preprocessing**  
- Loaded the CSV file into a Pandas DataFrame  
- Checked for missing values and duplicate records  
- Performed data cleaning to ensure accuracy  

### **Step 2: Exploratory Data Analysis (EDA)**  
- **Univariate Analysis**:  
  - Analyzed distribution of numeric and categorical variables  
  - Identified trends in heart attack cases  
- **Bivariate Analysis**:  
  - Compared numeric features with the target variable  
  - Examined categorical feature distributions  

### **Step 3: Machine Learning Models**  
We compared four machine learning models to predict heart attack risk:  

| Model                | Accuracy |  
|----------------------|----------|  
| Logistic Regression | **87%**   |  
| Decision Tree       | 83%       |  
| Random Forest       | **87%**   |  
| Neural Network      | 80%       |  

**Model Selection**:  
- Logistic Regression and Random Forest had the highest accuracy (87%).  
- **Logistic Regression** was chosen due to its interpretability, computational efficiency, and lower risk of overfitting.  

### **Step 4: Model Optimization**  
We tested several optimizations for Logistic Regression:  

| Optimization Technique | Accuracy | Misclassifications |  
|------------------------|----------|-------------------|  
| Increased iterations (1000) | 86% | +1 misclassification |  
| Standard scaling | 86% | +1 misclassification |  
| Feature selection + scaling | 86% | +1 misclassification |  

The **initial Logistic Regression model** was the most accurate and balanced.  

---

## Key Findings  

- **Most Important Risk Factors:**  
  - **Sex**: Males had a higher heart attack risk.  
  - **Chest pain type**: Strong predictor of heart attack.  
  - **Exercise-induced angina**: Increased risk factor.  
  - **ST depression (oldpeak)**: Higher values indicate risk.  
  - **Number of major vessels (caa)**: More vessels increase risk.  
  - **Thalassemia (thall)**: Certain types are associated with higher risk.  

- **Less Significant Factors:**  
  - Age  
  - Resting blood pressure  
  - Cholesterol levels  
  - Fasting blood sugar  
  - ECG results  

---

## How to Run the Project  

### **1. Clone the Repository**  
```bash  
git clone https://github.com/your-username/Heart_Attack_Analysis.git  
cd Heart_Attack_Analysis  
```  

### **2. Install Dependencies**  
```bash  
pip install pandas numpy matplotlib seaborn scikit-learn  
```  

### **3. Run the Notebook**  
- Open `Heart_Attack_Analysis.ipynb` in Jupyter Notebook.  
- Execute all cells to reproduce the analysis and results.  

---

## Tools and Technologies  

- **Python (Pandas, NumPy, Matplotlib, Seaborn)** for data analysis and visualization  
- **Scikit-Learn** for machine learning model implementation  
- **Jupyter Notebook** for interactive data exploration  

---

## Conclusion  

This project demonstrated how machine learning can be applied to **predict heart attack risk**. By analyzing patient data, we identified significant risk factors and optimized models for high accuracy. **Logistic Regression** emerged as the best model due to its performance and interpretability.  

### **Future Improvements**  
- Collect more diverse datasets to improve generalization.  
- Implement deep learning models for advanced feature extraction.  
- Develop an interactive web-based risk prediction tool.  
