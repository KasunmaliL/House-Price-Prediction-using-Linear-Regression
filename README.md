# House Price Prediction using Linear Regression

A complete end-to-end Machine Learning project that predicts house prices using the Boston Housing Dataset.  
This project covers data preprocessing, exploratory data analysis (EDA), model building, evaluation, and optimization using advanced regression techniques.

## Problem Statement
The objective of this project is to predict the median value of owner-occupied homes (MEDV) based on several explanatory variables such as crime rate, number of rooms, property tax rate, proximity to the Charles River, and more.  
Using the Boston Housing dataset, the project aims to:
• Understand relationships between housing attributes and prices  
• Build a predictive model using Linear Regression  
• Evaluate and optimize model performance  
• Compare Linear Regression with Polynomial, Ridge, and Lasso models  

## Dataset
**Boston Housing Dataset**  
Key features include:  
• crim: Per capita crime rate  
• rm: Average number of rooms  
• lstat: % of lower-status population  
• tax: Property tax rate  
• chas: Charles River dummy variable (1 = yes, 0 = no)  
• medv: Median home value (target variable)

---

## Project Workflow

### 1. Data Collection & Loading
• Load the CSV dataset into a pandas DataFrame  
• View structure, shape, and summary statistics  
• Identify missing values and inconsistencies  

### 2. Data Preprocessing
• Handle missing values  
• Encode categorical variables (e.g., CHAS)  
• Normalize/scale numerical features using StandardScaler  
• Split dataset into train and test sets  

### 3. Exploratory Data Analysis (EDA)
Visualize important relationships such as:  
• Price vs. Rooms (RM)  
• Price vs. LSTAT  
• Price vs. CHAS  
• Correlation Heatmap  
• Distribution plots and boxplots  
• Scatterplots to observe linearity  

**Insights:**  
✔ Prices increase with number of rooms  
✔ Prices decrease with higher LSTAT  
✔ Strong negative correlation between LSTAT and MEDV  
✔ RM is one of the strongest positive predictors  

### 4. Model Building
Implement multiple regression models:  
• Linear Regression (baseline model)  
• Fit using training data  
• Predict house prices for unseen test data  

### 5. Model Evaluation
Evaluate model performance using:  
• Mean Absolute Error (MAE)  
• Mean Squared Error (MSE)  
• R² Score  

Plots:  
• Actual vs Predicted Price scatter plot  

### 6. Model Optimization 
Experiment with improved models:  
• Polynomial Regression (degree 2 and above)  
• Ridge Regression (L2 regularization)  
• Lasso Regression (L1 regularization / feature selection)  

Conduct hyperparameter tuning (alpha selection) and compare:  
• MAE  
• MSE  
• R² Score  

**Conclusion:**  
Polynomial Regression usually improves predictive accuracy and Ridge and Lasso help control overfitting and handle multicollinearity.

### 7. Saving the Model
Save trained model for deployment:  
joblib.dump(model, "linear_regression_model.pkl")


---

## Results Summary
• Linear Regression creates a simple, interpretable model  
• Polynomial Regression captures non-linear patterns, improving accuracy  
• Ridge & Lasso offer regularization benefits  
• Best model depends on the balance between accuracy and interpretability  

---

## Project Files
• Project_2_House_Price_Prediction_using_Linear_Regression.ipynb  
• BostonHousing.csv  
• linear_regression_model.pkl (saved model)  
• README.md  
• Project Report  

---

## Key Takeaways
• Learned the complete workflow of a supervised ML project  
• Understood how to clean and preprocess real-world data  
• Improved skills in visualization and feature analysis  
• Built and evaluated a regression model using scikit-learn  
• Explored advanced techniques such as Ridge, Lasso, and Polynomial Regression  
• Strengthened understanding of how numerical features influence house prices  




