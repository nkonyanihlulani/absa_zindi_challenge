# Absa Zindi Challenge: Income Prediction

Two years ago, I participated in the Zindi competition where the task was to predict customer-declared income based on their transactions, account information, and personal details such as gender. The best model I submitted at the time ranked 12th out of 49 active participants, which was enough to earn me a silver medal. Recently, I decided to revisit the dataset to see if I could improve my model and achieve a better performance. After redoing the entire modeling process from scratch, I was able to develop a model that would have earned 2nd place in the competition. 
The dataset and the results can be found here : https://zindi.africa/competitions/absa-customer-income-prediction-challenge

## So, What Did I Do Differently This Time?

### 1. A More In-Depth Exploratory Data Analysis (EDA)
This time, I focused on performing a more thorough EDA. This step allowed me to better understand the nature of the target variable (income) and the key factors that could influence it. A key insight was identifying that we were modeling positive values with a right-skewed distribution. This discovery directly informed the choice of the objective function I used in the XGBoost model. By considering the skewness, I was able to select a more suitable approach for modeling.

### 2. Improved Feature Engineering on Transaction Data
Another major improvement was in the feature engineering phase. We went beyond the basic features from before and created additional ones to capture more detailed information about account types, channel usage, and transaction types. This enhanced the model's ability to learn from the data and ultimately improve its performance.

### 3. Recursive Feature Elimination (RFE)
Previously, we only removed highly correlated features, but this approach didn't fully address the complexity of the model. By adding more variables without much consideration, we assumed they weren't adding value and were merely increasing complexity. This time, I implemented Recursive Feature Elimination (RFE) with cross-validation to automatically determine the optimal number of features. Although computationally expensive, RFE helped identify the most important features and removed irrelevant ones, streamlining the model.

### 4. Bayesian Optimization
In our first attempt, we used Random Search to find the best hyperparameters. While effective, it didn't account for previous combinations of hyperparameters. Bayesian Optimization, on the other hand, takes a more intelligent approach by considering previously explored combinations when deciding which parameters to test next. This method converged faster and produced better results compared to Random Search.

## Model Evaluation Techniques

Along with the improvements in model training, I also implemented several techniques for model diagnostics:

### Feature Importance Analysis
I used the default feature importance method from the sklearn Random Forest implementation. While this method is helpful, it has drawbacks, such as overemphasizing high-cardinality features. To address this, sklearn suggests using permutation-based feature importance when this happens, which I applied for more accurate results.

### SHAP (Shapley Additive Explanations) Analysis
I used SHAP to analyze feature importance, assess where features are dense, and understand the relationships between predicted values and actual feature values. SHAP provided a more interpretable way to understand the contributions of different features to the model's predictions.

### Residual Analysis
Despite achieving a strong score, the residuals displayed heteroscedasticity — meaning the errors weren't constant across different income levels. This is likely due to smaller sample sizes at higher income levels. To address this, one possible approach would be to resample the higher-income groups to increase the number of observations, helping to balance the model's performance across all income levels.

## Additional Tools Used:
- **Amazon SageMaker Studio** — JupyterLab for modeling
- **Amazon S3** — Data storage

## Reflections
This project highlights the power of a solid exploratory data analysis (EDA), thoughtful feature engineering, and advanced hyperparameter optimization techniques in significantly improving a model's performance — even when using the same algorithms. It's a reminder that, in data science, a deeper understanding of the data and refined modeling techniques can lead to better results.

By revisiting the dataset and applying these techniques, I was able to make a tangible improvement to my model's performance. It's always exciting to see how small adjustments can lead to big improvements, and this project served as a great reminder of the importance of continuously iterating and refining our models.

## Next Steps (Going to Production):
The next phase of this project involves setting up a pipeline to automate preprocessing, training, hyperparameter optimization, evaluation, and deployment of the model. We are going to leverage SageMaker Inference Pipelines.
