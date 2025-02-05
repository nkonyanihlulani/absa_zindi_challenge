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

--------

In a previous article, we shared how our model for the Absa Zindi Challenge climbed from 12th place to 2nd, driven by strategic data engineering, feature selection, and hyperparameter optimization. That phase emphasized the exploratory and model-building aspects of machine learning. Now, we’re excited to dive into the next critical step: scaling machine learning projects effectively using Amazon SageMaker.

Leveraging Amazon SageMaker for Scalable ML Workflows
Amazon SageMaker, a fully managed service, offers a comprehensive suite of tools to streamline machine learning workflows at a relatively low cost. Its ability to seamlessly scale data science workloads by leveraging AWS infrastructure was pivotal for our project. We specifically utilized SageMaker Studio, an integrated development environment, to kickstart and manage our inference pipelines, enabling us to transition from experimentation to production with greater efficiency.

Here’s a link to the full implementation:
https://github.com/nkonyanihlulani/absa_zindi_challenge/blob/main/absa_inference_pipelines.ipynb

Previous article:
https://medium.com/@nkonyanihlulani/absa-zindi-challenge-income-prediction-6e94817e94e4

Our End-to-End Pipeline
Here’s a visual representation of the pipeline we built:


Key Components of the Pipeline
1. Preprocessing
We employed Recursive Feature Elimination (RFE) with a base estimator of a Random Forest Regressor. RFE with cross-validation is computationally intensive, so I wouldn’t recommend running this pipeline regularly unless you’re focused on discovering new features as fresh data comes in.

2. Hyperparameter Tuning
Instead of a conventional training step, we used SageMaker’s TuningStep to optimize hyperparameters. We focused on minimizing the Root Mean Squared Error (RMSE). This process was time-consuming, taking approximately an hour and 21 minutes to complete 100 iterations.

The tuning job summary highlights the best-performing model with an RMSE of 5572.60.These parameters were optimized using SageMaker’s efficient hyperparameter tuning capabilities, leveraging the `SKLearn` estimator within the `sagemaker.sklearn.estimator` module. The job execution details, including logs, were managed via CloudWatch Logs, enhancing our ability to track progress and debug effectively. The tuning step ouput:


3. Model Evaluation
The best model achieved an impressive RMSE of 6088 on the test set.

4. Conditional Step
We introduced a conditional check to ensure model performance consistency. If the RMSE was ≤ 6500, we proceeded to register the model, create endpoints, and carry out transform jobs. If the RMSE exceeded this threshold, the pipeline moved into a fail-safe step for review.

Custom Scripts and Challenges FInding Such Implementations online
Our implementation relied on custom scripts for preprocessing, training, evaluation, and inference. The bespoke nature of these scripts led to several debugging sessions, often resolved through community forums and Stack Overflow. The inference script required particular attention to ensure output consistency aligned with the competition’s submission requirements.

Enhanced Monitoring with Logging
To improve debugging and traceability, we implemented comprehensive logging. This facilitated better tracking of job executions through CloudWatch Logs.

Final Thoughts
Scaling machine learning models from prototypes to production can be complex, especially with custom deployments. However, leveraging tools like Amazon SageMaker, coupled with robust monitoring and custom scripting, made our journey both efficient and insightful. We hope this article provides valuable insights for your own ML scaling endeavors.

Feel free to connect and share your thoughts or experiences with similar projects!

By revisiting the dataset and applying these techniques, I was able to make a tangible improvement to my model's performance. It's always exciting to see how small adjustments can lead to big improvements, and this project served as a great reminder of the importance of continuously iterating and refining our models.

## Next Steps (Going to Production):
The next phase of this project involves setting up a pipeline to automate preprocessing, training, hyperparameter optimization, evaluation, and deployment of the model. We are going to leverage SageMaker Inference Pipelines.
