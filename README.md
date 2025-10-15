Customer Retention Prediction
Project Overview

The goal of this project is to build a comprehensive machine learning pipeline that predicts whether a customer will churn (turnover) based on historical customer data. The model is developed to help businesses identify customers who are at high risk of leaving and apply retention strategies accordingly.

The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, and performance evaluation using multiple machine learning algorithms. Models such as Random Forest, XGBoost, Logistic Regression, K-Nearest Neighbors, and Gradient Boosting are trained and compared using various evaluation metrics.

Dataset Description

The dataset used for this project contains the following columns:

Customer_ID: Unique identifier for each customer.

Age: Age of the customer.

Gender: Gender of the customer (Male, Female, Other).

Annual_Income: Annual income of the customer.

Total_Spend: Total amount spent by the customer.

Years_as_Customer: Number of years the customer has been with the company.

Num_of_Purchases: Total number of purchases made by the customer.

Average_Transaction_Amount: Average spend per transaction.

Num_of_Returns: Total number of returns made by the customer.

Num_of_Support_Contacts: Number of times the customer contacted support.

Satisfaction_Score: Customer’s satisfaction score (ranging from 1 to 5).

Last_Purchase_Days_Ago: Number of days since the last purchase.

Email_Opt_In: Whether the customer has opted in for email communication.

Promotion_Response: Customer's response to promotions (Responded, Ignored, Unsubscribed).

Target_Churn: The target variable indicating whether the customer has churned (True/False).

Setup Instructions

To run this project locally or in a cloud environment, follow these steps:

Clone this repository:

git clone <repository-url>
cd <repository-folder>


Install the required dependencies:
The following Python libraries are used in the project:

pandas

numpy

scikit-learn

imbalanced-learn

seaborn

matplotlib

You can install these dependencies by running:

pip install -r requirements.txt


Run the Jupyter Notebook:
After installing the required dependencies, open the Jupyter Notebook:

jupyter notebook Customer_Retention_Prediction_94+_Accuracy_By_TURJOY.ipynb


Dataset:
The dataset used for this project should be in CSV format. Ensure that the data is loaded properly before starting model training.

Model Training and Evaluation:
Follow the instructions in the notebook to train models and evaluate performance. The results will be displayed as confusion matrices, ROC curves, and evaluation metrics.

Key Results

The project resulted in the successful identification of high-risk customers (those likely to churn). The following key results were observed from the models tested:

Best Performing Model:

K-Nearest Neighbors (KNN) achieved the highest accuracy and F1-score among all models tested. The model showed a balanced performance across precision and recall.

Evaluation Metrics:

Accuracy: The accuracy of the models ranged from 80% to 94% based on the dataset.

Precision: Models with high precision, such as K-Nearest Neighbors, minimized false positives (non-churned customers wrongly identified as churned).

Recall: Recall scores were especially critical, as identifying actual churned customers was a priority.

F1-Score: The F1-score provides a balance between precision and recall, and the K-Nearest Neighbors model performed the best here as well.

ROC-AUC: The ROC-AUC scores for all models supported by probability prediction were high, indicating strong discriminative power.

Model Comparison:

A comprehensive model comparison was conducted, showing the performance of Random Forest, XGBoost, Logistic Regression, Gradient Boosting, AdaBoost, and Naive Bayes.

The XGBoost model had high AUC-ROC and showed the best overall results in terms of true positive detection.

Visualizations:

Confusion Matrices were plotted for each model to visualize classification performance.

ROC Curves were plotted for models that support predict_proba() to demonstrate how well each model distinguishes between churned and non-churned customers.

Conclusion

The project successfully developed a machine learning pipeline to predict customer churn based on historical data. By comparing multiple machine learning algorithms and using a variety of evaluation metrics, we were able to select the best-performing model, which can be deployed for real-world applications in customer retention and business strategy optimization.
