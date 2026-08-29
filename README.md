# Customer Churn Prediction Using Machine Learning

An end-to-end machine-learning project that predicts whether telecom customers are likely to churn and identifies the customer, account, contract, service, and billing characteristics associated with churn risk.

## Business Problem

Customer churn occurs when customers stop using a company’s services. For telecom companies, churn reduces recurring revenue and increases the cost of acquiring replacement customers.

The goal of this project is to identify customers at high risk of leaving so the company can take proactive retention actions, such as personalized offers, contract incentives, technical-support outreach, and service bundles.

## Project Objective

This project uses exploratory data analysis and machine learning to:

- Analyze customer churn patterns across contracts, tenure, payment methods, service subscriptions, billing, and demographics.
- Predict whether a customer is likely to churn.
- Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
- Handle class imbalance using SMOTE within the model-training pipeline.
- Tune Random Forest and XGBoost using 5-fold Stratified GridSearchCV.
- Select the best-performing model using ROC-AUC, recall, F1-score, and average precision.
- Translate results into data-driven business recommendations.

## Dataset Overview

- **Dataset:** Telco Customer Churn Dataset
- **Source:** Kaggle / IBM Sample Dataset
- **Rows:** 7,043 customers
- **Columns:** 21
- **Target variable:** `Churn`
- **Problem type:** Supervised binary classification
- **Model features:** 19 features after removing `customerID`

Each row represents one customer. The dataset includes demographics, service subscriptions, account information, contract type, payment method, monthly charges, total charges, and churn status.

### Data Preparation

The project included the following data-preparation steps:

- Removed `customerID` from machine-learning features because it is only an identifier.
- Converted `TotalCharges` from text/object format to numeric format.
- Identified 11 blank `TotalCharges` values and handled them through the preprocessing workflow.
- Converted `Churn` from `Yes` / `No` into binary values: `1` / `0`.
- Applied One-Hot Encoding to categorical variables.
- Applied Iterative Imputation and Standard Scaling to numeric variables.
- Used an 80/20 stratified train-test split.
- Applied SMOTE only inside the training pipeline to prevent data leakage.

## Overall Churn Rate

| Customer Status | Customer Count | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

The churn rate of 26.54% indicates class imbalance. Therefore, this project used stratified splitting, SMOTE in the training pipeline, ROC-AUC, recall, F1-score, average precision, confusion matrices, and precision-recall evaluation.

## Exploratory Data Analysis

### 1. Contract Type

Month-to-month customers had the highest churn rate at 42.71%, compared with 11.27% for one-year customers and only 2.83% for two-year customers.

![Customer churn rate by contract type](images/contract_churn_rate.jpg)

**Business insight:** Customers with short-term contracts are the most important retention segment. The company can test contract-upgrade incentives, loyalty rewards, bundled services, and price-protection offers.

### 2. Customer Tenure

Customers in their first 12 months had the highest churn rate at 47.44%. Churn steadily decreased as customer tenure increased.

| Tenure Group | Churn Rate |
|---|---:|
| 0–12 months | 47.44% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49–72 months | 9.51% |

![Customer churn rate by tenure group](images/tenure_churn_rate.jpg)

**Business insight:** Customer retention is especially important during the first year. Onboarding support, early service check-ins, plan reviews, and personalized offers may help reduce early churn.

### 3. Service Add-Ons

Customers without online security and technical support had much higher churn rates than customers who subscribed to those services.

| Service Add-On | Churn Without Service | Churn With Service | Difference |
|---|---:|---:|---:|
| Online Security | 41.77% | 14.61% | 27.16 percentage points |
| Tech Support | 41.64% | 15.17% | 26.47 percentage points |
| Online Backup | 39.93% | 21.53% | 18.40 percentage points |
| Device Protection | 39.13% | 22.50% | 16.63 percentage points |

![Customer churn rate by service add-on subscription](images/service_addon_churn.jpg)

**Business insight:** Security and technical-support services are strongly associated with lower churn. The company can test free trials, retention bundles, discounted support plans, and targeted add-on offers for high-risk customers.

### 4. Additional Findings

- Customers using **Electronic Check** had the highest payment-method churn rate at **45.29%**.
- **Fiber optic** customers had the highest internet-service churn rate at **41.89%**.
- Churned customers paid higher monthly charges on average: **$74.44**, compared with **$61.27** for retained customers.
- Churned customers had lower average tenure: **17.98 months**, compared with **37.57 months** for retained customers.
- Senior citizens had a churn rate of **41.68%**, compared with **23.61%** for non-senior customers.
- Customers with paperless billing had a churn rate of **33.57%**, compared with **16.33%** for customers without paperless billing.
- The strongest numeric correlation with churn was tenure at **-0.352**, indicating that longer customer relationships are associated with lower churn.

## Machine Learning Workflow

1. Defined 19 predictive features and the binary churn target.
2. Performed an 80/20 stratified train-test split.
3. Preprocessed numeric and categorical variables using `ColumnTransformer`.
4. Used `IterativeImputer` and `StandardScaler` for numeric features.
5. Used `SimpleImputer` and `OneHotEncoder` for categorical features.
6. Applied SMOTE only within model pipelines.
7. Compared Logistic Regression, Decision Tree, Random Forest, and XGBoost.
8. Tuned Random Forest and XGBoost using `GridSearchCV` with 5-fold `StratifiedKFold`.
9. Selected the final model based on test-set performance.

## Baseline Model Comparison

![Baseline model performance comparison](images/model_comparison.jpg)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.737 | 0.503 | 0.794 | 0.616 | 0.840 | 0.628 |
| Random Forest | 0.768 | 0.566 | 0.540 | 0.553 | 0.819 | 0.591 |
| XGBoost | 0.782 | 0.591 | 0.583 | 0.587 | 0.818 | 0.610 |
| Decision Tree | 0.717 | 0.470 | 0.532 | 0.499 | 0.658 | 0.375 |

Logistic Regression was the strongest baseline model based on ROC-AUC and recall. It identified 79.4% of actual churners but had lower precision, meaning it also produced more false positive churn alerts.

## Hyperparameter Tuning

The project used 5-fold Stratified GridSearchCV with ROC-AUC as the optimization metric.

| Tuned Model | Best Cross-Validation ROC-AUC |
|---|---:|
| Random Forest | 0.8419 |
| XGBoost | 0.8471 |

## Final Model: Tuned XGBoost

Tuned XGBoost was selected as the final model because it achieved the strongest overall test-set results, including the highest ROC-AUC, F1-score, and average precision among all evaluated models.

| Metric | Final Score |
|---|---:|
| Accuracy | 0.775 |
| Precision | 0.559 |
| Recall | 0.727 |
| F1 Score | 0.632 |
| ROC-AUC | 0.843 |
| Average Precision | 0.648 |

The final model correctly identified approximately 72.7% of actual churners in the held-out test set.

## Business Recommendations

1. **Prioritize month-to-month customers:** Their churn rate was 42.71%. Test incentives that encourage migration to one-year or two-year contracts.

2. **Focus on first-year retention:** Customers with 0–12 months tenure churned at 47.44%. Implement onboarding campaigns, service-quality follow-ups, and early loyalty offers.

3. **Improve the Electronic Check experience:** This payment group had a 45.29% churn rate. Encourage automatic payment adoption through discounts, reminders, simplified billing, or autopay incentives.

4. **Investigate the fiber-optic customer experience:** Fiber-optic customers had a churn rate of 41.89%. Review pricing, reliability, service quality, installation, and support interactions.

5. **Bundle online security and technical support:** Customers who lacked these services had churn rates above 41%, while subscribing customers had churn rates near 15%.

6. **Use model-driven prioritization:** Score active customers with the Tuned XGBoost model and send the highest-risk group into targeted retention campaigns.

## Tools and Technologies

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Joblib
- GitHub

## Repository Structure

```text
customer-churn-prediction/
│
├── README.md
├── Customer_Churn_Prediction.ipynb
├── requirements.txt
├── .gitignore
│
├── images/
│   ├── contract_churn_rate.jpg
│   ├── tenure_churn_rate.jpg
│   ├── service_addon_churn.jpg
│   └── model_comparison.jpg
│
└── documentation/
    └── Telco_Churn_Colab_Output.pdf
```

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/vivek0717/customer-churn-prediction.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Open and run:

```text
Customer_Churn_Prediction.ipynb
```

4. Download the Telco Customer Churn dataset from Kaggle and update the dataset path in the notebook if needed.

## Limitations and Future Improvements

- The data identifies associations with churn but does not prove causation.
- The dataset does not contain customer satisfaction scores, complaint history, competitor offers, detailed cancellation reasons, or call-center interactions.
- Future improvements could include threshold tuning based on campaign cost, customer lifetime value, retention-offer experiments, fairness testing, and a deployed dashboard for real-time churn-risk monitoring.

## Author

**Vivek Parmar**

Graduate-level project in business analytics, machine learning, and customer-retention strategy.
