# Bank Marketing Campaign Analysis and Term Deposit Prediction

## Project Overview

This project focuses on analyzing a bank marketing campaign dataset to identify and predict customers who are most likely to subscribe to a term deposit. By leveraging machine learning techniques, the goal is to help the bank optimize its marketing efforts, reduce costs associated with contacting low-potential customers, and increase the overall term deposit conversion rate.

## Business Problem

**Context:** A European bank aims to increase its term deposit utilization ratio, which was low in previous campaigns. The current campaign seeks to identify potential customers for term deposits to improve cost efficiency and maximize the conversion rate.

**Target:**
*   **0:** Customer will not subscribe to a term deposit.
*   **1:** Customer will subscribe to a term deposit.

**Problem Statement:** How can the bank identify and predict customers with a high probability of opening a term deposit? This will enable the marketing team to improve conversion rates through targeted outreach and optimize operational costs by reducing contact with low-potential customers.

**Goals:**
Based on these issues, banks want the ability to predict whether a customer will make a deposit. This allows marketing teams to focus their efforts on the most potential customers.

Furthermore, banks also want to understand the factors or variables that most influence a customer's decision to make a deposit. With this information, banks can develop more effective marketing campaign plans, such as personalized offers or improved communication strategies.

## Data Source

The dataset used for this analysis is the Bank Marketing Campaigns Dataset, available on Kaggle:
[https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.

## Data Understanding

The dataset contains information on bank customers and their interactions with marketing campaigns, along with socio-economic attributes.

**Key Observations:**
*   The dataset is imbalanced, with a significantly higher number of customers who did not subscribe to a term deposit (`no`) compared to those who did (`yes`).
*   Most features are categorical.
*   Each row represents a single customer and their outcome regarding the term deposit offer.

A detailed description of the attributes can be found in the notebook's Data Understanding section.

## Data Cleaning and Preprocessing

The following steps were performed to clean and prepare the data for modeling:

1.  **Handling Missing Values:**
    *   The value '999' in the `pdays` column, indicating no previous contact, was replaced with NaN and then imputed with -1 to represent this distinct state.
    *   'unknown' values in categorical columns were kept as they represent a significant portion of the data and dropping them could lead to information loss.
2.  **Removing Irrelevant Features:**
    *   The `duration` column was dropped as it is highly correlated with the target variable but is not known before a call is made, making it unsuitable for a predictive model in a real-world scenario.
    *   The `default` column was dropped due to a very small number of 'yes' values, making it highly imbalanced and less informative for prediction.
3.  **Handling Duplicates:** Duplicate rows were identified and removed from the dataset.

## Feature Engineering

Several new features were engineered to capture more meaningful information and potentially improve model performance:

*   **Age Grouping:** Binning the `age` into different age groups (`age_group`, `career_level`).
*   **Marital Grouping:** Grouping marital statuses into `married` and `non_married` (`marital_group`).
*   **Education Grouping:** Grouping less frequent or 'unknown' education levels into a 'low_education' category (`education_group`).
*   **Job Grouping:** Grouping similar job types into broader categories (`job_grouped`).
*   **Housing and Loan Status:** Creating a binary feature indicating if a customer has either housing or personal loan (`has_housing_or_loan`).
*   **Month and Quarter:** Mapping month names to numbers and creating a `quarter` feature (`month_num`, `quarter`).
*   **Total Contacts:** Summing `campaign` and `previous` contacts (`total_contacts`).
*   **Contact Frequency:** Creating a binary feature for high contact frequency (`high_contact_frequency`).
*   **Previous Contact Status:** Binary feature indicating if a customer was contacted before (`was_contacted_before`).
*   **Time Since Last Contact:** Feature representing days since last contact (with -1 for no prior contact) (`time_since_last_contact`).
*   **Recent Contact:** Binary feature indicating recent contact based on a threshold in `pdays` (`recent_contact`).
*   **Previous Outcome Success:** Binary feature indicating if the previous campaign outcome was a success (`previous_outcome_success`, `previous_outcome_success_str`).
*   **Success Rate Previous Contacts:** Ratio of successful previous contacts to total previous contacts (`success_rate_previous_contacts`, `success_rate_previous_contacts_v2`).
*   **Macroeconomic Indices:** Applying PCA to macroeconomic features (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`) to create composite indices (`macro_index_1`, `macro_index_2`).
*   **Interaction Feature:** Creating an interaction feature between age group and previous outcome success (`age_group_success_interaction`).
*   **Encoded Target:** Encoding the target variable 'y' into numerical form (0 for 'no', 1 for 'yes') (`y_encoded`).

Three different feature sets were created to experiment with different combinations of original and engineered features.

## Exploratory Data Analysis (EDA)

The EDA involved visualizing the distribution of features, identifying outliers, and analyzing the correlation between features.

**Key Findings from EDA:**
*   Significant class imbalance in the target variable.
*   Distributions of numerical features like `age`, `campaign`, and `pdays` showed skewness and outliers.
*   Correlation analysis revealed strong positive correlations between macroeconomic factors (`emp.var.rate`, `euribor3m`, `nr.employed`).
*   Cramer's V indicated relationships between categorical features, such as `housing` and `loan`, and `contact` and `month`.
*   Analysis of categorical feature proportions against the target variable highlighted categories with higher success rates (e.g., 'success' in `poutcome`, 'mar', 'dec', 'sep', 'oct' in `month`, 'student' and 'retired' in `job`).

## Methodology

The project followed a standard machine learning methodology:

1.  **Data Splitting:** The dataset was split into training, testing, and potentially validation sets (70% train, 30% test).
2.  **Preprocessing Pipeline:** A `ColumnTransformer` was used to apply different preprocessing steps to different types of features:
    *   Numeric features: Imputation (mean) and Scaling (RobustScaler).
    *   Ordinal features: Imputation (most frequent) and Ordinal Encoding.
    *   Categorical features (more than 2 unique values): Imputation (most frequent) and One-Hot Encoding.
    *   Categorical features (2 unique values): Imputation (most frequent) and Binary Encoding.
3.  **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data within the modeling pipeline to address the class imbalance issue by over-sampling the minority class (`yes`).
4.  **Model Selection:** Several classification models were evaluated using cross-validation on the training data, focusing on metrics relevant to the business problem (Precision, Recall, F1-Score, F2-Score, Accuracy):
    *   Decision Tree
    *   Random Forest
    *   Extra Trees
    *   XGBoost
    *   LightGBM
    XGBoost was selected as the primary model due to its performance, particularly its ability to achieve a good balance of Recall.
5.  **Hyperparameter Tuning:** `RandomizedSearchCV` and `GridSearchCV` were used to tune the hyperparameters of the XGBoost model to optimize performance based on the chosen evaluation metrics (primarily F1-Score, with consideration for Recall).

## Model Evaluation

The selected and tuned XGBoost model was evaluated on the unseen test set using various metrics and visualizations:

*   **Classification Report:** Provided Precision, Recall, F1-Score, and Accuracy for both classes ('no' and 'yes').
*   **Confusion Matrix:** Visualized the counts of True Positives, True Negatives, False Positives, and False Negatives.
*   **ROC Curve:** Plotted the True Positive Rate against the False Positive Rate at various threshold settings, with the Area Under the Curve (AUC) as a summary metric.
*   **Precision-Recall Curve:** Plotted Precision against Recall at various threshold settings, particularly useful for imbalanced datasets.

The evaluation showed that the model achieved a reasonable Recall, indicating its ability to identify a significant portion of potential term deposit subscribers. The Precision-Recall curve highlighted the trade-off between these two metrics.

## Model Interpretation

To understand how the model makes predictions, feature importance and SHAP (SHapley Additive exPlanations) values were analyzed:

*   **Feature Importance (F-Score):** Showed which features were most frequently used in the XGBoost trees.
*   **SHAP Summary Plot:** Provided a global view of feature importance and how the values of each feature impact the model's output (higher or lower probability of subscribing to a term deposit).

**Key Interpretations from SHAP:**
*   Features like `euribor3m`, `total_contacts`, `age_group`, and the macroeconomic indices (`macro_index_1`, `macro_index_2`) were among the most influential.
*   Lower `euribor3m` values and higher `total_contacts` generally increased the probability of a term deposit.
*   Older age groups tended to have a higher probability of subscribing.
*   Contacting customers in May (`one_hot__month_may`) seemed to have a negative impact on the probability of subscription, aligning with EDA findings.

These interpretations provide valuable insights into customer behavior and help validate the model's predictions.

## Conclusion and Recommendation

**Conclusion:**

The developed XGBoost model demonstrates the potential to effectively identify customers likely to subscribe to a term deposit. The model's ability to capture a significant portion of potential subscribers (indicated by Recall) is valuable for optimizing marketing campaigns. Key factors influencing subscription were identified through feature importance and SHAP analysis, including economic indicators and contact history. A preliminary cost analysis suggests that using the model for targeting can lead to significant cost savings compared to mass marketing, without losing a substantial number of potential customers.

**Recommendations:**

## Recommendations

Based on the data analysis and machine learning models created, here are some recommendations banks can implement to improve their models and future marketing campaigns:

1. **Improve Data Quality**: Some categorical features have a significant number of 'unknown' features. If possible, banks can strive to collect more complete and accurate data to reduce these 'unknown' features. Furthermore, historical campaign data with more detailed information (e.g., specific offer types, initial customer responses) can be very helpful.

2. **Implement a Probability-Based Targeting Strategy**: Use the best model to identify potential customers with the highest probability of making a deposit. Focus the marketing team's efforts on this segment.

3. **Monitor and Evaluate the Model Regularly**: Market conditions and customer behavior can change. It is important to regularly monitor model performance with the latest data and retrain the model if necessary.

4. **A/B Test Campaigns**: Conduct A/B testing by targeting customer groups based on model predictions and comparing them to a control group. This will provide empirical evidence of the effectiveness of model-based targeting.

5. **External Factor Analysis**: In addition to customer data, consider incorporating relevant external data such as regional economic trends, central bank interest rate policies, or competitor activity.
By implementing these steps, banks can continuously improve their forecasting capabilities, optimize marketing resource allocation, and ultimately increase deposit conversion rates more effectively.

By implementing these recommendations, the bank can leverage the predictive power of the model to make more informed marketing decisions, leading to increased efficiency and a higher term deposit conversion rate.

## How to Run the Code

1.  Clone the repository to your local machine or open it in Google Colab.
2.  Ensure you have the necessary libraries installed. You can install them using pip:

## Link Tabeleau : https://public.tableau.com/app/profile/rendi.dharmawan/viz/BankMarketingCampaign_17552330852340/MachineLearningDashboard?publish=yes

## Screen shoot tabelau

![Image Alt](https://github.com/PurwadhikaDev/DeltaGroup_JC_DS_AH_BDG_0108_FinalProject/blob/a4a2f4bbf82f4b03649a8d9f2efafa50cb1505f5/21.JPG)

![Image Alt](https://github.com/PurwadhikaDev/DeltaGroup_JC_DS_AH_BDG_0108_FinalProject/blob/a4a2f4bbf82f4b03649a8d9f2efafa50cb1505f5/22.JPG)

![Image Alt](https://github.com/PurwadhikaDev/DeltaGroup_JC_DS_AH_BDG_0108_FinalProject/blob/a4a2f4bbf82f4b03649a8d9f2efafa50cb1505f5/23.JPG)
