
# 🏥Detection of various types of thyroid-related disease using Machine Learning

Thyroid disorders encompass various medical conditions where the thyroid gland fails to produce hormones adequately. These conditions fall into categories like Hyperthyroidism, Hypothyroidism, Primary Hyperthyroidism, Primary Hypothyroidism, Pituitary Gland Abnormalities, and Early Hyperthyroidism, with Hypothyroidism being the most prevalent. Detecting thyroid diseases typically involves complex blood tests, which can be challenging to interpret due to the large volume of data involved. An alternative approach is to predict thyroid diseases by assessing the levels of T3, T4, and TSH in the body. To achieve this, the study compares seven machine learning algorithms, including Logistic Regression, Support Vector Classifier (SVC), Random Forest Classifier, KNeighborsClassifier, Gaussian Naive Bayes (GaussianNB), XGBoost, and Artificial Neural Networks (ANN). Additionally, techniques like Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are utilized to enhance prediction accuracy.


## Dataset

We gathered thyroid reports from local pathology labs near Panvel, Maharashtra, creating a dataset containing 5743 entries with metadata. The dataset included information on the patient's age, sex, report date, reference range, parameter name, parameter values, abnormality status (classified as N for Normal, A for Abnormal, and L for Lower), and UOM Code (representing standard international units). Notably:

- **Age:** Denotes the age of the patient during the test.
- **Sex:** Specifies the patient's gender.
- **ReportDate:** Indicates the date of report generation, primarily used for analytical purposes.
- **Parameter Name and Values:** Encompassed parameters vital for disease detection, specifically T3 (Triiodothyronine), T4 (Thyroxine), and TSH (Thyroid Stimulating Hormone).
- **UOM Code:** Represents the standard international unit for each parameter.

This dataset provides essential information for the assessment and diagnosis of thyroid diseases.



## Data Preprocessing

**Thyroid Disease Classification Methodology**

In the pursuit of identifying thyroid diseases, we implemented the following methodology:

1. **Creation of the "Results" Column:** A new column, "Results," was introduced to classify thyroid diseases into eight distinct categories based on the parameters T3, T4, and TSH. These categories include Hyperthyroidism, Hypothyroidism, Primary Hyperthyroidism, Primary Hypothyroidism, Pituitary Gland Abnormal, Early Hyperthyroidism, Ok (indicating normal results), and More Diagnosis.

2. **Data Encoding:** The dataset involved two categorical input features, "Sex" and "Results." To facilitate data analysis, these features were encoded into numeric values using the LabelEncoder module from the sklearn library.

3. **Parameter Value Transformation:** Parameter values (T3, T4, TSH), initially represented as strings, were transformed into numeric values using pandas' to_numeric function.

4. **Oversampling:** The dataset exhibited an imbalance, with a significant portion of data falling into the "OK" class, indicating normal hormone values. To address this skewness and enhance the performance of classification models, oversampling was performed. We utilized the SMOTE (Synthetic Minority Oversampling Technique) algorithm to generate new data, ensuring an equal representation of each class.

5. **Feature Extraction:** We conducted feature extraction to determine the effect of input variables on data classification. Notably, no feature extraction was executed to retain all relevant input variables. To comprehend the relationship between features and output values, and to eliminate insignificant input variables, we applied the PCA (Principal Component Analysis) modules provided by Sklearn.

These steps collectively formed the foundation of our approach to classify thyroid diseases and enhance the effectiveness of machine learning algorithms.




## Training Phase

In the training phase, the preprocessed data were divided into training data (75%) and testing data (25%). We employed various classification models to train the data, and each model's performance was evaluated using different metrics. Here's an overview of the models and metrics used:

**Classification Models:**
1. **Logistic Regression:** Logistic regression predicts the output of categorical dependent variables, making it suitable for categorical or discrete results.
2. **Support Vector Classifier (SVC):** SVC aims to find a hyperplane in an N-Dimensional space to classify data points distinctly.
3. **Random Forest Classifier:** This model uses decision trees generated from a randomly selected subset of training data and aggregates predictions from different decision trees.
4. **KNeighborsClassifier:** It classifies new data based on similarity with previously stored data.
5. **GaussianNB:** Gaussian Naive Bayes is a specialized type of Naive Bayes algorithm.
6. **XGboost:** Extreme Gradient Boosting implements gradient boosted decision trees and is effective for skewed datasets.
7. **Artificial Neural Networks (ANN):** ANNs simulate the behavior of biological neurons and are part of computing algorithms.

--

**Metrics Calculated:**
- **Accuracy:** Measures the proportion of correct predictions to all predictions, indicating the classification model's overall performance.
- **Classification Report:** Provides comprehensive details on various metrics, including Precision, Recall, F1 score, and support for specific models.
- **Matthews Correlation Coefficient (MCC):** A value between -1 and +1 that represents the correlation between predictions and actual values, with +1 indicating perfect prediction.
- **Confusion Matrix:** A matrix used to analyze the results of a classification model, comparing projected goal values to actual goal values.

These metrics were essential in evaluating the performance of each model during the training phase.


## Conclusion

In this study, thyroid data from a local pathology lab were analyzed using various classification algorithms to identify different thyroid defect classes based on thyroid hormone parameters (T3, T4, TSH). Notably, XG Boost, Random Forest Classifier, and ANN demonstrated superior performance compared to other models. This is primarily attributed to the robustness of these algorithms against data skewness. Furthermore, the use of oversampling techniques significantly improved accuracy.

