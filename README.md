# BITS-ML-AS-2
BITS MTECH AIML Semester 1 ML assignment 2 (15-Feb-2026)

a. Problem statement
====================
Step 1: Dataset choice
Choose ONE classification dataset of your choice from any public repository -
Kaggle or UCI. It may be a binary classification problem or a multi-class
classification problem.
Minimum Feature Size: 12
Minimum Instance Size: 500
Step 2: Machine Learning Classification models and Evaluation metrics
Implement the following classification models using the dataset chosen above. All
the 6 ML models have to be implemented on the same dataset.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
For each of the models above, calculate the following evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeﬃcient (MCC Score)

b. Dataset description [ 1 mark ]
====================================
Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation. A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.

c. Models used: [ 6 marks - 1 marks for all the metrics for each model ]
========================================================================
Make a Comparison Table with the evaluation metrics calculated for all the 6
models as below:

Model: Logistic Regression
Accuracy: 0.9219
AUC Score: 0.9947
Precision: 0.9225
Recall: 0.9219
F1 Score: 0.9220
MCC: 0.9056

Model: Decision Tree
Accuracy: 0.8920
AUC Score: 0.9436
Precision: 0.8917
Recall: 0.8920
F1 Score: 0.8918
MCC: 0.8694

Model: K-Nearest Neighbors
Accuracy: 0.9175
AUC Score: 0.9846
Precision: 0.9178
Recall: 0.9175
F1 Score: 0.9175
MCC: 0.9002

Model: Gaussian Naive Bayes
Accuracy: 0.7640
AUC Score: 0.9669
Precision: 0.7663
Recall: 0.7640
F1 Score: 0.7615
MCC: 0.7158

Model: Random Forest
Accuracy: 0.9207
AUC Score: 0.9919
Precision: 0.9207
Recall: 0.9207
F1 Score: 0.9206
MCC: 0.9041

Model: XGBoost
Accuracy: 0.9209
AUC Score: 0.9940
Precision: 0.9211
Recall: 0.9209
F1 Score: 0.9209
MCC: 0.9043

ML Model Name          | Accuracy | AUC      | Precision.   |  Recall.    | F1         | MCC.   |
Logistic Regression.   |
Decision Tree
kNN
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)

- Add your observations on the performance of each model on the chosen
dataset. [ 3 marks ]
ML Model Name Observation about model performance
Logistic
Regression
Decision Tree
kNN
Naive Bayes
Random Forest
(Ensemble)
XGBoost
(Ensemble)
Step 6: Deploy on Streamlit Community Cloud
