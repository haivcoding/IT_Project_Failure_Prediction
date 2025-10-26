# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#0 INPUT DATAs
df = pd.read_csv("encoded_dataset.csv")
# Define feature columns and the target variable
feature_columns = ["Restarts", "CO", "TO", "CD", "UI", "TC", "TL", "US", "EM", "SR", "Planning", "Expectations", "RS", "RA", "ITM", "Country_encoded","Type_encoded","Sub-Category_encoded"]
X = df[feature_columns].values
y = df['State'].values

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create an SVM classifier with a linear kernel
svm = SVC(kernel='linear')
svm.fit(X, y)

# Get the coefficients (weights) assigned to each feature
feature_weights = svm.coef_[0]

# Calculate the absolute values of feature weights for ranking
abs_feature_weights = np.abs(feature_weights)

# Sort the features by their absolute weights in descending order
feature_ranking = np.argsort(-abs_feature_weights)

# Display the ranked features along with their scores (absolute weights)
ranked_features = [feature_columns[i] for i in feature_ranking]
scores = abs_feature_weights[feature_ranking]
print("Ranked features and their scores:")
for rank, (feature, score) in enumerate(zip(ranked_features, scores), start=1):
    print(f"{rank}. {feature}: Score = {score:.4f}")
"""#Ranked score: Ranked features and their scores:
1. UI: Score = 1.2819
2. Restarts: Score = 1.1129
3. RA: Score = 0.7336
4. TC: Score = 0.7192
5. TL: Score = 0.7014
6. US: Score = 0.6860
7. RS: Score = 0.6000
8. CD: Score = 0.5864
9. Type_encoded: Score = 0.5400
10. CO: Score = 0.4582
11. Planning: Score = 0.2512
12. Country_encoded: Score = 0.2348
13. EM: Score = 0.2183
14. Expectations: Score = 0.2163
15. SR: Score = 0.1652
16. Sub-Category_encoded: Score = 0.0396
17. ITM: Score = 0.0157
18. TO: Score = 0.0156
"""
#Using Backward selection technique: eliminating feature [18] to [5] will not affect the model accuracy score
"""
Accuracy: 0.9411764705882353
              precision    recall  f1-score   support

      failed       1.00      0.86      0.92        14
  successful       0.91      1.00      0.95        20

    accuracy                           0.94        34
   macro avg       0.95      0.93      0.94        34
weighted avg       0.95      0.94      0.94        34 
"""
#Eliminating the next features from [4] to [1] will decrease the model accuracy score
"""
Accuracy: 0.8529411764705882
              precision    recall  f1-score   support

      failed       0.76      0.93      0.84        14
  successful       0.94      0.80      0.86        20

    accuracy                           0.85        34
   macro avg       0.85      0.86      0.85        34
weighted avg       0.87      0.85      0.85        34
"""
# Therefore, the code below shows the best features that given the highest model accuracy score
feature_df= df[["UI", "Restarts","RA","TC"]]
X= np.asarray(feature_df)
y= np.asanyarray(df['State'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the SVM model on the test set
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)

from sklearn.metrics import classification_report

# Print the classification report
print(classification_report(y_test, y_pred))
#Result:
"""
Accuracy: 0.9411764705882353
              precision    recall  f1-score   support

      failed       1.00      0.86      0.92        14
  successful       0.91      1.00      0.95        20

    accuracy                           0.94        34
   macro avg       0.95      0.93      0.94        34
weighted avg       0.95      0.94      0.94        34
"""





