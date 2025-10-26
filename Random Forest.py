# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
import math
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#0 INPUT DATA
df = pd.read_csv("Dataset_Final_v6.csv", encoding = "utf-8")

# Rename the columns
df.rename(columns={'Cost overruns': 'CO', 'Time overruns': 'TO', 'Content deficiency': 'CD', 'User Involvement': 'UI', 'Technical Competence': 'TC',
                   'Technical Literacy': 'TL','User satisfaction': 'US','Executive Management': 'EM','Statement of Requirements': 'SR','Requirements & Specifications':'RS', 
                   'Resources Availability': 'RA','Project Necessity (Still needed?)': 'PN','IT Management': 'ITM','Changing Requirements & Specifications':'CRS'}, inplace=True)

#df.to_csv("dfname0.csv", index=False)
# Convert all text to lowercase
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
#df.to_csv("dfname0.csv", index=False)
print(df)
print(df.columns)
################################################### encode Restarts,Cost overruns,Time overruns,Content deficiency #####################################################
columns_to_encode = ["Restarts", "CO", "TO", "CD"]
# Create a dictionary to map values
mapping = {"yes": 1, "no": 0, "n/a":"n/a"}
# Apply the mapping to all specified columns
df[columns_to_encode] = df[columns_to_encode].applymap(mapping.get)

################################################### encode User Involvement", "Technical Competence", "Technical Literacy", "User satisfaction #####################################################
# List of columns to encode
columns_to_encode1 = ["UI", "TC", "TL", "US"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "low": 0, "high": 1}
# Apply the mapping to all specified columns
df[columns_to_encode1] = df[columns_to_encode1].applymap(mapping.get)

############################################### encode Executive Management ###############################################################################
# List of columns to encode
columns_to_encode2 = ["EM"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "poor": 0, "high": 1}
# Apply the mapping to all specified columns
df[columns_to_encode2] = df[columns_to_encode2].applymap(mapping.get)

############################################### encode Statement of Requirements ###############################################################################
# List of columns to encode
columns_to_encode3 = ["SR"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "unclear": 0, "clear": 1}
# Apply the mapping to all specified columns
df[columns_to_encode3] = df[columns_to_encode3].applymap(mapping.get)

############################################### encode Planning ###############################################################################
# List of columns to encode
columns_to_encode4 = ["Planning"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "poor": 0, "proper": 1}
# Apply the mapping to all specified columns
df[columns_to_encode4] = df[columns_to_encode4].applymap(mapping.get)

############################################### encode Expectations ###############################################################################
# List of columns to encode
columns_to_encode5 = ["Expectations"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "unrealistic": 0, "realistic": 1}
# Apply the mapping to all specified columns
df[columns_to_encode5] = df[columns_to_encode5].applymap(mapping.get)

############################################### encode Requirements & Specifications  ###############################################################################
# List of columns to encode
columns_to_encode6 = ["RS"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "incomplete": 0, "complete": 1}
# Apply the mapping to all specified columns
df[columns_to_encode6] = df[columns_to_encode6].applymap(mapping.get)

############################################### encode Changing Requirements & Specifications  ###############################################################################
# List of columns to encode
columns_to_encode7 = ["CRS"]
# Define a mapping for the values
mapping= {"frequently changing": 1, "unstable": 2, "stable": 3}
# Apply the mapping to all specified columns
df[columns_to_encode7] = df[columns_to_encode7].applymap(mapping.get)

############################################### encode Resources Availability ###############################################################################
# List of columns to encode
columns_to_encode8 = ["RA"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "inadequate": 0, "adequate": 1}
# Apply the mapping to all specified columns
df[columns_to_encode8] = df[columns_to_encode8].applymap(mapping.get)

############################################### encode Project Necessity (Still needed?)###############################################################################
# List of columns to encode
# columns_to_encode9 = ["PN"]
# # Define a mapping for the values
# mapping= {"n/a": "n/a", False: 0, True: 1}
# # Apply the mapping to all specified columns
# df[columns_to_encode9] = df[columns_to_encode9].applymap(mapping.get)

############################################### encode IT Management###############################################################################
# List of columns to encode
columns_to_encode10 = ["ITM"]
# Define a mapping for the values
mapping= {"n/a": "n/a", "poor": 0, "strong": 1}
# Apply the mapping to all specified columns
df[columns_to_encode10] = df[columns_to_encode10].applymap(mapping.get)
#df.to_csv("newdf.csv")
# Split the dataset based on "State" column
grouped= df.groupby('State')
df1 = grouped.get_group("successful")  # Use .copy() to create a new DataFrame
df2 = grouped.get_group("failed") # Use .copy() to create a new DataFrame

#df1.to_csv("df1.csv")
#df2.to_csv("df2.csv")
# Fill NaN values with the mode for df1
columns_to_fill = ["Restarts", "CO", "TO", "CD", "UI", "TC", "TL", "US", "EM", "SR", "Planning", "Expectations", "RS","RA","ITM", "PN","CRS", 'Duration (Years)']
for column in columns_to_fill:
    if column in df1 and column in df2:
        mode_value_df1 = df1[column].mode()[0]
        mode_value_df2 = df2[column].mode()[0]
        df1[column].fillna(mode_value_df1, inplace=True)
        df2[column].fillna(mode_value_df2, inplace=True)
    else:
        print(f"Column '{column}' not found in either df1 or df2.")

merged_df = pd.concat([df1, df2], ignore_index=True)
columns_to_encode = ["State"]
# Create a dictionary to map values
mapping = {"successful": 1, "failed": 0}
# Apply the mapping to all specified columns
merged_df[columns_to_encode] = merged_df[columns_to_encode].applymap(mapping.get)
# Save the merged DataFrame to a new CSV file
# merged_df.to_csv('encoded_dataset_final6.csv', index=False)


label_encoder = LabelEncoder()
merged_df['Country_encoded'] = label_encoder.fit_transform(merged_df['Country'])

merged_df['Type_encoded'] = label_encoder.fit_transform(merged_df['Type'])

merged_df['Sub-Category_encoded'] = label_encoder.fit_transform(merged_df['Sub-Category'])
# print(merged_df)
# print(merged_df.columns)
f_Data_column_model = ['State','Duration (Years)','Restarts',
       'CO', 'TO', 'CD', 'UI', 'EM', 'SR', 'Planning', 'Expectations',
       'RS', 'CRS', 'TC', 'RA', 'ITM', 'TL', 'US','Country_encoded', 'Type_encoded',
       'Sub-Category_encoded' ]

df_model = merged_df[f_Data_column_model]
# print(df_model)
# missing_values = df_model.isnull().sum()
# print("Missing Values:", missing_values)
model_columns_2 = ['State',
          'UI',  'SR', 'Planning', 'Expectations',
       'RS', 'CRS', 'TC', 'RA', 'ITM', 'TL', 'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model2 = merged_df[model_columns_2]

model_columns_3 = ['State',
          'UI',  'Planning', 'Expectations',
       'RS', 'CRS', 'TC', 'RA', 'ITM', 'TL', 'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model3 = merged_df[model_columns_3]
model_columns_4 = ['State',
          'UI',  'Planning', 'Expectations',
        'CRS', 'TC', 'RA', 'ITM', 'TL', 'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model4 = merged_df[model_columns_4]
model_columns_5 = ['State',
          'UI',  'Planning', 'Expectations',
        'CRS', 'TC', 'RA', 'ITM', 'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model5 = merged_df[model_columns_5]
model_columns_6 = ['State',
          'UI',  'Planning', 'Expectations',
        'CRS', 'TC', 'RA',  'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model6 = merged_df[model_columns_6]
model_columns_7 = ['State',
          'UI',  'Planning', 'Expectations',
        'CRS', 'TC',  'US','Country_encoded', 
       'Sub-Category_encoded' ]
df_model7 = merged_df[model_columns_7]

model_columns_8 = ['State',
          'UI',  'Planning', 'Expectations',
        'CRS', 'TC',  'US', 
       'Sub-Category_encoded' ]
df_model8 = merged_df[model_columns_8]


model_columns_9 = ['State',
          'UI',  'Planning', 
        'CRS', 'TC',  'US', 
       'Sub-Category_encoded' ]
df_model9 = merged_df[model_columns_9]

model_columns_10 = ['State',
          'UI',  'Planning', 
        'CRS', 'TC',  'US', 
        ]
df_model10 = merged_df[model_columns_10]

def train_random_forest(df,n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest model.

    Parameters:
    - df (DataFrame): The data table containing training data.
    - target_column (str): The name of the target variable (target).
    - feature_columns (list): List of column names containing independent variables (features).
    - n_estimators (int): Number of decision trees in the Random Forest (default is 100).
    - max_depth (int): Maximum depth of decision trees (default is None).
    - random_state (int): Seed for random number generation (default is None).

    Returns:
    - model (RandomForestClassifier): The trained Random Forest model.
    """
    # Create X (features) and y (target)
    X = df.iloc[:,1::].values
    y = df.iloc[:,0].values
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)
    cross_val_score(model, X, y, cv = 10)
    model.score(X_test, y_test)

    # Evaluate the model's accuracy on the testing set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the testing set: {accuracy:.2f}")
    
    model.predict_proba(X_test)
    # Cross-Val score for Classification Model
    
    cv_acc = cross_val_score(model, X,y, scoring="accuracy")
    cv_precision = cross_val_score(model, X,y, scoring="precision")
    cv_recall = cross_val_score(model, X,y, scoring="recall")
    cv_f1 = cross_val_score(model, X,y, scoring="f1")
    CrossValScore = pd.DataFrame({
       "The accuracy": [f'{np.mean(cv_acc)*100:.2f}%'],
       "The precision": [f'{np.mean(cv_precision)*100:.2f}%'],
        "The recall": [f'{np.mean(cv_recall)*100:.2f}%'],
        "The f1": [f'{np.mean(cv_f1)*100:.2f}%']
    })
    
    df_predict = pd.DataFrame(data={"actual values": y_test,
                        "predicted values" : y_pred})
    df_predict["differences"] = df_predict["predicted values"] - df_predict["actual values"]
    # Store the performance metrics results in DataFrame
   
    feature_imp = pd.Series(model.feature_importances_,index=df.columns[1:]).sort_values(ascending=False)
    # feature_importances = model.feature_importances_
    print("Cross-Val Score:")
    print(CrossValScore)
    print("\nPredictions:")
    print(df_predict)
    print("Feature Importances:")
    print(feature_imp)
    # for feature, importance in zip(feature_columns, feature_importances):
    #     print(f"{feature}: {importance}")
    # BASELINE MODELS
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# print("Performance model 1")
# RandomForest_Result = train_random_forest(df_model)
# print("Performance model 2")
# RandomForest_Result = train_random_forest(df_model2)
# print("Performance model 3")
# RandomForest_Result = train_random_forest(df_model3)
# print("Performance model 4")
# RandomForest_Result = train_random_forest(df_model4)
# print("Performance model 5")
# RandomForest_Result = train_random_forest(df_model5)
# print("Performance model 6")
# RandomForest_Result = train_random_forest(df_model6)
# print("Performance model 7")
# RandomForest_Result = train_random_forest(df_model7)
# print("Performance model 8")
# RandomForest_Result = train_random_forest(df_model8)
# print("Performance model 9")
# RandomForest_Result = train_random_forest(df_model9)
print("Performance model 10")
RandomForest_Result = train_random_forest(df_model10)