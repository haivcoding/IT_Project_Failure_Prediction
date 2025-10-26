import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#0 INPUT DATA
df = pd.read_csv("Dataset_Final_v6.csv", encoding = "utf-8")

# Rename the columns
df.rename(columns={'Cost overruns': 'CO', 'Time overruns': 'TO', 'Content deficiency': 'CD', 'User Involvement': 'UI', 'Technical Competence': 'TC',
                   'Technical Literacy': 'TL','User satisfaction': 'US','Executive Management': 'EM','Statement of Requirements': 'SR','Requirements & Specifications':'RS', 
                   'Resources Availability': 'RA','Project Necessity (Still needed?)': 'PN','IT Management': 'ITM','Changing Requirements & Specifications':'CRS'}, inplace=True)
label_encoder_country = LabelEncoder()
df['Country_encoded'] = label_encoder_country.fit_transform(df['Country'])

label_encoder_type = LabelEncoder()
df['Type_encoded'] = label_encoder_type.fit_transform(df['Type'])

label_encoder_sub = LabelEncoder()
df['Sub-Category_encoded'] = label_encoder_sub.fit_transform(df['Sub-Category'])
#df.to_csv("dfname0.csv", index=False)
# Convert all text to lowercase
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
#df.to_csv("dfname0.csv", index=False)

################################################### encode Restarts,Cost overruns,Time overruns,Content deficiency #####################################################
columns_to_encode = ["Restarts", "CO", "TO", "CD"]
# Create a dictionary to map values
mapping = {"yes": 2, "no": 1, "n/a":0}
# Apply the mapping to all specified columns
df[columns_to_encode] = df[columns_to_encode].applymap(mapping.get)

################################################### encode User Involvement", "Technical Competence", "Technical Literacy", "User satisfaction #####################################################

# List of columns to encode
columns_to_encode1 = ["UI", "TC", "TL", "US"]
# Define a mapping for the values
mapping= {"n/a": 0, "low": 1, "high": 2}
# Apply the mapping to all specified columns
df[columns_to_encode1] = df[columns_to_encode1].applymap(mapping.get)

############################################### encode Executive Management ###############################################################################
# List of columns to encode
columns_to_encode2 = ["EM"]
# Define a mapping for the values
mapping= {"n/a": 0, "poor": 1, "high": 2}
# Apply the mapping to all specified columns
df[columns_to_encode2] = df[columns_to_encode2].applymap(mapping.get)

############################################### encode Statement of Requirements ###############################################################################
# List of columns to encode
columns_to_encode3 = ["SR"]
# Define a mapping for the values
mapping= {"n/a": 0, "unclear": 1, "clear": 2}
# Apply the mapping to all specified columns
df[columns_to_encode3] = df[columns_to_encode3].applymap(mapping.get)

############################################### encode Planning ###############################################################################
# List of columns to encode
columns_to_encode4 = ["Planning"]
# Define a mapping for the values
mapping= {"n/a": 0, "poor": 1, "proper": 2}
# Apply the mapping to all specified columns
df[columns_to_encode4] = df[columns_to_encode4].applymap(mapping.get)

############################################### encode Expectations ###############################################################################
# List of columns to encode
columns_to_encode5 = ["Expectations"]
# Define a mapping for the values
mapping= {"n/a": 0, "unrealistic": 1, "realistic": 2}
# Apply the mapping to all specified columns
df[columns_to_encode5] = df[columns_to_encode5].applymap(mapping.get)

############################################### encode Requirements & Specifications  ###############################################################################
# List of columns to encode
columns_to_encode6 = ["RS"]
# Define a mapping for the values
mapping= {"n/a": 0, "incomplete": 1, "complete": 2}
# Apply the mapping to all specified columns
df[columns_to_encode6] = df[columns_to_encode6].applymap(mapping.get)

############################################### encode Changing Requirements & Specifications  ###############################################################################
label_encoder_CRS = LabelEncoder()
df['CRS'] = label_encoder_CRS.fit_transform(df['CRS'])

############################################### encode Resources Availability ###############################################################################
# List of columns to encode
columns_to_encode8 = ["RA"]
# Define a mapping for the values
mapping= {"n/a": 0, "inadequate": 1, "adequate": 2}
# Apply the mapping to all specified columns
df[columns_to_encode8] = df[columns_to_encode8].applymap(mapping.get)

############################################### encode Project Necessity (Still needed?)###############################################################################
# List of columns to encode
columns_to_encode9 = ["PN"]
# Define a mapping for the values
mapping= {"n/a": 0, False: 1, True: 2}
# Apply the mapping to all specified columns
df[columns_to_encode9] = df[columns_to_encode9].applymap(mapping.get)

############################################### encode IT Management###############################################################################
# List of columns to encode
columns_to_encode10 = ["ITM"]
# Define a mapping for the values
mapping= {"n/a": 0, "poor": 1, "strong": 2}
# Apply the mapping to all specified columns
df[columns_to_encode10] = df[columns_to_encode10].applymap(mapping.get)
#df.to_csv("newdf.csv")
# Split the dataset based on "State" column
grouped= df.groupby('State')
df1 = grouped.get_group('successful')  # Use .copy() to create a new DataFrame
df2 = grouped.get_group('failed') # Use .copy() to create a new DataFrame

#df1.to_csv("df1.csv")
#df2.to_csv("df2.csv")
# Fill NaN values with the mode for df1
columns_to_fill = ["Restarts", "CO", "TO", "CD", "UI", "TC", "TL", "US", "EM", "SR", "Planning", "Expectations", "RS","RA","ITM", "PN"]
for column in columns_to_fill:
    if column in df1 and column in df2:
        mode_value_df1 = df1[column].mode()[0]
        mode_value_df2 = df2[column].mode()[0]
        df1[column].fillna(mode_value_df1, inplace=True)
        df2[column].fillna(mode_value_df2, inplace=True)
    else:
        print(f"Column '{column}' not found in either df1 or df2.")

merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('encoded_dataset.csv', index=False)
