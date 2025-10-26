import pandas as pd
import re

# Read your CSV file into a DataFrame
df = pd.read_csv('PACreport.csv', encoding='ISO-8859-1')

# Define a function to extract and compare budget and cost information
def extract_and_compare_budget_and_cost(text):
    # Find all numeric values with optional commas following the "$" sign in the text
    monetary_values = re.findall(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
    
    initial_budget = final_cost = None

    if monetary_values:
        for value in monetary_values:
            # Remove commas and convert to float
            value_numeric = float(value.replace(',', ''))

            # Check if it mentions million, thousand, or billion and convert accordingly
            if 'million' in text.lower():
                value_numeric *= 1e6
            elif 'thousand' in text.lower():
                value_numeric *= 1e3  # Convert to million
            elif 'billion' in text.lower():
                value_numeric *= 1e9

            # If initial budget is not set, assign the first value as initial budget
            if initial_budget is None:
                initial_budget = value_numeric
            elif value_numeric > initial_budget:
                final_cost = value_numeric
            else:
                initial_budget = value_numeric

    return initial_budget, final_cost

# Apply the function to the 'Text' column of your DataFrame
df['Initial Budget'], df['Final Cost'] = zip(*df['Text'].apply(extract_and_compare_budget_and_cost))

# Convert values to million
df['Initial Budget'] /= 1e6
df['Final Cost'] /= 1e6

# Add a new column for Over_Run
df['Over_Run'] = df['Final Cost'] - df['Initial Budget']

# Print the updated DataFrame
print(df[['Text', 'Initial Budget', 'Final Cost', 'Over_Run']])

# Define a function to extract project names
def extract_project_name(text):
    pattern = r'[A-Z][A-Za-z\s&]+\s*\(.*?\)\s*project'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]
    else:
        return None

# Apply the function to the 'Text' column and create a new 'Project Name' column
df['Project Name'] = df['Text'].apply(extract_project_name)

# Display the resulting DataFrame
print(df)
