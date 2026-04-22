import pandas as pd

def preparar_dados():
    df = pd.read_csv('Wa_Fn-UseC_-HR-Employee-Attrition.csv')
    df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18'], inplace=True)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=[
        'BusinessTravel', 'Department',
        'EducationField', 'JobRole', 'MaritalStatus'
    ])
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    return X, y
