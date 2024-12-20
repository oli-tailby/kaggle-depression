import pandas as pd
import numpy as np

def prepare_depression_data(df):
    # This function applies useful data prep and feature creation to the base data provided.
    # This aspect of the project would be best placed inside a data warehouse, perhaps using PySpark or dbt to transform data prior to loading it into pandas

    # convert yes/no columns to numeric
    df['History of Mental Illness'] = df["History of Mental Illness"].replace(('Yes', 'No'), (1, 0))
    df['History of Substance Abuse'] = df["History of Substance Abuse"].replace(('Yes', 'No'), (1, 0))
    df['Family History of Depression'] = df["Family History of Depression"].replace(('Yes', 'No'), (1, 0))
    df['Chronic Medical Conditions'] = df["Chronic Medical Conditions"].replace(('Yes', 'No'), (1, 0))
    # create children flag
    df['Has Children Flag'] = df['Number of Children'].where(df['Number of Children']==0, 1)
    # reduce skew in income variable
    df['Income Square Root'] = np.sqrt(df['Income'])
    # try interaction terms on possible useful variables
    df['Divorced_Children'] = np.where((df['Marital Status']=='Divorced') & df['Has Children Flag']==1, 1, 0)
    df['Widowed_Children'] = np.where((df['Marital Status']=='Widowed') & df['Has Children Flag']==1, 1, 0)
    df['Unhealthy_Lifestyle_All'] = np.where((df['Alcohol Consumption']=='High')&(df['Sleep Patterns']=='Poor')&(df['Dietary Habits']=='Unhealthy')&(df['Physical Activity Level']=='Sedentary'), 1, 0)
    df['Unhealthy_Lifestyle_Sum'] = np.where(df['Alcohol Consumption']=='High', 1, 0) + np.where(df['Sleep Patterns']=='Poor', 1, 0) + np.where(df['Dietary Habits']=='Unhealthy', 1, 0) + np.where(df['Physical Activity Level']=='Sedentary', 1, 0) + np.where(df['Smoking Status']=='Current', 1, 0)
    df['Unemployed_ChronicCondition'] = np.where((df['Employment Status']=='Unemployed')&(df['Chronic Medical Conditions']==1), 1, 0)


    return df