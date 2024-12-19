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
    df['Income Square Root'] = np.sqrt(df['Income'])

    return df