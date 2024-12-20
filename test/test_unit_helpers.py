import sys
sys.path.append('./src')
import helper_functions as helper

import unittest
from unittest import mock
import pandas as pd

class TestLoadFunction(unittest.TestCase):
    def test_load_rows(self):
        read_df = helper.load_dataframe('./resources/inputs/inference_data.csv')
        self.assertEqual(len(read_df), 2)

    def test_load_cols(self):
        read_df = helper.load_dataframe('./resources/inputs/inference_data.csv')
        self.assertEqual(list(read_df.columns), ['Name','Age','Marital Status','Education Level','Number of Children','Smoking Status','Physical Activity Level','Employment Status','Income','Alcohol Consumption','Dietary Habits','Sleep Patterns','History of Mental Illness','History of Substance Abuse','Family History of Depression','Chronic Medical Conditions'])

class TestDataSplitFunction(unittest.TestCase):
    def test_split_cols(self):
        df = pd.read_csv('./resources/inputs/inference_data.csv')
        X_train, X_test, y_train, y_test = helper.data_split(df, ['Income', 'Age'], 'History of Mental Illness', 0.5)
        self.assertEqual(X_train.shape[0], X_test.shape[0])
        self.assertEqual(y_train.shape[0], y_test.shape[0])
        self.assertEqual(list(X_train.columns), ['Income', 'Age'])
        self.assertEqual(list(X_test.columns), ['Income', 'Age'])

    def test_split_rows(self):
        df = pd.read_csv('./resources/inputs/inference_data.csv')
        X_train, X_test, y_train, y_test = helper.data_split(df, ['Income', 'Age'], 'History of Mental Illness', 0.5)
        self.assertEqual(X_train.shape[0], 0.5*df.shape[0])
        self.assertEqual(X_test.shape[0], 0.5*df.shape[0])
        self.assertEqual(y_train.shape[0], 0.5*df.shape[0])
        self.assertEqual(y_test.shape[0], 0.5*df.shape[0])

    def test_split_target(self):
        df = pd.read_csv('./resources/inputs/inference_data.csv')
        _, _, y_train, y_test = helper.data_split(df, ['Income', 'Age'], 'History of Mental Illness', 0.5)
        self.assertEqual(sum(y_train.index) + sum(y_test.index), sum(df['History of Mental Illness'].index))


if __name__ == '__main__':
    unittest.main()