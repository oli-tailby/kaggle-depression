import sys
sys.path.append('./src')
import train_eval as train
import inference as infer

import unittest
from unittest import mock
import pandas as pd
import pathlib as pl
from io import StringIO

class TestTrainFunction(unittest.TestCase):

    def test_model_exists(self):
        with mock.patch('sys.stdout', new = StringIO()):
            train.main(testing_mode=True, path='./test/resources')
            path = pl.Path('./test/resources/outputs/depression-model.pkl')
            self.assertEqual((str(path), path.is_file()), (str(path), True))

class TestInferenceFunction(unittest.TestCase):
    def test_inference_all(self):
        with mock.patch('sys.stdout', new = StringIO()):
            train.main(testing_mode=True, path='./test/resources')
            infer.main(path='./test/resources')
        df_in = pd.read_csv('./test/resources/inputs/inference_data.csv')
        df_out = pd.read_csv('./test/resources/outputs/output_inference.csv')

        self.assertEqual(df_in.shape[0], df_out.shape[0])
        self.assertEqual(df_in['Name'].min(), df_out['Name'].min())
        self.assertEqual(df_in['Name'].max(), df_out['Name'].max())

    def test_inference_vals(self):
        with mock.patch('sys.stdout', new = StringIO()):
            train.main(testing_mode=True, path='./test/resources')
            infer.main(path='./test/resources')
        df_out = pd.read_csv('./test/resources/outputs/output_inference.csv')

        self.assertFalse(df_out['prediction'].isna().any())
        self.assertTrue(df_out['prediction'].isin([0, 1]).all())
        self.assertTrue(df_out['prediction_prob'].max() <= 1)
        self.assertTrue(df_out['prediction_prob'].min() >= 0)


if __name__ == '__main__':
    unittest.main()