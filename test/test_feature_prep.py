import unittest
from src.helper_functions import *
from src.feature_prep import *

class TestPrepFunction(unittest.TestCase):
    def test_rows(self):
        read_df = load_dataframe('./resources/inference_data.csv')
        df = prepare_depression_data(read_df)
        self.assertEqual(df.shape[0], read_df.shape[0])
        self.assertEqual(list(df['Name'].unique()), list(read_df['Name'].unique()))

if __name__ == '__main__':
    unittest.main()