import unittest
from src.helper_functions import *

class TestLoadFunction(unittest.TestCase):
    def test_load_rows(self):
        read_df = load_dataframe('./resources/inference_data.csv')
        self.assertEqual(len(read_df), 2)

    def test_load_cols(self):
        read_df = load_dataframe('./resources/inference_data.csv')
        self.assertEqual(list(read_df.columns), ['Name','Age','Marital Status','Education Level','Number of Children','Smoking Status','Physical Activity Level','Employment Status','Income','Alcohol Consumption','Dietary Habits','Sleep Patterns','History of Mental Illness','History of Substance Abuse','Family History of Depression','Chronic Medical Conditions'])



if __name__ == '__main__':
    unittest.main()