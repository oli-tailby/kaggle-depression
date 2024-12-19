from helper_functions import load_dataframe
from feature_prep import prepare_depression_data
import mlflow
import pandas as pd
import pickle

def import_prep_data(path):

    print("Loading Data")
    path = './resources/inference_data.csv'
    df_raw = load_dataframe(path)
    df = prepare_depression_data(df_raw)
    print(df.head())

    return df

def model_inference(df, model, threshold):

    print('Running inference...')
    threshold = 0.4
    y_prob = model.predict_proba(df)[:,1]
    y_pred = (y_prob >= threshold).astype('int')

    return y_prob, y_pred


def main():
    mlflow.start_run()

    # import and prepare training data
    df = import_prep_data('./resources/inference_data.csv')

    # load model object
    with open('./resources/depression-model.pkl', 'rb') as f:
        model = pickle.load(f)

    # run model inference and get predictions at specified threshold
    y_prob, y_pred = model_inference(df, model, threshold=0.4)

    # append predictions to input dataset and save
    y_pred_df = pd.concat([
        df,
        pd.DataFrame({
            'prediction_prob': y_prob,
            'prediction': y_pred
        })
    ], axis=1)

    y_pred_df.to_csv('./resources/output_inference.csv', ignore_index=True)


if __name__ == '__main__':
    main()