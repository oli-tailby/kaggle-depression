from helper_functions import load_dataframe
from feature_prep import prepare_depression_data
import mlflow
import pandas as pd
import pickle

def import_prep_data(path):

    print("Loading Data")
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


def main(path='./resources'):
    mlflow.start_run()

    # import and prepare training data
    df = import_prep_data(f'{path}/inputs/inference_data.csv')

    # load model object
    with open(f'{path}/outputs/depression-model.pkl', 'rb') as f:
        model = pickle.load(f)

    # run model inference and get predictions at specified threshold
    y_prob, y_pred = model_inference(df, model, threshold=0.4)

    print(y_prob)

    # append predictions to input dataset and save
    y_pred_df = pd.concat([
        df,
        pd.DataFrame({
            'prediction_prob': y_prob,
            'prediction': y_pred
        })
    ], axis=1)

    y_pred_df.to_csv(f'{path}/outputs/output_inference.csv', index=False)
    mlflow.log_artifact(f'{path}/outputs/output_inference.csv')

    mlflow.end_run()


if __name__ == '__main__':
    main()