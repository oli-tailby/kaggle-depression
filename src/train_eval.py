from helper_functions import *
from feature_prep import prepare_depression_data
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import mlflow
import pickle

FEATURES = [
    'Age',
    # 'Marital Status',
    'Education Level',
    # 'Number of Children',
    # 'Income',
    # 'Smoking Status', 
    # 'Physical Activity Level',
    'Employment Status',
    'Alcohol Consumption',
    'Dietary Habits',
    'Sleep Patterns',
    # 'History of Substance Abuse',
    # 'Family History of Depression',
    # 'Chronic Medical Conditions',
    # 'Has Children Flag',
    'Income Square Root', 
    # 'Divorced_Children',
    # 'Widowed_Children', 
    # 'Unhealthy_Lifestyle_All',
    'Unhealthy_Lifestyle_Sum', 
    'Unemployed_ChronicCondition'    
]

TARGET = 'History of Mental Illness'

def import_prep_data(path):

    print("Loading Data")
    df_raw = load_dataframe(path)
    df = prepare_depression_data(df_raw)
    print(df.head())

    return df

def preprocessing(cat_cols, num_cols):

    numeric_transform = Pipeline(
        steps = [
            ("impnum", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transform = Pipeline(
        steps = [
            ("impnum", SimpleImputer(strategy="constant", fill_value='MISSING')),
            ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_transform, num_cols),
            ("cat", categorical_transform, cat_cols)
        ]
    )

    return preprocessor


def main(testing_mode=False, path='./resources'):

    mlflow.start_run()

    # import and prep training data
    df = import_prep_data(f'{path}/inputs/depression_data.csv')

    # split data into test and train, feature and target based on the provided variables
    X_train, X_test, y_train, y_test = data_split(df, FEATURES, TARGET, 0.2)

    print(f"Features: {X_train.columns}")
    mlflow.log_metric('Train Samples', X_train.shape[0])
    mlflow.log_metric('Test Samples', X_test.shape[0])
    print(X_train.dtypes)

    # separate categorical and numerical columns for preprocessing
    cat_cols=X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")

    # construct model pipeline
    preprocessor = preprocessing(cat_cols, num_cols)
    lr = LogisticRegression()
    pipe = Pipeline(
        [
            ('preprocessor', preprocessor), 
            ('model', lr)
        ]
    )

    # fit and tune the model using a grid search approach
    params_full = {
        'model__penalty': [None,'l2'], 
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    params_basic = {
        'model__penalty': ['l2'],
        'model__C': [0.01, 1, 100]
    }

    search = GridSearchCV(
        estimator = pipe,
        param_grid = params_basic if testing_mode else params_full,
        scoring = 'roc_auc',
        cv = 2 if testing_mode else 5
    )

    search.fit(X_train, y_train)

    # log training scores and cross-validation results
    mlflow.log_metric('Training Score', search.score(X_train, y_train))
    mlflow.log_metric('Test Score', search.score(X_test, y_test))

    cv_res = pd.DataFrame(search.cv_results_)
    cv_res.to_csv(f'{path}/evaluation/cv_results.csv', index=False)
    mlflow.log_artifact(f'{path}/evaluation/cv_results.csv')

    # save the best performing model
    best_model = search.best_estimator_
    with open(f'{path}/outputs/depression-model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # run inference on the test set for model evaluation
    threshold = 0.3
    y_prob = best_model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype('int')

    imp_df = feat_importance(list(pd.get_dummies(df[FEATURES]).columns), best_model.named_steps['model'])
    imp_df.to_csv(f'{path}/evaluation/feature_importance.csv', index=False)
    plot_importance(imp_df, f'{path}/evaluation/feature_importance.png')

    print(model_scores(y_test, y_pred))
    roc_auc_plot(y_test, y_prob, f'{path}/evaluation/roc_plot.png')
    precision_recall_plot(y_test, y_prob, f'{path}/evaluation/precision_recall_plot.png')
    confusion_matrix_plot(y_test, y_pred, f'{path}/evaluation/confusion_matrix.png')

    mlflow.end_run()

if __name__=='__main__':
    main()