import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score, 
    roc_curve, 
    auc,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def load_dataframe(path: str):
    return pd.read_csv(path)


def data_split(df: pd.DataFrame, 
               features: list[str], 
               target: str,
               test_size: float
               ):
    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size)


def feat_importance(model):

    feat_imp = model.named_steps['model'].feature_importances_
    feat_names = model.named_steps['preprocessor'].get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feat_names,
        'importance': feat_imp
    }).sort_values(by='importance', ascending=False)

    return importance_df


def plot_importance(importance_df, path='./resources/feature_importance.png'):

    top_importance_df = importance_df.head(8)

    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title('Feature Importance')
    sns.barplot(
        ax=ax,
        x='importance',
        y='feature',
        data=top_importance_df
    )

    ax.set_xlabel('Importance')
    ax.set_ylabel("Feature")

    fig.savefig(path)
    mlflow.log_artifact(path)


def model_scores(y_test, y_pred):

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    pred_positive = y_pred.sum() 

    return precision, recall, accuracy, f1, pred_positive

def roc_auc_plot(y_test, y_prob, path='./resources/roc_plot.png'):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    fig.savefig(path)
    mlflow.log_artifact(path)


def precision_recall_plot(y_test, y_prob, path='./resources/precision_recall_plot.png'):
    display = PrecisionRecallDisplay.from_predictions(y_test, y_prob, plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")

    display.plot().figure_.savefig(path)
    mlflow.log_artifact(path)

def confusion_matrix_plot(y_test, y_pred, path='./resources/confusion_matrix.png'):
    conf_mat = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix = conf_mat, 
        display_labels = [0, 1])
    
    display.plot()
    display.ax_.set_title("Confusion Matrix")
    display.figure_.savefig(path)
    mlflow.log_artifact(path)
