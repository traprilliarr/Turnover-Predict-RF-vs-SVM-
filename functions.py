
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random_forest as rf
import svm
from streamlit.runtime.uploaded_file_manager import UploadedFile
import numpy as np


def preprocess_data(df: pd.DataFrame, label_encoders, std_scalers):
    df = df.drop_duplicates()

    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical
            df[column] = df[column].fillna(df[column].mode()[0])
        else:  # Numerical
            df[column] = df[column].fillna(df[column].median())

    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoders[column].transform(df[column])

    return df


def predict_for_rf(model: rf.RandomForestClassifier, X, y, n_trees: int, file: UploadedFile):
    rf.RandomForestClassifier.re_fit(model, n_trees, X, y)

    y_pred = model.predict(X)

    scores = [
        accuracy_score(y, y_pred),
        precision_score(y, y_pred, pos_label=0, average="binary"),
        recall_score(y, y_pred, pos_label=0, average="binary"),
        f1_score(y, y_pred, pos_label=0, average="binary")
    ]
    return pd.DataFrame([scores], columns=['accuracy', 'precision', 'recall', 'f1 score'])


def predict_for_svm(model: svm.SVM, X, y, n_iters: int, lambda_param: int, file: UploadedFile):
    y = np.where(y <= 0, -1, 1)

    svm.SVM.re_fit(model, n_iters, lambda_param, X, y)

    y_pred = model.predict(X)

    # reconstruct y value
    y_normalized_pred = np.where(y_pred == -1, 0, 1)
    y_normalized = np.where(y == -1, 0, 1)

    # print(y_normalized)
    # print(y_normalized_pred)

    scores = [
        accuracy_score(y_normalized, y_normalized_pred),
        precision_score(y_normalized, y_normalized_pred,
                        pos_label=0, average="binary"),
        recall_score(y_normalized, y_normalized_pred,
                     pos_label=0, average="binary"),
        f1_score(y_normalized, y_normalized_pred,
                 pos_label=0, average="binary")
    ]
    return pd.DataFrame([scores], columns=['accuracy', 'precision', 'recall', 'f1 score'])


def preprocess_for_svm(df, std_scalers):
    new_df = df.copy()

    columns_for_std_scales = ['stag', 'age', 'extraversion', 'independ', 'selfcontrol',
                              'anxiety', 'novator']

    for column in columns_for_std_scales:
        # perform standard scaling
        new_df[column] = std_scalers[column].transform(df[[column]])

    return new_df


def get_df(file: UploadedFile, label_encoders, std_scalers):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    df = preprocess_data(df, label_encoders, std_scalers)

    return df
