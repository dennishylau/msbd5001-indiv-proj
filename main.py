# %%
import os
from typing import Union
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
from datetime import datetime

# %%
XGB_PARAM = {
    'n_estimators': range(10, 60, 10),
    'learning_rate': [0.01, 0.1, 0.25, 0.5, 1],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eval_metric': ['error', 'logloss']
}

DATA_TRAIN_PATH = './data/train.csv'
DATA_TEST_PATH = './data/test.csv'

MODEL_PATH = 'output/model.json'


# %%

def load_data(path: str) -> tuple['pd.DataFrame', pd.Series]:
    df = pd.read_csv(path, index_col='id')
    df.dropna(inplace=True)
    y = df['label']
    X = df.loc[:, df.columns != 'label']
    X, y = SMOTE().fit_resample(X, y)
    return (X, y)


def save_model(clf: GridSearchCV, path: str):
    clf.best_estimator_.save_model(path)


def load_model(path: str) -> xgb.XGBRegressor:
    clf = xgb.XGBClassifier()
    clf.load_model(path)
    return clf


def train(X, y, xgb_param: dict):

    clf = GridSearchCV(
        xgb.XGBClassifier(use_label_encoder=False),
        xgb_param,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    print("Detailed classification report:")
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))

    return clf


def pred(
        data_path: str,
        clf: Union[xgb.XGBRegressor, GridSearchCV]) -> str:
    df_pred = pd.read_csv(data_path, index_col='id')
    df_pred['label'] = clf.predict(df_pred)
    file_path = f'output/csv/{datetime.now()}_submission.csv'
    df_pred['label'].to_csv(file_path)
    return file_path


def submit(file_path: str):
    os.system(
        f'kaggle competitions submit -c msbd5001-spring-2022 -f "{os.getcwd()}/{file_path}" -m ""')

# %%


if __name__ == "__main__":
    try:
        clf = load_model(MODEL_PATH)
    except xgb.core.XGBoostError:
        # path does not exist
        X, y = load_data(DATA_TRAIN_PATH)
        clf = train(X, y, XGB_PARAM)
        save_model(clf, MODEL_PATH)
    pred_file_path = pred(DATA_TEST_PATH, clf)
    submit(pred_file_path)
