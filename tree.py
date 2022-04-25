# %% adaboost
import os
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./data/train.csv', index_col='id')
df.dropna(inplace=True)
y = df['label']
X = df.loc[:, df.columns != 'label']
X, y = SMOTE().fit_resample(X, y)

feat = [
    'MO HLADR+ MFI (cells/ul)',
    'Neu CD64+MFI (cells/ul)',
    'CD3+T (cells/ul)',
    'CD8+T (cells/ul)',
    'CD4+T (cells/ul)',
    'NK (cells/ul)',
    'CD19+ (cells/ul)',
    'CD45+ (cells/ul)',
    'Age',
    'Sex 0M1F',
    'Mono CD64+MFI (cells/ul)'
]
X = X[feat]

# feature selection
# feat_sel = SelectKBest(f_classif, k=4)
# X = feat_sel.fit_transform(X, y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0)


# %%
scores = [
    'accuracy',
    # 'precision_macro', 'recall_macro', 'f1_macro'
]

# adaboost_param = {
#     'n_estimators': range(10, 60, 10),
#     'learning_rate': [0.01, 0.1, 0.25, 0.5, 1],
#     'algorithm': ['SAMME', 'SAMME.R'],
# }

# gradientboost_param = {

# }

xgb_param = {
    'n_estimators': range(10, 60, 10),
    'learning_rate': [0.01, 0.1, 0.25, 0.5, 1],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eval_metric': ['error', 'logloss']
}


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(
        # AdaBoostClassifier(),
        # adaboost_param,
        xgb.XGBClassifier(use_label_encoder=False),
        xgb_param,
        cv=5,
        scoring=score,
        n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    print("Detailed classification report:")
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))


# %%
# xgb plot importance
# xgb.plot_importance(clf.best_estimator_)
# plt.figure(figsize=(16, 12))
# plt.show()

selection_models = []
accuracies = []

# select features using threshold
# for thres in np.sort(clf.best_estimator_.feature_importances_):
#     selection = SelectFromModel(clf.best_estimator_, threshold=thres, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = xgb.XGBClassifier()
#     selection_model.fit(select_X_train, y_train)
#     selection_models.append(selection_model)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     print(classification_report(y_true, y_pred))
#     accuracies.append(accuracy_score(y_true, y_pred))

# %%
df_pred = pd.read_csv('./data/test.csv', index_col='id')
# X_pred = feat_sel.transform(df_pred)
df_pred['label'] = clf.predict(df_pred)
file = f'output/csv/{datetime.now()}_submission.csv'
df_pred['label'].to_csv(file)


# %%
os.system(
    f'kaggle competitions submit -c msbd5001-spring-2022 -f "{os.getcwd()}/{file}" -m ""')

# %%
results = cross_validate(
    estimator=GradientBoostingClassifier(),
    X=X,
    y=y,
    cv=5,
    scoring=[
        'accuracy',
        # 'precision', 'recall', 'f1'
    ],
    return_train_score=True)

print(results)
print(np.average(results['test_accuracy']))
