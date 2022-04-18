# %%
from sklearn.model_selection import KFold, StratifiedKFold
from pandas import DataFrame, Series

# %%


def kfold(
    X: DataFrame, y: Series,
    *, num_fold: int = 10, stratify: bool = True) -> tuple[
        DataFrame, DataFrame, Series, Series
]:
    '''
    Input: X and y
    Returns: tuple(X_train, X_test, y_train, y_test)
    '''
    fold_func = StratifiedKFold if stratify else KFold
    skf = fold_func(n_splits=num_fold, shuffle=True)

    # for train_index, test_index in skf.split(X, y):
    train_index, test_index = next(skf.split(X, y))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test
