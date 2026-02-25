from sklearn.base import BaseEstimator, TransformerMixin

class ModeImputation(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.modes_ = None
        self.dtypes_ = None
        self.columns_ = columns


    def fit(self, X_train, y=None):

        if X_train.empty:
            raise ValueError("Empty DataFrame passed to mode_imputation")

        self.modes_ = X_train.mode(dropna=True).iloc[0]
        self.dtypes_ = X_train.dtypes


    def transform(self, X_train):

        for col in X_train.columns:
            if col in self.columns_:
                try:
                    X_train[col] = X_train[col].fillna(self.modes_[col]).astype(self.dtypes_[col])
                except Exception as e:
                    raise Exception(f"Mode imputation failed for column '{col}': {e}")

        return X_train


    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
