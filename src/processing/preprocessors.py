import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder

### Preprocessing classes
class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        for feature in self.variables:
            # print(f"feature: {feature}, {self.imputer_dict_[feature]}")
            X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class DatesImputer(BaseEstimator, TransformerMixin):
    """Numerical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self,X):
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self
    
    def transform(self, X):
        for feature in self.variables:
            na_value = self.imputer_dict_[feature]
            if X[feature].dtype.name == 'category' and na_value not in X[feature].cat.categories:
                X[feature] = X[feature].cat.add_categories(na_value)
            X[feature] = X[feature].fillna(na_value)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[], threshold=100):
        super().__init__()
        self.num_cols = num_cols
        self.threshold = threshold
        self.outliers_cols=[]
        
    def fit(self, X, y=None):
        X = X[self.num_cols]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            X_desc = X.describe()
        max_to_mean = (
            np.abs(X_desc.loc['max'] / X_desc.loc['mean'])
        )
        min_to_mean = (
            np.abs(X_desc.loc['min'] / X_desc.loc['mean'])
        )
        max_to_mean_finite = max_to_mean[np.isfinite(max_to_mean)]
        min_to_mean_finite = min_to_mean[np.isfinite(min_to_mean)]
        outliers_cols_max = list(max_to_mean_finite[max_to_mean_finite > self.threshold].index)
        outliers_cols_min = list(min_to_mean_finite[min_to_mean_finite > self.threshold].index)
        self.outliers_cols = outliers_cols_max + outliers_cols_min
        return self
    
    def transform(self, X):
        for col in self.outliers_cols:
            X[col] = np.log1p(np.abs(X[col])) * np.sign(X[col])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class DateColsTransformer(BaseEstimator, TransformerMixin):
    """Feature Engineering"""
    def __init__(self, reference_date_col='date_decision', date_cols=[]):
        self.date_cols = date_cols
        self.ref_col = reference_date_col
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        X['month_decision'] = X["date_decision"].dt.month.astype('int16')
        X['weekday_decision'] = X["date_decision"].dt.month.astype('int16')
        X['day_decision'] = X["date_decision"].dt.month.astype('int16')
        
        for col_name in self.date_cols:
            if col_name == 'date_decision':
                continue
            X[col_name] = X[col_name] - X[self.ref_col]
            X[col_name] = X[col_name].dt.days.astype('int32')
        X = X.drop("date_decision", axis=1)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class TableDtypesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.columns = X.columns
        self.int64_cols = []
        self.date_cols = []
        self.float64_cols = []
        self.cat_cols = []
        for col in self.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                self.int64_cols.append(col)
            elif col in ["date_decision"] or col[-1] in ("D",):
                self.date_cols.append(col)
            elif col[-1] in ("P", "A"):
                self.float64_cols.append(col)
            elif col[-1] in ("M",) or 'person' in col or pd.api.types.is_object_dtype(X[col].dtype):
                self.cat_cols.append(col)
        return self

    def transform(self, X):
        for col in self.int64_cols:
            X[col] = X[col].astype('int64')
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col])
        for col in self.float64_cols:
            X[col] = X[col].astype('float64')
        for col in self.cat_cols:
            X[col] = X[col].astype('category')
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.columns

class DowncastTransformer(BaseEstimator, TransformerMixin):
    """
    Reduce memory usage of a Pandas DataFrame by converting 
    object types to categories and downcasting numeric columns
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.int_cols = [] 
        self.float_cols = []
        self.object_cols = []
        for col, dtype in X.dtypes.items():
            if pd.api.types.is_object_dtype(dtype):
                self.object_cols.append(col)
            elif pd.api.types.is_integer_dtype(dtype):
                self.int_cols.append(col)
            elif pd.api.types.is_float_dtype(dtype):
                self.float_cols.append(col)
        return self

    def transform(self, X):
        start_mem = X.memory_usage().sum() / 1024**2
        logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        X[self.object_cols] = X[self.object_cols].astype('category')
        X[self.int_cols] = X[self.int_cols].apply(pd.to_numeric, downcast='integer')
        X[self.float_cols] = X[self.float_cols].apply(pd.to_numeric, downcast='float')
        
        end_mem = X.memory_usage().sum() / 1024**2
        logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class Debugger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        logging.info(f"X.shape: {X.shape}")
        logging.info(f"X type: {type(X)}")
        logging.info(f"inf count: {np.count_nonzero(np.isinf(X))}")
        logging.info(f"nan count: {np.count_nonzero(np.isnan(X))}")
        logging.info(f"nan cols: {X.columns[X.isna().any()].tolist()}")            
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class TargetEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
    
    def fit(self, X, y=None):
        self.encoder = TargetEncoder().fit(X[self.cols], y)
        return self
    
    def transform(self, X):
        X_tr = self.encoder.transform(X[self.cols])
        X[self.cols] = X_tr
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features