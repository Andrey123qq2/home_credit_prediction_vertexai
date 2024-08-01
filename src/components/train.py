from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Model,
    Output,
    ClassificationMetrics,
    component,
    pipeline,
)

@component(
    packages_to_install=['pyarrow', 'category_encoders', 'dill'],
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
)
def model_train(
    model_name:str,
    target_col: str,
    train_set: Input[Dataset],
    model: Output[Model],
    serving_container_image_uri: str,
    model_params: dict = None,
):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import BaseEstimator, TransformerMixin
    from  category_encoders import CatBoostEncoder
    from pathlib import Path
    import pandas as pd
    import numpy as np    
    import warnings
    import logging
    import gc

    if model_params is None:
        model_params = {}

    ### Features
    #region
    NUMERICAL_FEATURES = [
        'MONTH',
        'WEEK_NUM',
        'actualdpdtolerance_344P',
        'amtinstpaidbefduel24m_4187115A',
        'annuity_780A',
        'annuitynextmonth_57A',
        'applicationcnt_361L',
        'applications30d_658L',
        'applicationscnt_1086L',
        'applicationscnt_464L',
        'applicationscnt_629L',
        'applicationscnt_867L',
        'avgdbddpdlast24m_3658932P',
        'avgdbddpdlast3m_4187120P',
        'avgdbdtollast24m_4525197P',
        'avgdpdtolclosure24_3658938P',
        'avginstallast24m_3658937A',
        'avglnamtstart24m_4525187A',
        'avgmaxdpdlast9m_3716943P',
        'avgoutstandbalancel6m_4187114A',
        'avgpmtlast12m_4525200A',
        'clientscnt12m_3712952L',
        'clientscnt3m_3712950L',
        'clientscnt6m_3712949L',
        'clientscnt_100L',
        'clientscnt_1022L',
        'clientscnt_1071L',
        'clientscnt_1130L',
        'clientscnt_136L',
        'clientscnt_157L',
        'clientscnt_257L',
        'clientscnt_304L',
        'clientscnt_360L',
        'clientscnt_493L',
        'clientscnt_533L',
        'clientscnt_887L',
        'clientscnt_946L',
        'cntincpaycont9m_3716944L',
        'cntpmts24_3658933L',
        'commnoinclast6m_3546845L',
        'credamount_770A',
        'currdebt_22A',
        'currdebtcredtyperange_828A',
        'daysoverduetolerancedd_3976961L',
        'deferredmnthsnum_166L',
        'disbursedcredamount_1113A',
        'downpmt_116A',
        'eir_270L',
        'homephncnt_628L',
        'inittransactionamount_650A',
        'interestrate_311L',
        'interestrategrace_34L',
        'lastapprcredamount_781A',
        'lastdependentsnum_448L',
        'lastotherinc_902A',
        'lastotherlnsexpense_631A',
        'lastrejectcredamount_222A',
        'maininc_215A',
        'mastercontrelectronic_519L',
        'mastercontrexist_109L',
        'maxannuity_159A',
        'maxannuity_4075009A',
        'maxdbddpdlast1m_3658939P',
        'maxdbddpdtollast12m_3658940P',
        'maxdbddpdtollast6m_4187119P',
        'maxdebt4_972A',
        'maxdpdfrom6mto36m_3546853P',
        'maxdpdinstlnum_3546846P',
        'maxdpdlast12m_727P',
        'maxdpdlast24m_143P',
        'maxdpdlast3m_392P',
        'maxdpdlast6m_474P',
        'maxdpdlast9m_1059P',
        'maxdpdtolerance_374P',
        'maxinstallast24m_3658928A',
        'maxlnamtstart6m_4525199A',
        'maxoutstandbalancel12m_4187113A',
        'maxpmtlast3m_4525190A',
        'mindbddpdlast24m_3658935P',
        'mindbdtollast24m_4525191P',
        'mobilephncnt_593L',
        'monthsannuity_845L',
        'numactivecreds_622L',
        'numactivecredschannel_414L',
        'numactiverelcontr_750L',
        'numcontrs3months_479L',
        'numincomingpmts_3546848L',
        'numinstlallpaidearly3d_817L',
        'numinstls_657L',
        'numinstlsallpaid_934L',
        'numinstlswithdpd10_728L',
        'numinstlswithdpd5_4187116L',
        'numinstlswithoutdpd_562L',
        'numinstmatpaidtearly2d_4499204L',
        'numinstpaid_4499208L',
        'numinstpaidearly3d_3546850L',
        'numinstpaidearly3dest_4493216L',
        'numinstpaidearly5d_1087L',
        'numinstpaidearly5dest_4493211L',
        'numinstpaidearly5dobd_4499205L',
        'numinstpaidearly_338L',
        'numinstpaidearlyest_4493214L',
        'numinstpaidlastcontr_4325080L',
        'numinstpaidlate1d_3546852L',
        'numinstregularpaid_973L',
        'numinstregularpaidest_4493210L',
        'numinsttopaygr_769L',
        'numinsttopaygrest_4493213L',
        'numinstunpaidmax_3546851L',
        'numinstunpaidmaxest_4493212L',
        'numnotactivated_1143L',
        'numpmtchanneldd_318L',
        'numrejects9m_859L',
        'pctinstlsallpaidearl3d_427L',
        'pctinstlsallpaidlat10d_839L',
        'pctinstlsallpaidlate1d_3546856L',
        'pctinstlsallpaidlate4d_3546849L',
        'pctinstlsallpaidlate6d_3546844L',
        'pmtnum_254L',
        'posfpd10lastmonth_333P',
        'posfpd30lastmonth_3976960P',
        'posfstqpd30lastmonth_3976962P',
        'price_1097A',
        'sellerplacecnt_915L',
        'sellerplacescnt_216L',
        'sumoutstandtotal_3546847A',
        'sumoutstandtotalest_4493215A',
        'totaldebt_9A',
        'totalsettled_863A',
        'totinstallast1m_4525188A',
        'contractssum_5085716L',
        'days120_123L',
        'days180_256L',
        'days30_165L',
        'days360_512L',
        'days90_310L',
        'firstquarter_103L',
        'for3years_128L',
        'for3years_504L',
        'for3years_584L',
        'formonth_118L',
        'formonth_206L',
        'formonth_535L',
        'forquarter_1017L',
        'forquarter_462L',
        'forquarter_634L',
        'fortoday_1092L',
        'forweek_1077L',
        'forweek_528L',
        'forweek_601L',
        'foryear_618L',
        'foryear_818L',
        'foryear_850L',
        'fourthquarter_440L',
        'numberofqueries_373L',
        'pmtaverage_3A',
        'pmtaverage_4527227A',
        'pmtaverage_4955615A',
        'pmtcount_4527229L',
        'pmtcount_4955617L',
        'pmtcount_693L',
        'pmtscount_423L',
        'pmtssum_45A',
        'riskassesment_940T',
        'secondquarter_766L',
        'thirdquarter_1082L'
    ]
    CATEGORICAL_FEATURES = [
        'bankacctype_710L',
        'cardtype_51L',
        'credtype_322L',
        'disbursementtype_67L',
        'equalitydataagreement_891L',
        'equalityempfrom_62L',
        'inittransactioncode_186L',
        'isbidproduct_1095L',
        'isbidproductrequest_292L',
        'isdebitcard_729L',
        'lastapprcommoditycat_1041M',
        'lastapprcommoditytypec_5251766M',
        'lastcancelreason_561M',
        'lastrejectcommoditycat_161M',
        'lastrejectcommodtypec_5251769M',
        'lastrejectreason_759M',
        'lastrejectreasonclient_4145040M',
        'lastst_736L',
        'opencred_647L',
        'paytype1st_925L',
        'paytype_783L',
        'previouscontdistrict_112M',
        'twobodfilling_608L',
        'typesuite_864L',
        'description_5085714M',
        'education_1103M',
        'education_88M',
        'maritalst_385M',
        'maritalst_893M',
        'requesttype_4525192L',
        'riskassesment_302T'
    ]
    DATE_FEATURES = [
        'date_decision',
        'datefirstoffer_1144D',
        'datelastinstal40dpd_247D',
        'datelastunpaid_3546854D',
        'dtlastpmtallstes_4499206D',
        'firstclxcampaign_1125D',
        'firstdatedue_489D',
        'lastactivateddate_801D',
        'lastapplicationdate_877D',
        'lastapprdate_640D',
        'lastdelinqdate_224D',
        'lastrejectdate_50D',
        'lastrepayingdate_696D',
        'maxdpdinstldate_3546855D',
        'validfrom_1069D',
        'assignmentdate_238D',
        'assignmentdate_4527235D',
        'assignmentdate_4955616D',
        'birthdate_574D',
        'dateofbirth_337D',
        'dateofbirth_342D',
        'responsedate_1012D',
        'responsedate_4527233D',
        'responsedate_4917613D',
        'payvacationpostpone_4187118D',
    ]
    #endregion

    ### Preprocessing classes
    #region
    #Numerical Imputer
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
                X[feature] = X[feature].fillna(self.imputer_dict_[feature])
            return X
        
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

    #Categorical Imputer
    class CategoricalImputer(BaseEstimator, TransformerMixin):
        """Categorical Data Missing Value Imputer"""
        def __init__(self, variables=None):
            self.variables = variables
        
        def fit(self, X,y=None):
            self.imputer_dict_={}
            for feature in self.variables:
                self.imputer_dict_[feature] = X[feature].mode()[0]
            return self
        
        def transform(self, X):
            for feature in self.variables:
                X[feature] = X[feature].fillna(self.imputer_dict_[feature])
            return X

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
        
    class TableDtypesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            for col in X.columns:
                if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                    X[col] = X[col].astype('int64')
                elif col in ["date_decision"]:
                    X[col] = pd.to_datetime(X[col])
                elif col[-1] in ("P", "A"):
                    X[col] = X[col].astype('float64')
                elif col[-1] in ("M",) or 'person' in col:
                    X[col] = X[col].astype('category')
                elif col[-1] in ("D",):
                    X[col] = pd.to_datetime(X[col])
            return X
        
        def get_feature_names_out(self, input_features=None):
            return input_features

    class DowncastTransformer(BaseEstimator, TransformerMixin):
        """
        Reduce memory usage of a Pandas DataFrame by converting 
        object types to categories and downcasting numeric columns
        """
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            start_mem = X.memory_usage().sum() / 1024**2
            logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
            object_cols, int_cols, float_cols = [], [], []
            for col, dtype in X.dtypes.items():
                if pd.api.types.is_object_dtype(dtype):
                    object_cols.append(col)
                elif pd.api.types.is_integer_dtype(dtype):
                    int_cols.append(col)
                elif pd.api.types.is_float_dtype(dtype):
                    float_cols.append(col)
            X[object_cols] = X[object_cols].astype('category')
            X[int_cols] = X[int_cols].apply(pd.to_numeric, downcast='integer')
            X[float_cols] = X[float_cols].apply(pd.to_numeric, downcast='float')
            
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
    #endregion

    all_feats = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES
    train_df = pd.read_parquet(train_set.path)
    logging.info("train_df load completed")
    X = train_df[all_feats]
    y = train_df[target_col]

    del train_df
    gc.collect()

    model_pipeline = Pipeline(
        [
            ('TableDtypes Transformer', TableDtypesTransformer()),
            ('Downcast Transformer', DowncastTransformer()),
            ('Numerical Imputer', NumericalImputer(variables=NUMERICAL_FEATURES)),
            ('Categorical Imputer', CategoricalImputer(variables=CATEGORICAL_FEATURES)),
            ('Categorical Encoder', CatBoostEncoder(cols=CATEGORICAL_FEATURES)),
            ('Dates Imputer', DatesImputer(variables=DATE_FEATURES)),
            ('Dates Transformer', DateColsTransformer(date_cols=DATE_FEATURES)),
            # ('Debugger', Debugger()),
            ('Log Transform', Log1pTransformer(num_cols=NUMERICAL_FEATURES)),
            ('Scaler Transform', MinMaxScaler(copy=False)),
            ('Linear Model', LogisticRegression(**model_params))
        ], verbose=True
    )
    model_pipeline.fit(X, y)
    logging.info("model fit completed")

    pickle_output_path = model.path + '.pkl'
    import dill
    with open(pickle_output_path, 'wb') as file:
        dill.dump(
            obj=model_pipeline,
            file=file,
            recurse=True,
        )

    if model:
        model.metadata = {
            "containerSpec": {"imageUri": serving_container_image_uri},
            "framework": "scikit-learn",
            "model_name": model_name,
            "model_path": pickle_output_path,
        }
        model.path = pickle_output_path