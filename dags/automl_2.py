import pandas as pd
import numpy as np
from joblib import dump, load
from pycaret.regression import *
import zipfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import xgboost
from sklearn.impute import IterativeImputer

class MLSystem:
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = 'arturolinares'
        os.environ['KAGGLE_KEY'] = '5b239d1dc000d187595039f836a68a40'
        self.api = KaggleApi()
        pass

    def load_data(self):


        self.api.authenticate()
        # Download the competition files
        competition_name = 'playground-series-s4e4'
        download_path = '/opt/airflow/dags/data/'
        self.api.competition_download_files(competition_name, path=download_path)
        # Unzip the downloaded files
        for item in os.listdir(download_path):
            if item.endswith('.zip'):
                zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
                zip_ref.extractall(download_path)
                zip_ref.close()
                print(f"Unzipped {item}")

    def preprocess_data(self):

        def undummify(df, prefix_sep="__"):
            cols2collapse = {
                item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
            }
            series_list = []
            for col, needs_to_collapse in cols2collapse.items():
                if needs_to_collapse:
                    undummified = (
                        df.filter(like=col)
                        .idxmax(axis=1)
                        .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                        .rename(col)
                    )
                    series_list.append(undummified)
                else:
                    series_list.append(df[col])
            undummified_df = pd.concat(series_list, axis=1)
            return undummified_df

        def reduce_mem_usage(df):
            """ iterate through all the columns of a dataframe and modify the data type
                to reduce memory usage.
            """
            start_mem = df.memory_usage().sum() / 1024 ** 2
            print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

            for col in df.columns:
                col_type = df[col].dtype

                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype('object')

            end_mem = df.memory_usage().sum() / 1024 ** 2
            print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
            print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

            return df


        train = pd.read_csv('/opt/airflow/dags/data/train.csv')
        test = pd.read_csv('/opt/airflow/dags/data/test.csv')
        train = train.drop(['id'], axis=1)
        test = test.drop(['id'], axis=1)
        submission = pd.read_csv('/opt/airflow/dags/data/sample_submission.csv')

        train.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                         'Viscera weight', 'Shell weight', 'Rings']
        test.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                        'Viscera weight', 'Shell weight']

        train['Height'] = train['Height'].replace(0, np.NaN)
        test['Height'] = test['Height'].replace(0, np.NaN)
        train = pd.get_dummies(train, prefix_sep='__')
        test = pd.get_dummies(test, prefix_sep='__')

        col = train.columns.tolist()
        col.remove('Rings')
        col[:5]
        iimp = IterativeImputer(
            estimator=xgboost.XGBRegressor(),
            random_state=42
        )
        train_ = iimp.fit_transform(train[col])
        train_ = pd.DataFrame(train_, columns=col)

        test = iimp.transform(test[col])
        test = pd.DataFrame(test, columns=col)

        train_['Rings'] = train['Rings']

        train = undummify(train_)

        test = undummify(test)

        train = reduce_mem_usage(train)

        test = reduce_mem_usage(test)

        return train,test,submission

    def train_model(self, train):
        # Simula el entrenamiento de un modelo
        reg = setup(train,use_gpu=True,
                       target='Rings',
                       session_id=42,
                       train_size=0.999)

        models()

        top = compare_models(sort='rmsle', include=['xgboost', 'lightgbm',
                                              'ada', 'dt',
                                              'gbr', 'rf',
                                              'et'
                                              ])
        lightgbm = create_model('lightgbm')
        tuned_lightgbm = tune_model(lightgbm,
                                    optimize='rmsle',
                                    search_library='scikit-optimize')
        tuned_lightgbm

        model1_path = '/opt/airflow/dags/models/model1.joblib'
        # Guardar los modelos en archivos
        dump(tuned_lightgbm, model1_path)
        model2_path = 1
        model3_path = 1
        return model1_path,model2_path,model3_path

    def create_submission_file(self, test ,submission, model1_path, model2_path=None, model3_path=None):
        model1 = load(model1_path)
        test_predictions1 = predict_model(model1, data=test)
        submission['Rings'] = test_predictions1['prediction_label']
        print(submission)
        # Verificar si hay valores nulos en la columna 'Rings'
        rings_nulls = submission['Rings'].isnull().any()

        # Imprimir el resultado
        print("¿La columna 'Rings' contiene valores nulos?", rings_nulls)
        submission.to_csv('/opt/airflow/dags/data/submission_final.csv', index=False)

    def run_entire_workflow(self):
        try:

            self.load_data()
            train, test, submission = self.preprocess_data()
            model1_path, model2_path, model3_path = self.train_model(train)
            self.create_submission_file(test ,submission, model1_path, model2_path, model3_path)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'message': str(e)}
