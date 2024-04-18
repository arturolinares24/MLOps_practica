import pandas as pd
import numpy as np
from joblib import dump, load
from pycaret.regression import *
import zipfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
# from sklearn.impute import KNNImputer
# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score, accuracy_score
# from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
# from sklearn.neural_network import MLPRegressor
# from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer, PowerTransformer, PolynomialFeatures
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor

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
        train = pd.read_csv('/opt/airflow/dags/data/train.csv')
        test = pd.read_csv('/opt/airflow/dags/data/test.csv')
        submission = pd.read_csv('/opt/airflow/dags/data/sample_submission.csv')
        original= pd.read_csv('/opt/airflow/dags/data/original.csv')
        original.columns = train.columns
        # train = pd.concat([train, original], axis=0, ignore_index=True)
        test['Rings'] = np.nan
        df = pd.concat([train, test], ignore_index=True, sort=False)

        train = df[df['Rings'].notnull()]
        test = df[df['Rings'].isnull()].drop('Rings', axis=1)
        # train['Sex'] = train['Sex'].map({'M': 0, 'F': 1, 'I': 2})
        # test['Sex'] = test['Sex'].map({'M': 0, 'F': 1, 'I': 2})

        APPLY_LOG_TRANSFORMATION = True
        def log_transformation(data, columns):
            for column in columns:
                positive_values = data[column] - data[column].min() + 1
                data[f'{column}_log'] = np.log(positive_values)
            return data

        if APPLY_LOG_TRANSFORMATION:
            columns_to_transform = [col for col in train.columns if col not in ['id', 'Rings','Sex']]
            train = log_transformation(train, columns_to_transform)
            test = log_transformation(test, columns_to_transform)

        train = pd.get_dummies(train, drop_first=False, dtype=float)
        test = pd.get_dummies(test, drop_first=False, dtype=float)
        # def drop_columns(df_train, df_test, col1, col2, col_t):
        #     df_train['Target'] = df_train[col_t]
        #     df_train = df_train.drop(col1, axis=1)
        #     df_test = df_test.drop(col2, axis=1)
        #     return df_train, df_test
        #
        # drop_columns_train = ['id', 'Rings']
        # drop_columns_test = ['id']
        # col_target = 'Rings'
        #
        # train, test = drop_columns(train, test, drop_columns_train, drop_columns_test, col_target)

        # train['W1_Weight_Ratio'] = train['Whole weight.1'] / train['Whole weight']
        # train['W2_Weight_Ratio'] = train['Whole weight.2'] / train['Whole weight']
        # train['Shell_Weight_Ratio'] = train['Shell weight'] / train['Whole weight']
        #
        # train['Weight_remains'] = train['Whole weight'] - train['Whole weight.1'] - train['Whole weight.2'] - train[
        #     'Shell weight']
        # train["Volume"] = train["Length"] * train["Diameter"] * train["Height"]
        #
        # test['W1_Weight_Ratio'] = test['Whole weight.1'] / test['Whole weight']
        # test['W2_Weight_Ratio'] = test['Whole weight.2'] / test['Whole weight']
        # test['Shell_Weight_Ratio'] = test['Shell weight'] / test['Whole weight']
        #
        # test['Weight_remains'] = test['Whole weight'] - test['Whole weight.1'] - test['Whole weight.2'] - test[
        #     'Shell weight']
        # test["Volume"] = test["Length"] * test["Diameter"] * test["Height"]

        numerical_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2',
                              'Shell weight', 'Sex_F', 'Sex_I', 'Sex_M','Rings']
        # numerical_features = ['W1_Weight_Ratio', 'W2_Weight_Ratio', 'Shell_Weight_Ratio',
        #                       'Weight_remains', 'Volume', 'Sex', 'Rings']
        numerical_features2 = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2',
                               'Shell weight', 'Sex_F', 'Sex_I', 'Sex_M']
        # numerical_features2 = ['W1_Weight_Ratio', 'W2_Weight_Ratio', 'Shell_Weight_Ratio',
        #                       'Weight_remains', 'Volume', 'Sex']
        train=train[numerical_features]
        train["Rings"]=np.log(1 + train["Rings"])  # Because RMSLE score, We make a conversion then
        test = test[numerical_features2]
        return train,test,submission

    def train_model(self, train):
        # Simula el entrenamiento de un modelo
        reg = setup(train,use_gpu=True, target='Rings',
                    ignore_features=["id"],
                    # max_encoding_ohe = 5,
                    # bin_numeric_features = [""],
                    # normalize=True,
                    # normalize_method="zscore",
                    # remove_multicollinearity=True,
                    # multicollinearity_threshold=0.99,
                    # remove_outliers=True,
                    # feature_selection = True,
                    # pca = True,
                    # polynomial_features=True,  # create feature (polynomial)
                    # polynomial_degree=2,
                    # fold=5,
                    # group_features = [""]
                    # create_clusters = True
                    )
        top3 = compare_models(n_select=3, sort="rmsle")
        model1 = create_model(top3[0])  # Modelo número 1
        model2 = create_model(top3[1])  # Modelo número 2
        model3 = create_model(top3[2])  # Modelo número 3
        model1_path = '/opt/airflow/dags/models/model1.joblib'
        model2_path = '/opt/airflow/dags/models/model2.joblib'
        model3_path = '/opt/airflow/dags/models/model3.joblib'
        # Guardar los modelos en archivos
        dump(model1, model1_path)
        dump(model2, model2_path)
        dump(model3, model3_path)

        return model1_path, model2_path, model3_path

    def create_submission_file(self, test ,submission, model1_path, model2_path, model3_path):
        model1 = load(model1_path)
        model2 = load(model2_path)
        model3 = load(model3_path)
        test_predictions1 = predict_model(model1, data=test)
        test_predictions2 = predict_model(model2, data=test)
        test_predictions3 = predict_model(model3, data=test)
        average_predictions = pd.DataFrame()
        average_predictions["prediction_label"] = (test_predictions1["prediction_label"] + test_predictions2[
            "prediction_label"] + test_predictions3["prediction_label"]) / 3
        average_predictions = average_predictions.reset_index(drop=True)
        submission['Rings'] = np.exp(average_predictions)-1
        submission['Rings'] = submission['Rings'].abs()
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
