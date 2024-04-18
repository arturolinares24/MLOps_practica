import unittest
import os
from MLOps_practica.dags.automl import MLSystem

class TestMLSystem(unittest.TestCase):
    def setUp(self):
        # Configura las variables de entorno para Kaggle
        os.environ['KAGGLE_USERNAME'] = 'arturolinares'
        os.environ['KAGGLE_KEY'] = '5b239d1dc000d187595039f836a68a40'

    def test_load_data(self):
        # Crea una instancia de MLSystem
        system = MLSystem()
        # Verifica que no haya errores al cargar los datos
        self.assertIsNone(system.load_data())

    def test_preprocess_data(self):
        # Crea una instancia de MLSystem
        system = MLSystem()
        # Carga los datos
        system.load_data()
        # Verifica que los datos se preprocesen correctamente
        train, test, submission = system.preprocess_data()
        # Verifica que train y test no estén vacíos
        self.assertFalse(train.empty)
        self.assertFalse(test.empty)
        # Verifica que 'Rings' se elimine de test
        self.assertNotIn('Rings', test.columns)

    def test_train_model(self):
        # Crea una instancia de MLSystem
        system = MLSystem()
        # Carga y preprocesa los datos
        system.load_data()
        train, _, _ = system.preprocess_data()
        # Entrena el modelo
        model1_path, model2_path, model3_path = system.train_model(train)
        # Verifica que los archivos de modelo se creen correctamente
        self.assertTrue(os.path.exists(model1_path))
        self.assertTrue(os.path.exists(model2_path))
        self.assertTrue(os.path.exists(model3_path))

    def test_create_submission_file(self):
        # Crea una instancia de MLSystem
        system = MLSystem()
        # Carga y preprocesa los datos
        system.load_data()
        train, test, submission = system.preprocess_data()
        # Entrena el modelo
        model1_path, model2_path, model3_path = system.train_model(train)
        # Crea el archivo de sumisión
        system.create_submission_file(test, submission, model1_path, model2_path, model3_path)
        # Verifica que el archivo de sumisión se haya creado correctamente
        self.assertTrue(os.path.exists('/opt/airflow/dags/data/submission_final.csv'))

    def test_run_entire_workflow(self):
        # Crea una instancia de MLSystem
        system = MLSystem()
        # Ejecuta el flujo de trabajo completo
        result = system.run_entire_workflow()
        # Verifica que el resultado sea exitoso
        self.assertTrue(result['success'])

if __name__ == '__main__':
    unittest.main()
