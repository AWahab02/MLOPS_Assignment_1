import unittest
from main import salary, X_train, X_test, y_train, y_test, model

class TestMain(unittest.TestCase):
    def test_salary_data_loaded(self):
        self.assertIsNotNone(salary)
        self.assertGreater(len(salary), 0)
        self.assertIn('Experience Years', salary.columns)
        self.assertIn('Salary', salary.columns)

    def test_train_test_split(self):
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertEqual(X_train.shape[1], 1)
        self.assertEqual(X_test.shape[1], 1)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_linear_regression_model(self):
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'LinearRegression')
        self.assertEqual(model.coef_.shape[0], 1)
        self.assertIsNotNone(model.intercept_)

if __name__ == '__main__':
    unittest.main()
