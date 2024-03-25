import pytest
from main import salary, X_train, X_test, y_train, y_test, model

def test_salary_data_loaded():
    assert salary is not None
    assert len(salary) > 0
    assert 'Experience Years' in salary.columns
    assert 'Salary' in salary.columns

def test_train_test_split():
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert X_train.shape[1] == 1
    assert X_test.shape[1] == 1
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
