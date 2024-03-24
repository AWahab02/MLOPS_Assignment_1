import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

salary_url = 'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv'
salary = pd.read_csv(salary_url)

X = salary[['Experience Years']]
y = salary['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)



joblib.dump(model, 'model.pkl')

predictions = model.predict(X_test[:5])
print("Predictions:", predictions)