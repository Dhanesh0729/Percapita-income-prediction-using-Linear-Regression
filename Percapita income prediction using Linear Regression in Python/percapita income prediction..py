import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv(r"C:\Users\Dhanesh\Python 3.11\house price pediction linear regression py\canada_per_capita_income.csv")
print(df1)
print("\n")

plt.xlabel('year')
plt.ylabel('per_capita_income')
plt.scatter(df1.year,df1.per_capita_income)

yeary=df1['year'].values.reshape(-1,1)
print(yeary)

income=df1['per_capita_income']
print(income)

reg = linear_model.LinearRegression()
reg.fit(yeary,income)
predicted_value=reg.predict([[2023]])
print(predicted_value)

