!mamba install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
#loading Dataset
housing = fetch_california_housing()
#converting to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

df['price'] = housing.target

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDataset Description:\n")
print(df.describe())
plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(12, 10), bins=30)

plt.tight_layout()
plt.show()

X = df.drop('price', axis=1)

#Target
y = df['price']

#split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape :", X_test.shape)

#plot
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("\nMAE :", mae)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", rmse)
r2 = r2_score(y_test, y_pred)
print("R2 Score :", r2)

#plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_pred,y=residuals)
plt.axhline(y=0,color='red',linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])

print("\nFeature Coefficients:\n")

print(coeff_df)

