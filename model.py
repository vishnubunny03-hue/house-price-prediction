import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# dataset
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [10, 15, 20, 25, 30]
})

X = data[['area']]
y = data['price']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# evaluate
error = mean_squared_error(y_test, pred)
print("Error:", error)

print("Prediction for 1800:", model.predict([[1800]]))
