# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
# Loading the dataframe
df = pd.read_csv(path)
# Loading first five records
df.head()
# Independent Variables
X = df.drop("Price", axis = 1)
#Store dependent variables
y = df['Price']
# Spiliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)
#Check correlation
corr = X_train.corr()
# Print correlation
print(corr)
#Code starts here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
# Instantiate the model
regressor = LinearRegression()
# fit the model
regressor.fit(X_train, y_train)
# Predict the model
y_pred = regressor.predict(X_test)
# Find the r-score
r2 = regressor.score(X_test, y_test)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = lasso.score(X_test, y_test)


# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
r2_ridge = ridge.score(X_test, y_test)
# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor = LinearRegression()
score = cross_val_score(regressor, X_train, y_train,scoring = "r2", cv = 10)
print(score)
mean_score = np.mean(score)
print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_poly = model.score(X_test, y_test)


