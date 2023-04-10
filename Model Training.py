import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn import set_config
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split

data = pd.read_csv('Cleaned_data.csv', index_col=[0])
x = data.drop(columns=['price'])
y = data['price']

# Splitting the dataset into training data and testing data.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

print(x_train.shape)
print(x_test.shape)

"""Apply Linear Regression"""
# Using the OneHotEncoder to convert the nominal categorical data into numbers.
# Also ColumnTransformer is used to transform the columns.
column_trans = ColumnTransformer(transformers=[
    ('location', OneHotEncoder(sparse_output=False), ['location'])
], remainder='passthrough')

# StandardScaler is useed to scale the data accordiing to the normal distribution.
scaler = StandardScaler(with_mean=False)

# Here, Linear Regression is used to fit the data and train the model.
lr = LinearRegression()
#  Pipeline is created for LR to fit the model to the training data.
pipe = make_pipeline(column_trans, scaler, lr)

set_config(display='diagram')

# Fit the data.
pipe.fit(x_train, y_train)
print(pipe)
y_pred_lr = pipe.predict(x_test)

# Performance of the model.
print(r2_score(y_test, y_pred_lr))
# print(accuracy_score(y_test, y_pred_lr))


"""Applying Lasso"""
lasso = Lasso()

# Fit the data.
pipe1 = make_pipeline(column_trans, scaler, lasso)
pipe1.fit(x_train, y_train)
y_pred_lasso = pipe1.predict(x_test)

# Performance of the model.
print(r2_score(y_test, y_pred_lasso))

"""Applying Ridge"""
ridge = Ridge()

# Fit the data.
pipe2 = make_pipeline(column_trans, scaler, ridge)
pipe2.fit(x_train, y_train)
y_pred_ridge = pipe2.predict(x_test)

# Performance of the model.
print(r2_score(y_test, y_pred_ridge))

# Pickling the LR model into a file to unpickle later and predict the results of new data.
pickle.dump(pipe, open('LinearModel.pkl', 'wb'))