import zipfile
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with zipfile.ZipFile("boston-house-prices.zip","r") as zip_ref:
    zip_ref.extractall("./dataset")

class Housing:
    def __init__(self):
        header_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.data = pd.read_csv('./dataset/housing.csv', names = header_list, delim_whitespace=True)

    def scale_features(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def fill_null(self, df):
        df = df.fillna(0)
        return df

    def plot_fit_line(self, X_test, y_test, prediction, feature, target, i):
        f = plt.figure()
        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, prediction,'-', color='blue')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title('Figure %d' %i)
        plt.savefig('line_plot.pdf') 
        plt.show()   
        f.savefig('%d.jpg'%i)
        pass
    
    def find_one_best_feature(self):
        best_r2 = -1;
        best_feature = '';

        for v in self.data.columns[:-1]:

            X = self.data.loc[:, v]
            y = self.data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            X_train = X_train.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)
            y_train = y_train.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            prediction = lr.predict(X_test)
            r2 = r2_score(y_test, prediction)
            #print(v, ' ', r2)
            if r2 > best_r2:
                best_r2 = r2
                best_feature = v
        return best_feature


    def linear_regression_predict(self, feature):
        X = self.data.loc[:, feature]
        y = self.data.iloc[:, -1]
        X = X.values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        return X_test, y_test, prediction

    def polynomial_predict(self, feature, degree):
        X = self.data.loc[:, feature]
        X = X.values.reshape(-1, 1)
        y = self.data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        lr = LinearRegression()
        lr.fit(X_train_poly, y_train)
        X_test_poly = poly.fit_transform(X_test)
        prediction = lr.predict(X_test_poly)
        return X_test, y_test, prediction

    def multi_linear_predict(self, features):
        X = self.data.loc[:, features]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = self.data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        return X_test, y_test, prediction


        
    #def plot_polynomial()

housing = Housing()
#Task I: Linear Regression
#Build a model using a linear regression (Scikit-learn) algorithm to predict house prices.
#Find the feature with the best r2_score
feature = housing.find_one_best_feature()
#Use this feature to make prediction
X_test, y_test, prediction_linear = housing.linear_regression_predict(feature)
#Plot the prediction and the best fit line
housing.plot_fit_line(X_test, y_test, prediction_linear, feature, 'MEDV', 1)
#RMSE
rmse_linear = mean_squared_error(y_test, prediction_linear, squared = False)
print('Linear regression (feature: %s) RMSE = ' %feature, rmse_linear)
# R2 score
r2_linear = r2_score(y_test, prediction_linear)
print('Linear regression (feature: %s) r2 score = ' %feature, r2_linear)



## II - Polynomial Regression
X_test, y_test, prediction_poly_2 = housing.polynomial_predict(feature, 2)
#Plot the prediction and the best fit line
X_test = X_test.reshape(-1)
y_test = y_test.values.reshape(-1)
ind = X_test.argsort()
housing.plot_fit_line(X_test[ind], y_test[ind], prediction_poly_2[ind], feature, 'MEDV', 2)
#RMSE
rmse_poly = mean_squared_error(y_test, prediction_poly_2, squared = False)
print('Polynomial regression (feature: %s) RMSE = ' %feature, rmse_poly)
#R2 score
r2_poly = r2_score(y_test, prediction_poly_2)
print('Polynomial regression (feature: %s) r2 score = ' %feature, r2_poly)
#Plot another diagram for degree=20.
X_test, y_test, prediction_poly_20 = housing.polynomial_predict(feature, 20)
#Plot the prediction and the best fit line for degree 20
X_test = X_test.reshape(-1)
y_test = y_test.values.reshape(-1)
ind = X_test.argsort
housing.plot_fit_line(X_test[ind], y_test[ind], prediction_poly_20[ind], feature, 'MEDV', 3)



## III - Multiple Regression, 
features = ['LSTAT', 'RM', 'PTRATIO']
X_test, y_test, prediction_multi = housing.multi_linear_predict(features)
#RMSE
rmse_multi = mean_squared_error(y_test, prediction_multi, squared = False)
print('Multi linear regression (feature: %s) RMSE = ' %features, rmse_multi)
#R2 score
r2_multi = r2_score(y_test, prediction_multi)
print('Multi linear regression (feature: %s) r2 score = ' %features, r2_multi)
#Adjusted R2 score
r = r2_multi
n = len(X_test)
k = 3
r_adj = 1 - (1 - r)* (n - 1) / (n - k - 1)
print('Multi linear regression (feature: %s) adjusted r2 score = ' %features, r_adj)
