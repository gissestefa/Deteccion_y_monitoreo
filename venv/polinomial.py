import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
boston = datasets.load_boston()
print(boston)
print()
print('Informacion del Dataset')
print(boston.keys())
print('Cantidad de datos:')
print(boston.data.shape)
#print(boston.DESCR)
print('Nombres columnas')
print(boston.feature_names)
X = boston.data[:, np.newaxis, 5]
y = boston.target

plt.scatter(X, y)
plt.xlabel('numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
Y_pred = lr.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresion LINEAL SIMPLE')
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()
print()
print('DATOS DEL MODELO REGRESION LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a"')
print(lr.coef_)
print('Valor de la interseccion o coeficiente "b"')
print(lr.intercept_)
print()
print('La ecuacion del modelo es igual a:')
print(lr.coef_)
print('Valor de la interseccion o coeficiente "b"')
print(lr.intercept_)
print()
print('La ecuacion del modelo es igual a:')
print('y=', lr.coef_, 'x', lr.intercept_)


#print(y_test)
#print(Y_pred)



#from sklearn.datasets import fetch_openml

#housing = fetch_openml(name="house_prices", as_frame=True)

#print=datasets.load_boston()

#print(housing)
#print(housing.keys())
#print(housing.DESCR)
#print(housing.feature_names)
#X = housing.data[:, np.newaxis, 5]
#y = housing.target
#plt.scatter(X, y)



