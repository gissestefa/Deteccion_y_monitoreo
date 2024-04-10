import pandas as pd
from sklearn import linear_model
# cargar el dataset en memoria
training_dataset = pd.read_excel('datos1.xlsx', header=None)
# crear un modelo que use el algoritmo de regresion lineal
# y entrenarlo con los datos de nuestro csv
regression_model = linear_model.LinearRegression()
print ("Training model...")
# entrenamiento del modelo
regression_model.fit(training_dataset['Correcta'])
print ("Model trained.")
# pedir al usuario que introduzca un area y calcular
# su precio usando nuestro modelo
input_area = int(input("Enter area: "))
proped_price = regression_model.predict([[input_area]])
print ("Proped price:", round(proped_price[0], 2))