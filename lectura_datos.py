import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
#import tensorflow as tf
df = pd.read_excel('datos1.xlsx', header=None)
df.head()


matrix=np.array(df)


X = df.iloc[:, 0:4].values


y=df.iloc[:, 0:4]



#y = df.target
#z = df.iloc[:,2]
#w = df.iloc[:,3]









plt.scatter(X,y)
plt.show()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = linear_model.LinearRegression()


lr.fit(np.array(X_train), y_train)

Y_pred = lr.predict(np.array(X_test))
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresion LINEAL SIMPLE')
plt.xlabel('2')
plt.ylabel('Valor medio')
plt.show()






