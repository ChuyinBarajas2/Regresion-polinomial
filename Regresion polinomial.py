import numpy as np
import matplotlib.pyplot as plt

# Datos 
Y = np.array([60, 70, 80, 100, 120, 150, 200, 250, 300, 400, 500, 750, 1000, 2000, 3000])
X = np.array([180, 180, 200, 200, 200, 220, 240, 240, 300, 350, 350, 360, 365, 365, 365])

objetosY=len(Y)
n = len(X)
X_squared = X ** 2
mse=0
sst=0
ssr=0
mediaY=sum(Y)/len(Y)

a = (n * np.sum(X * X_squared) - np.sum(X) * np.sum(X_squared)) / (n * np.sum(X * 2) - (np.sum(X)) * 2)
b = (np.sum(X) * np.sum(X * Y) - np.sum(X_squared) * np.sum(Y)) / (n * np.sum(X * 2) - (np.sum(X)) * 2)
c = (np.sum(Y) - a * np.sum(X ** 2) - b * np.sum(X)) / n


coef = np.polyfit(X, Y, 2)
modelo = np.poly1d(coef)

X_pred = np.linspace(0, 3000, 100)
Y_pred = modelo(X_pred)

#mse
for i in range(objetosY):
    mse += (Y[i] - Y_pred[i]) ** 2
mse = mse/objetosY

for i in range(len(Y)):
    sst += (Y[i] - mediaY) ** 2
    ssr += (Y[i] - Y_pred[i]) ** 2
r2 = 1 - (ssr / sst)


print("El mse es:" , mse)
print("R2 es igual a", r2)

# Graficar los datos originales y la curva de regresión polinomial
plt.scatter(X, Y, label='Datos reales')
plt.plot(X_pred, Y_pred, color='red', label='Regresión polinomial (grado 2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
