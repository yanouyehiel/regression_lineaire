import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Etape 1 : Generer un ensemble de donnees
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# Etape 2 : Diviser les donnees en ensembles d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Etape 3 : Creer et entrainer le modele de regression lineaire
model = LinearRegression()
model.fit(X_train, y_train)

# Etape 4 : Predire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Etape 5 : Evaluer le modele
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur quadratique moyenne : {mse:.2f}")
print(f"Score R^2 : {r2:.2f}")

#  Etape 6 : Visualiser les resultats
plt.scatter(X_test, y_test, color='blue', label='Donnees reelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predictions')
plt.title('Regression lineaire')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
