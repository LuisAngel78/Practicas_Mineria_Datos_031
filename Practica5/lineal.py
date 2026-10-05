import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

'''
Ya como es costumbre aqui cargamos y preparamos los datos, tambien creo una columna de tiemmpo de envio
para usarla más adelante en un modelo lineal de regresion multiple por una estructura vista en la grafica
del primer modelo lineal simple
'''
df = pd.read_csv('limpio.csv')

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Tiempo_Envio"] = (df["Ship Date"] - df["Order Date"]).dt.days

'''
Aqui hacemos el primer modelo que es uno lineal simple de profit y sales, su objetivo en si es tener una
base para poder comparar los demás modelos que se van a hacer y ver si mejoran en la prediccion de datos
o no
'''

print('Modelo Lineal Simple (Base): Profit ~ Sales')

datos_simple = df[["Sales", "Profit"]].dropna()
X_simple = datos_simple[["Sales"]]
y_simple = datos_simple["Profit"]

modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y_simple)
y_pred_simple = modelo_simple.predict(X_simple)
r2_simple = r2_score(y_simple, y_pred_simple)

print(f'R^2 = {r2_simple:.4f}')

'''
Aqui podemos ver en las graficas que se forma una V desde el origen intersectada por la linea de regresion que
cubre la mayoria de los datos, esto sumado a que tenemos un R^2 de 0.8830 quiere decir que si hay una buena relacion
entre el modelo lineal y los datos  tanto asi que el modelo puede predecir alrededor del 88.3% de todos los datos,
sin embargo que hay un margen que se escapa y aqui es donde entra la forma V y es que esto se explica con que Sales
no es la unica variable que afecta a profit si no que hay otra que no está siendo considerada, más sin embargo para
este modelo está bien ya que es el modelo base más simple posible para que sirva de comparacion con lo demás
'''

plt.figure(figsize=(8, 6))
sns.regplot(x="Sales", y="Profit", data=datos_simple, scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
plt.title(f'Modelo Base: Profit vs Sales\nR^2 = {r2_simple:.4f}')
plt.tight_layout()
plt.savefig("modelo_base.png")
plt.show()

residuos_simple = y_simple - y_pred_simple
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_simple, y=residuos_simple, lowess=True, scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
plt.axhline(y=0, color="gray", linestyle="--")
plt.title('Modelo Base: Residuos vs valores predichos')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.tight_layout()
plt.savefig("modelo_base_residuos.png")
plt.show()

'''
Aqui hacemos el segundo modelo lineal pero con la diferencia que este es multiple e incluimos tiempo de envio y 
profit margin además de sales para verificar si esto hace el modelo más preciso y si mejor R^2
'''

print('Modelo Lineal Multiple 1 - Profit ~ Sales + Profit_Margin + Tiempo_Envio')
 
variables_x_m1 = ["Sales", "Profit_Margin", "Tiempo_Envio"]
datos_multiple_m1 = df[variables_x_m1 + ["Profit"]].dropna()
X_multiple_m1 = datos_multiple_m1[variables_x_m1]
y_multiple_m1 = datos_multiple_m1["Profit"]
 
modelo_multiple_m1 = LinearRegression()
modelo_multiple_m1.fit(X_multiple_m1, y_multiple_m1)
y_pred_multiple_m1 = modelo_multiple_m1.predict(X_multiple_m1)
r2_multiple_m1 = r2_score(y_multiple_m1, y_pred_multiple_m1)

'''
Aqui podemos ver que el coeficiente de Profit_Margin es muy alto comparado con los otros dos lo que cual considero
extraño ya que varia poco y no deberia tener tanto impacto, sin embargo sospecho que es porque profit deriva de 
sales y profit_margin
'''

for var, coef in zip(variables_x_m1, modelo_multiple_m1.coef_):
    print(f'Coeficiente de {var}: {coef:.4f}')
print(f'R^2 = {r2_multiple_m1:.4f}')


residuos_multiple_m1 = y_multiple_m1 - y_pred_multiple_m1
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_multiple_m1, y=residuos_multiple_m1, lowess=True, scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
plt.axhline(y=0, color="gray", linestyle="--")
plt.title('Modelo Multiple 1: Residuos vs valores predichos')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.tight_layout()
plt.savefig("modelo_multiple_residuos.png")
plt.show()

'''
Aqui vamos a hacer un modelo de diagnostico para evaluar si el modelo anterior tiene un R^2 alto porque profit_margin
es parte del calculo para derivar profit, es decir que el modelo intenta probar que profit_margin es componente en
la derivacion de profit, si es cierto debe haber un R^2 ~ 1.0 
'''
print('Modelo de diagnostico: Sales x Profit_Margin')

variables_x_diagnostico = ["Sales", "Profit_Margin", "Tiempo_Envio"]
datos_interaccion = df[variables_x_diagnostico + ["Profit"]].dropna().copy()
datos_interaccion["Sales_x_Margin"] = datos_interaccion["Sales"] * datos_interaccion["Profit_Margin"]

variables_interaccion = variables_x_diagnostico + ["Sales_x_Margin"]
X_interaccion = datos_interaccion[variables_interaccion]
y_interaccion = datos_interaccion["Profit"]

modelo_interaccion = LinearRegression()
modelo_interaccion.fit(X_interaccion, y_interaccion)
y_pred_interaccion = modelo_interaccion.predict(X_interaccion)
r2_interaccion = r2_score(y_interaccion, y_pred_interaccion)

'''
Efectivamente Profit = Sales * Profit_Margin por lo que el modelo anterior no nos sirve para predecir ya que es más
una formula
'''

for var, coef in zip(variables_interaccion, modelo_interaccion.coef_):
    print(f'Coeficiente de {var}: {coef:.4f}')
print(f'R^2 = {r2_interaccion:.4f}')

residuos_interaccion = y_interaccion - y_pred_interaccion
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_interaccion, y=residuos_interaccion, lowess=True, scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
plt.axhline(y=0, color="gray", linestyle="--")
plt.title('Modelo de diagnóstico: Residuos vs valores predichos')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.tight_layout()
plt.savefig("modelo_diagnostico_residuos.png")
plt.show()

'''
Aqui volvemos a hacer el modelo multiple pero quitando profit_margin para ver si hay una mejora en R^2 y ver cuales
variables afectan realmente al profit o a las ganancias de la tienda
'''

print('Modelo Multiple 2: Profit ~ Sales + Tiempo_Envio + Category + Region')

dummies = pd.get_dummies(df[["Category", "Region"]], drop_first=True)
datos_honesto = pd.concat([df[["Sales", "Tiempo_Envio", "Profit"]], dummies], axis=1).dropna()

variables_m2 = [c for c in datos_honesto.columns if c != "Profit"]
X_m2 = datos_honesto[variables_m2]
y_m2 = datos_honesto["Profit"]

modelo_m2 = LinearRegression()
modelo_m2.fit(X_m2, y_m2)
y_pred_m2 = modelo_m2.predict(X_m2)
r2_m2 = r2_score(y_m2, y_pred_m2)

'''
Aqui podemos ver que aumenta el R^2 a 0.9047 lo que efectivamente nos dice que además de que el modelo es más preciso
que las categorias y regiones afectan al profit
'''
for var, coef in zip(variables_m2, modelo_m2.coef_):
    print(f'Coeficiente de {var}: {coef:.4f}')
print(f'R^2 = {r2_m2:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_m2, y_pred_m2, alpha=0.4, s=10, color="steelblue")
plt.plot([y_m2.min(), y_m2.max()], [y_m2.min(), y_m2.max()],
         color="red", linestyle="--", linewidth=2, label="Predicción perfecta")
plt.title(f'Modelo Multiple 2: Profit real vs predicho\nR^2 = {r2_m2:.4f}')
plt.xlabel('Profit real')
plt.ylabel('Profit predicho')
plt.legend()
plt.tight_layout()
plt.savefig("modelo_multiple2.png")
plt.show()

residuos_m2 = y_m2 - y_pred_m2
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_m2, y=residuos_m2, lowess=True, scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
plt.axhline(y=0, color="gray", linestyle="--")
plt.title('Modelo Multiple 2: Residuos vs valores predichos')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.tight_layout()
plt.savefig("modelo_multiple2_residuos.png")
plt.show()

'''
Conclusiones:
Como era de esperarse las ventas explican gran parte de la ganancia, sin embargo hay más factores que influencian en
la ganancia final, tales como el tiempo de envio, la categoria o la region, con esta informacion podemos priorizar
nuestros analisis de informacion para tomar mejores decisiones financieras como hacer recortes de subcategorias o
productos asi como priorizar algunos en ciertas regiones con descuentos u ofertas más atractivas (lo haré en el pia
en esta seccion porque se sale del proposito de la practica) y con los modelos ya entrenados se puede estimar la 
rentabilidad de nuevos pedidos, auditorias de ventas mediante deteccion de anomalias, entre muchas otras cosas
'''