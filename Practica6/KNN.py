import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

'''
Aqui cargamos y preparamos los datos, tambien creamos la columna de tiempo de envio a partir de las
fechas para poder usarla como variable predictora en los modelos de KNN
'''

df = pd.read_csv('limpio.csv')

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Tiempo_Envio"] = (df["Ship Date"] - df["Order Date"]).dt.days

'''
Antes de hacer el modelo hay que decidir que variable vamos a clasificar. La idea inicial era clasificar si una venta
genera perdida (es decir que el proft fuera menor a 0) pero en este dataset las 9800 ventas son ganancia, por lo que 
solo habia una clase y no se podia entrenar nada. Despues probé clasificar si el margen era alto o no, pero la tabla 
de Category contra el nivel de margen mostro que office supplies siempre tiene margen alto y furniture y technology 
nunca, es decir que Category determina el margen por completo, y por eso el modelo daba un 100% con k = 1, que no es 
un patron descubierto por el modelo sino más bien una caracteristica del dataset.
Por todo esto la variable a clasificar va a ser Category, y para que no salga 100% otra vez no se usan ni Profit ni
Profit_Margin como predictores porque ambos dependen directamente de la categoria
'''

print('Distribución de clases:')
print(df["Category"].value_counts())
print(df["Category"].value_counts(normalize=True).round(4))

'''
Aqui vemos que las clases estan desbalanceadas, office supplies es el 60.3% de los datos, por lo que un modelo que
siempre diga office supplies ya acierta el 60% sin aprender nada, por eso mas adelante la accuracy sola no sirve para
evaluar y se usan tambien la exactitud balanceada, F1 macro y ROC-AUC
'''

print('\nSales por Category:')
print(df.groupby("Category")["Sales"].describe().round(2))

'''
Aqui podemos ver que  office supplies tiene una mediana de ventas de 27 mientras que furniture tiene 182 y technology 168, 
por lo que las ventas chicas son casi siempre office supplies, por el otro lado furniture y technology tienen una 
distribucion de ventas muy parecida entre si, entonces esperaria que al modelo le cueste separar esas dos
'''

'''
Aqui creamos una variable nueva a partir de la columna Sales la cual seria el logaritmo de Sales, porque Sales tiene una 
cola muy larga con ventas de hasta 22638 y como KNN usa distancias, si no la transformamos las ventas enormes dominarian 
la escala y las ventas chicas que son la mayoria quedarian casi en el mismo punto, y justo ahi es donde se separa
office supplies de las otras dos categorias. Con el logaritmo la diferencia entre una venta de 10 y una de 80 pesa
parecido a la diferencia entre una de 500 y una de 4000.
'''
df["log_Sales"] = np.log(df["Sales"])

datos = pd.concat([df[["log_Sales", "Tiempo_Envio"]],
                   pd.get_dummies(df[["Region", "Segment"]], drop_first=True, dtype=int),
                   df["Category"]], axis=1).dropna()

y = datos["Category"]

'''
Aqui hacemos el primer modelo de KNN con todas las variables disponibles y su objetivo es tener una base para comparar
con el modelo que viene despues a ver si mejora.
Se separa el 20% de los datos para prueba, con stratify para que las clases mantengan su proporcion en ambos conjuntos, 
y las variables se estandarizan porque KNN usa distancias.
'''

print('\nModelo KNN 1: Category ~ todas las variables')

X_m1 = datos.drop(columns="Category")
X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(
    X_m1, y, test_size=0.2, random_state=42, stratify=y
)

'''
La estandarizacion se hace dentro de un Pipeline para que en cada particion de la validacion cruzada el scaler se ajuste 
solo con los datos de entrenamiento y no se filtre informacion de validacion.
Despues elegimos el mejor k con validacion cruzada solo sobre entrenamiento, usando StratifiedKFold para que cada 
particion mantenga la proporcion de clases (office supplies cubre el 60%) y F1 macro porque promedia las 3 clases 
por igual y no deja que office supplies domine como pasaria con la accuracy
'''

f1_cv_m1 = []
for k in range(1, 51, 2):
    pipeline_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    scores = cross_val_score(pipeline_k, X_train_m1, y_train_m1,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")
    f1_cv_m1.append(scores.mean())

mejor_k_m1 = range(1, 51, 2)[f1_cv_m1.index(max(f1_cv_m1))]
print(f'Mejor k según validación cruzada: {mejor_k_m1} (F1 macro = {max(f1_cv_m1):.4f})')

modelo_knn_m1 = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=mejor_k_m1, weights="distance"))
])
modelo_knn_m1.fit(X_train_m1, y_train_m1)
y_pred_m1 = modelo_knn_m1.predict(X_test_m1)

print(f'Exactitud (accuracy): {accuracy_score(y_test_m1, y_pred_m1):.4f}')
print(f'Exactitud de referencia (siempre la clase mayoritaria): {y_test_m1.value_counts(normalize=True).max():.4f}')
print(f'Exactitud balanceada: {balanced_accuracy_score(y_test_m1, y_pred_m1):.4f} (azar = 0.3333)')
print(f'ROC-AUC: {roc_auc_score(y_test_m1, modelo_knn_m1.predict_proba(X_test_m1), multi_class="ovr", average="macro"):.4f} (azar = 0.5)')

'''
Aqui podemos ver que el modelo si "aprende" algo porque la exactitud balanceada (0.4467) y el ROC-AUC (0.6758) quedan
arriba del azar, sin embargo la accuracy (0.5847) queda por debajo de la referencia (0.6031), que no es un error
sino que el modelo se afina para repartir mejor entre las 3 clases y no para acertar mas en total. 
En una version anterior que puse separado el mes, el dia de la semana y el año el ROC-AUC era de 0.6189, asi que quitar 
esas variables ayudó y confirma que las fechas no aportan informacion.
'''

'''
Aqui hacemos el segundo modelo de KNN pero quitando las variables que al parecer no aportan y dejando solamente log_Sales
y Tiempo_Envio, para mejorar el modelo reduciendo ruido en la distancia. 
Se usa la misma division entrenamiento/prueba para que la comparacion con el modelo 1 sea pareja
'''

print('\nModelo KNN 2: Category ~ log_Sales + Tiempo_Envio')

X_m2 = datos[["log_Sales", "Tiempo_Envio"]]
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(
    X_m2, y, test_size=0.2, random_state=42, stratify=y
)

f1_cv_m2 = []
for k in range(1, 51, 2):
    pipeline_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    scores = cross_val_score(pipeline_k, X_train_m2, y_train_m2,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")
    f1_cv_m2.append(scores.mean())

mejor_k_m2 = range(1, 51, 2)[f1_cv_m2.index(max(f1_cv_m2))]
print(f'Mejor k según validación cruzada: {mejor_k_m2} (F1 macro = {max(f1_cv_m2):.4f})')

plt.figure(figsize=(8, 6))
plt.plot(list(range(1, 51, 2)), f1_cv_m2, marker="o", color="steelblue")
plt.axvline(x=mejor_k_m2, color="red", linestyle="--", label=f"Mejor k = {mejor_k_m2}")
plt.title('Modelo KNN 2: F1 macro (validación cruzada) según el valor de k')
plt.xlabel('k (número de vecinos)')
plt.ylabel('F1 macro')
plt.legend()
plt.tight_layout()
plt.savefig("knn_categoria_eleccion_k.png")
plt.show()

modelo_knn_m2 = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=mejor_k_m2, weights="distance"))
])
modelo_knn_m2.fit(X_train_m2, y_train_m2)
y_pred_m2 = modelo_knn_m2.predict(X_test_m2)

print(f'Exactitud (accuracy): {accuracy_score(y_test_m2, y_pred_m2):.4f}')
print(f'Exactitud de referencia (siempre la clase mayoritaria): {y_test_m2.value_counts(normalize=True).max():.4f}')
print(f'Exactitud balanceada: {balanced_accuracy_score(y_test_m2, y_pred_m2):.4f} (azar = 0.3333)')
print(f'ROC-AUC: {roc_auc_score(y_test_m2, modelo_knn_m2.predict_proba(X_test_m2), multi_class="ovr", average="macro"):.4f} (azar = 0.5)')
print('\nReporte de clasificación:')
print(classification_report(y_test_m2, y_pred_m2, zero_division=0))

'''
Aqui podemos ver que el modelo mejora en todas las metricas que nos importan que son el F1 macro que sube de 0.4367 a
0.5012, la exactitud balanceada de 0.4467 a 0.4721 y el ROC-AUC de 0.6758 a 0.7120. 
El mejor k tambien sube de 25 a 37, lo que tiene sentido porque con una señal debil y mucho traslape entre clases hace 
falta promediar mas vecinos para que la prediccion sea estable. 
La accuracy casi no cambia respecto a la referencia (0.6077 contra 0.6031) pero eso pasa porque el 60% de las ventas
son office supplies, asi que la mejora real se ve en las metricas balanceadas y no ahi
'''

matriz = confusion_matrix(y_test_m2, y_pred_m2, labels=modelo_knn_m2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=modelo_knn_m2.classes_)
disp.plot(cmap="Blues")
plt.title(f'Modelo KNN 2: Matriz de confusión - Category (k = {mejor_k_m2})')
plt.tight_layout()
plt.savefig("knn_categoria_matriz_confusion.png")
plt.show()

'''
En la matriz de confusion y en el reporte se ve que office supplies se reconoce bien (F1 de 0.76) porque sus ventas
son claramente mas bajas y por ende más reconocible como grupo, mientras que furniture y technology siguen siendo 
dificiles (F1 de 0.33 y 0.34) porque sus distribuciones de ventas se parecen demasiado y ninguna otra columna las separa
entre si como grupos
'''