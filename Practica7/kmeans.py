import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

'''
Aqui cargamos y preparamos los datos, tambien creamos la columna de tiempo de envio con las fechas para usarla como una
de las variables de cada mes
'''

df = pd.read_csv('limpio.csv')

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Tiempo_Envio"] = (df["Ship Date"] - df["Order Date"]).dt.days

'''
K-means es un algoritmo no supervisado que solo agrupa objetos en k grupos segun que tan parecidos son, esto lo hace 
usando distancias para ver que tan lejos estan las variables entre si, por ello solo sirven variables numericas y hay 
que estandarizarlas.
ya que en las ultimas practicas siento que los modelos daban conclusiones parecidas o que apuntaban hacia un mismo lugar
en este modelo queria agrupar por meses en vez de ventas individuales, para ver si hay meses que se comportan parecido 
y asi encontrar patrones en diferentes estaciones o tendencias, en este caso para el modelo vamos a describir cada mes 
con el numero  de pedidos, la venta mediana (en vez de promedio debido a la diferencia entre la cantidad de ventas) y 
el tiempo de envio promedio.
'''

print('Modelo K-means: meses (pedidos, venta_mediana, envio)')

mensual = df.groupby(df["Order Date"].dt.to_period("M")).agg(
    pedidos=("Sales", "count"),
    venta_mediana=("Sales", "median"),
    envio=("Tiempo_Envio", "mean"))

print(f'Meses en el historial: {len(mensual)}')
print('Primer y ultimo mes:')
print(mensual[["pedidos", "venta_mediana", "envio"]].iloc[[0, -1]].round(2))

'''
Antes de agrupar graficamos los datos de los meses sin clusters para ver si encontramos una estructura general, como 
cada mes tiene 3 variables y es mas facil ver 2 a la vez, vamos a hacer dos graficas una de los pedidos a lo largo del 
tiempo y una de los pedidos contra la venta mediana
'''

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
ejes[0].plot(mensual.index.to_timestamp(), mensual["pedidos"], marker="o", color="steelblue")
ejes[0].set_title('Pedidos por mes')
ejes[0].set_xlabel('Mes')
ejes[0].set_ylabel('Pedidos')
ejes[1].scatter(mensual["pedidos"], mensual["venta_mediana"], color="steelblue", s=60)
ejes[1].set_title('Pedidos contra venta mediana')
ejes[1].set_xlabel('Pedidos')
ejes[1].set_ylabel('Venta mediana')
plt.tight_layout()
plt.savefig("kmeans_meses_datos.png")
plt.show()

'''
Para elegir k probamos de 2 a 6 con la inercia usando el metodo del codo, que siempre baja cuando k sube y se busca
el punto donde el cambio no es tan significativo, y el coeficiente de silueta, que mide que tan separados estan los 
grupos, siendo un numero más grande mayor separacion y entre más pequeño más despreciable.
En este caso como cada mes es un punto solo probamos hasta k posible de 6 porque con tan pocos puntos un k mas alto 
dejaria clusters de muy pocos meses y eso es más dificil de interpretar
'''

escalador = StandardScaler()
X = escalador.fit_transform(mensual[["pedidos", "venta_mediana", "envio"]])

inercia = []
silueta = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inercia.append(km.inertia_)
    silueta.append(silhouette_score(X, km.labels_))
    print(f'k = {k}: inercia = {km.inertia_:.1f}, silueta = {silhouette_score(X, km.labels_):.4f}')

mejor_k = range(2, 7)[silueta.index(max(silueta))]
print(f'Mejor k según silueta: {mejor_k} (silueta = {max(silueta):.4f})')

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
ejes[0].plot(range(2, 7), inercia, marker="o", color="steelblue")
ejes[0].set_title('Método del codo')
ejes[0].set_xlabel('k clusters')
ejes[0].set_ylabel('Inercia')
ejes[1].plot(range(2, 7), silueta, marker="o", color="darkorange")
ejes[1].axvline(x=mejor_k, color="red", linestyle="--", label=f"Mejor k = {mejor_k}")
ejes[1].set_title('Coeficiente de silueta')
ejes[1].set_xlabel('k clusters')
ejes[1].set_ylabel('Silueta')
ejes[1].legend()
plt.tight_layout()
plt.savefig("kmeans_meses_eleccion_k.png")
plt.show()

modelo = KMeans(n_clusters=mejor_k, n_init=10, random_state=42).fit(X)
mensual["Cluster"] = modelo.labels_

'''
interpretando lo que nos dicen los grupos de meses, vemos cuantos meses tiene cada cluster y su perfil promedio y los
cruzamos con el mes del calendario (sin importar el año) para ver si un cluster junta los mismos meses en varios años,
lo que sugiere estacionalidad, por el otro lado si hacemos lo mismo pero cruzando con el año podemos ver si un cluster
se concentra en ciertos años, lo que sugiere una tendencia
'''

print('\nTamaño de cada cluster:')
print(mensual["Cluster"].value_counts().sort_index())

print('\nPerfil promedio de cada cluster:')
print(mensual.groupby("Cluster")[["pedidos", "venta_mediana", "envio"]].mean().round(2))

print('\nMeses del calendario en cada cluster:')
print(pd.crosstab(mensual["Cluster"], mensual.index.month))

print('\nAños en cada cluster:')
print(pd.crosstab(mensual["Cluster"], mensual.index.year))

'''
Aqui podemos ver que no hay una estructura muy definida ya que la silueta va de 0.26 a 0.32 para todos los k y la mas
alta es 0.3242 con k = 5, casi igual a la de k = 3 de 0.3219, asi que elegir k = 5 no es significativamente mejor y
ademas la inercia baja cada vez menos.
Aun asi los clusters se pueden interpretar por epoca del año, ya que el cluster 1 (12 meses, 350 pedidos en promedio)
tiene septiembre a diciembre con noviembre y diciembre en los 4 años, lo que indica una temporada alta al final del
año, el cluster 2 (14 meses, 176 pedidos) tiene casi solo meses de marzo a agosto, el cluster 0 (8 meses, 119 pedidos)
es el de menos pedidos y menor venta mediana con la mitad de sus meses en enero y febrero, el cluster 4 (11 meses) 
mezcla meses sin un patron claro y finalmente el cluster 3 son solo 3 meses con una venta mediana mas alta (83 contra
40 a 61 de los demas).
Al final tambien podemos ver que el tiempo de envio casi no cambia entre clusters (3.6 a 4.2 dias), por lo que los 
grupos se forman sobre todo por pedidos y venta mediana, y como casi todos los clusters aparecen en todos los años esta
tabla no muestra una tendencia clara de crecimiento
'''

'''
Aqui graficamos las mismas dos vistas del principio pero coloreadas por cluster, con los centroides en la grafica de 
pedidos contra venta mediana, es importante aclarar que como el modelo usa 3 variables y solo estamos graficando con 2
hay puntos que se superponen aunque formen parte de grupos diferentes y pareciera que deberian ser parte del 
mismo cluster cuando en realidad la diferencia esta en la tercer variable
'''

centros = escalador.inverse_transform(modelo.cluster_centers_)

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
ejes[0].plot(mensual.index.to_timestamp(), mensual["pedidos"], color="lightgray", zorder=1)
ejes[0].scatter(mensual.index.to_timestamp(), mensual["pedidos"], c=mensual["Cluster"], cmap="viridis", s=60, zorder=2)
ejes[0].set_title(f'Pedidos por mes según cluster (k = {mejor_k})')
ejes[0].set_xlabel('Mes')
ejes[0].set_ylabel('Pedidos')
ejes[1].scatter(mensual["pedidos"], mensual["venta_mediana"], c=mensual["Cluster"], cmap="viridis", s=60)
ejes[1].scatter(centros[:, 0], centros[:, 1], c="red", marker="X", s=200, label="Centroides")
ejes[1].set_title(f'Pedidos contra venta mediana según cluster (k = {mejor_k})')
ejes[1].set_xlabel('Pedidos')
ejes[1].set_ylabel('Venta mediana')
ejes[1].legend()
plt.tight_layout()
plt.savefig("kmeans_meses_clusters.png")
plt.show()

'''
Ya para concluir de forma más clara podemos decir que los meses no forman grupos bien separados y esto se puede ver
claramente en la silueta, sin embargo, si se observa estacionalidad, ya que septiembre a diciembre concentra los meses
de mas pedidos y los meses de menos pedidos se encuentran sobre todo a inicio de año, aunque la estacionalidad no es 
perfecta porque una parte de los meses no sigue un patron por epoca, es algo esperable. 
Por el lado de la tendencia podemos ver en la distribucion por años que no se observa nada claro, aunque esto no 
descarta que las ventas hayan crecido, ya que el modelo agrupa por parecido entre meses y no por su orden temporal.
'''