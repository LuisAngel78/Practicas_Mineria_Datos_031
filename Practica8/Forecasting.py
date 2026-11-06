import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
    
'''
Aqui cargamos los datos igual que en el modelo anterior pero ahora nos quedamos solo con los pedidos por mes, ya que
k means y en general el clustering no toma en cuenta el orden de los meses y no podia decirnos si las ventas crecen, 
por eso ahora usamos un modelo de series de tiempo para pronosticar los pedidos de los meses siguientes, cada fila del
dataset es un producto dentro de un pedido, asi que aqui pedidos se refiere a lineas de pedido y no a Order ID unicos
'''

df = pd.read_csv('limpio.csv')
df["Order Date"] = pd.to_datetime(df["Order Date"])

print('Modelo de pronostico: pedidos por mes')

serie = df.groupby(df["Order Date"].dt.to_period("M")).size()
serie.index = serie.index.to_timestamp()
serie = serie.asfreq("MS")

print(f'Meses en el historial: {len(serie)}')

anual = serie.groupby(serie.index.year).sum()
print('\nPedidos por año:')
print(anual)
print('\nCrecimiento contra el año anterior (%):')
print((anual.pct_change() * 100).round(1))

'''
Antes de modelar graficamos la serie, a la izquierda los pedidos en el tiempo para ver si hay tendencia y a la derecha
una linea por año contra el mes del calendario para ver si la estacionalidad se repite
'''

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
ejes[0].plot(serie.index, serie.values, marker="o", color="steelblue")
ejes[0].set_title('Pedidos por mes')
ejes[0].set_xlabel('Mes')
ejes[0].set_ylabel('Pedidos')
for año in anual.index:
    ejes[1].plot(range(1, 13), serie[serie.index.year == año].values, marker="o", label=str(año))
ejes[1].set_title('Pedidos por mes del calendario, una linea por año')
ejes[1].set_xlabel('Mes del calendario')
ejes[1].set_ylabel('Pedidos')
ejes[1].set_xticks(range(1, 13))
ejes[1].legend()
plt.tight_layout()
plt.savefig("Forecasting_datos.png")
plt.show()

'''
En esta practica vamos a hacer 3 modelos, uno va a ser la base para comparar con los otros 2 que son holt-winters
y SARIMA, el entrenamiento se hace con los primeros 3 años de informacion y la prueba con el ultimo año para ver que
tan bien predicen los modelos y cual considerar como el mejor para pronosticar los siguientes 12 meses
'''

entrenamiento = serie.iloc[:-12]
prueba = serie.iloc[-12:]

#el modelo base nada más repite los pedidos del mismo mes del año anterior y realmente no estamos entrenando nada
base = entrenamiento.iloc[-12:].values
'''
esto lo habia hecho para decidir si usar estacionalidad aditiva o multiplicativa y la aditiva resulto mejor con 
menor error
for tipo in ["add", "mul"]:
    m = ExponentialSmoothing(entrenamiento, trend="add", seasonal=tipo, seasonal_periods=12).fit()
    p = m.forecast(12)
    print(tipo, mean_absolute_error(prueba, p))
'''
modelo = ExponentialSmoothing(entrenamiento, trend="add", seasonal="add", seasonal_periods=12).fit()
holt = modelo.forecast(12).values

'''
Ahora aqui primero antes de entrar el modelo necesito ver cual es el mejor orden para sarima (pdq)(PDQM), en este 
caso como los datos son mensuales podemos poner directamente 12 en M porque es la estacionalidad, ahora otros dos elementos
que ya tenemos son d que es 1 porque podemos ver que cada año sube de manera sostenida y D es 1 porque podemos observar
que los picos en cada año tienen mas o menos la misma forma

Ahora tenemos que ver cuales son los mejores p,q y P,Q y para eso vamos a hacer combinaciones con los meses de entrenamiento
para encontrar el mejor orden para sarima porque hacer acf y pacf se volvió muy complicado y vamos a usar aic para ver cual
es el mejor modelo
'''

resultados = []
# Aqui pongo esto porque me salian demasiadas advertencias en la terminal y siento que ensucia el output
warnings.filterwarnings("ignore")
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    try:
        m = SARIMAX(entrenamiento, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
        resultados.append({'orden': (p, 1, q), 'estacional': (P, 1, Q, 12), 'aic': m.aic})
    except Exception:
        continue

tabla = pd.DataFrame(resultados).sort_values('aic').reset_index(drop=True)
print(tabla.head(10))

mejor = tabla.iloc[0]
print(f"\nMejor orden: {mejor['orden']}{mejor['estacional']}, AIC = {mejor['aic']:.2f}")

m = SARIMAX(entrenamiento, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)

'''
Aqui puse esto porque me di cuenta que habian advertencias porque habian modelos que no estaban convergiendo y esta linea
es una confirmacion para ver si el mejor modelo si convergió o no y afortunadamente si sale True
print(m.mle_retvals['converged'])

'''
'''
El mejor orden resultó ser (0,1,1)(0, 1, 0, 12) con el AIC mas bajo de 213.80 pero habrá que ver que tan bueno es
prediciendo los siguientes 12 meses para ver si nos quedamos con SARIMA o Holt-Winters
'''
modelo = SARIMAX(entrenamiento, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
sarima = modelo.forecast(12).values

print(f'\nMeses de entrenamiento: {len(entrenamiento)}, meses de prueba: {len(prueba)}')
print('\nError en los ultimos 12 meses (prueba):')
print(f'base: error promedio = {mean_absolute_error(prueba, base):.2f} pedidos, '
      f'error en porcentaje = {mean_absolute_percentage_error(prueba, base) * 100:.2f}%')
print(f'Holt-Winters: error promedio = {mean_absolute_error(prueba, holt):.2f} pedidos, '
      f'error en porcentaje = {mean_absolute_percentage_error(prueba, holt) * 100:.2f}%')
print(f'SARIMA: error promedio = {mean_absolute_error(prueba, sarima):.2f} pedidos, '
      f'error en porcentaje = {mean_absolute_percentage_error(prueba, sarima) * 100:.2f}%')

fig, ejes = plt.subplots(figsize=(12, 5))
ejes.plot(entrenamiento.index[-12:], entrenamiento.values[-12:], color="lightgray", label="Entrenamiento (ultimos 12 meses)")
ejes.plot(prueba.index, prueba.values, color="black", marker="o", label="Real")
ejes.plot(prueba.index, base, marker="o", linestyle="--", color="gray", label="Base")
ejes.plot(prueba.index, holt, marker="o", linestyle="--", color="darkorange", label="Holt-Winters")
ejes.plot(prueba.index, sarima, marker="o", linestyle="--", color="steelblue", label="SARIMA")
ejes.set_title('Pronostico contra datos reales en los ultimos 12 meses')
ejes.set_xlabel('Mes')
ejes.set_ylabel('Pedidos')
ejes.legend()
plt.tight_layout()
plt.savefig("Forecasting_prueba.png")
plt.show()

'''
Aqui podemos ver que los dos modelos le ganan al base (error promedio de 60.33 pedidos) y que SARIMA y Holt-Winters quedan
empatados, SARIMA con 31.21 pedidos y 10.52% contra 30.91 y 10.77% de Holt-Winters, por lo que no basta con repetir el año anterior ya que los
pedidos estan creciendo, asi que usamos SARIMA para el pronostico, volviendolo a entrenar con los 48 meses para
predecir los siguientes 12 que todavia no existen
'''

modelo = SARIMAX(serie, order=(0, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
futuro = modelo.forecast(12)

print('\nPronostico de pedidos para los proximos 12 meses:')
print(futuro.round(0).astype(int))

ultimo_año = serie.iloc[-12:].sum()
proximo_año = futuro.sum()
print(f'\nPedidos en los ultimos 12 meses del historial: {ultimo_año}')
print(f'Pedidos pronosticados para los proximos 12 meses: {proximo_año:.0f}')
print(f'Cambio: {(proximo_año / ultimo_año - 1) * 100:.1f}%')

modelo = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12).fit()
holt_año = modelo.forecast(12).sum()
print(f'\nPedidos pronosticados por Holt-Winters para los proximos 12 meses: {holt_año:.0f}')
print(f'Cambio con Holt-Winters: {(holt_año / ultimo_año - 1) * 100:.1f}%')

'''
Graficamos el pronostico junto al historial y los pedidos totales por año contra el total pronosticado
'''

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
ejes[0].plot(serie.index, serie.values, color="steelblue", marker="o", label="Historial")
ejes[0].plot(futuro.index, futuro.values, color="darkorange", marker="o", linestyle="--", label="Pronostico")
ejes[0].set_title('Pronostico de pedidos para los proximos 12 meses')
ejes[0].set_xlabel('Mes')
ejes[0].set_ylabel('Pedidos')
ejes[0].legend()
ejes[1].bar(anual.index.astype(str), anual.values, color="steelblue", label="Real")
ejes[1].bar(str(futuro.index[0].year), proximo_año, color="darkorange", label="Pronostico")
ejes[1].set_title('Pedidos totales por año')
ejes[1].set_xlabel('Año')
ejes[1].set_ylabel('Pedidos')
ejes[1].legend()
plt.tight_layout()
plt.savefig("Forecasting_futuro.png")
plt.show()

'''
Aqui podemos ver que el modelo pronostica 4476 pedidos para 2019, un 37.4% mas que los 3258 de 2018, la estacionalidad
se mantiene con enero y febrero como los meses mas bajos y septiembre, noviembre y diciembre como los mas altos, pero
ese crecimiento es mayor que el de cualquier año anterior siendo el maximo de 28.6% en 2018 y menor que los 4971 pedidos que
daba Holt-Winters
'''

'''
Conclusiones
Podemos decir que los pedidos tienen estacionalidad, como lo sugeria K means, y tambien tienen una tendencia de 
crecimiento que K means no podia captar, esta tendencia se distingue ya que pasaron de 1953 pedidos en 2015 a 3258 en 
2018, de los modelos entrenados SARIMA y Holt-Winters fueron los mejores en la prueba, empatados, con un error promedio de unos 31 pedidos por mes y 
SARIMA pronostica 4476 pedidos para 2019, aunque con solo 4 años de historial y una prueba de solo 12 meses la prediccion
realmente es una referencia y no un valor exacto, asi que teniendo en cuenta la prediccion de Holt-Winters de 
crecimiento de 52.6% junto con la de SARIMA de 37.4% podemos decir que el crecimiento esperado probablemente se ubique
entre el 37.4% y el 52.6%, aunque ambos extremos estan por encima del maximo historico de 28.6% y Holt-Winters es todavia
mas agresivo que SARIMA.
'''