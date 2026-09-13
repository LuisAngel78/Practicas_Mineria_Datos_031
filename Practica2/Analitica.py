import pandas as pd

df = pd.read_csv('limpio.csv')
grupos = pd.DataFrame(df.columns, columns=['Nombres_Columnas'])

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Tiempo_Envio"] = (df["Ship Date"] - df["Order Date"]).dt.days

Frecuencias = df["Region"].value_counts().rename('Frecuencia Absoluta').to_frame()
Frecuencias['Frecuencia Relativa'] = df["Region"].value_counts(normalize=True)
print(Frecuencias.round(2))

print('Estadistica Descriptiva \nFechas:')
print(f'El rango  de fechas en las que se hicieron pedidos es de {df["Order Date"].min().strftime("%Y-%m-%d")} a {df["Order Date"].max().strftime("%Y-%m-%d")}')
print(f'El rango  de fechas en las que se entregaron pedidos es de {(df["Ship Date"].min().strftime("%Y-%m-%d"))} a {(df["Ship Date"].max().strftime("%Y-%m-%d"))}')
print(f'Número de categorías en Category: {df["Category"].nunique()}')
print(f'Estadística descriptiva:\n{df[["Sales", "Profit", "Profit_Margin","Tiempo_Envio"]].describe().round(2)}')
print(f'Mediana de "Sales": {round(df["Sales"].median(), 2)}')
print(f'Desviación estándar de "Profit": {round(df["Profit"].std(), 2)}')

print(f'Correlación entre variables:\n{df[["Sales", "Profit", "Profit_Margin"]].corr().round(2)}')

print('\nMétricas de datos agrupados')

print('\nVentas y ganancia por Region:')
print(df.groupby("Region").agg(
    Total_Ventas=("Sales", "sum"),
    Promedio_Ventas=("Sales", "mean"),
    Total_Ganancia=("Profit", "sum"),
    Promedio_Ganancia=("Profit", "mean"),
    Promedio_Margen=("Profit_Margin", "mean"),
    N_Pedidos=("Order ID", "nunique")
).round(2))

print('\nVentas y ganancia por Category:')
print(df.groupby("Category").agg(
    Total_Ventas=("Sales", "sum"),
    Promedio_Ventas=("Sales", "mean"),
    Total_Ganancia=("Profit", "sum"),
    Promedio_Margen=("Profit_Margin", "mean"),
    N_Pedidos=("Order ID", "nunique")
).round(2))

print('\nVentas y ganancia por Sub-Category:')
print(df.groupby("Sub-Category").agg(
    Total_Ventas=("Sales", "sum"),
    Total_Ganancia=("Profit", "sum"),
    Promedio_Margen=("Profit_Margin", "mean")
).round(2).sort_values("Total_Ventas", ascending=False))

print('\nVentas y ganancia por Segment:')
print(df.groupby("Segment").agg(
    Total_Ventas=("Sales", "sum"),
    Total_Ganancia=("Profit", "sum"),
    N_Clientes=("Customer ID", "nunique")
).round(2))

print('\nGanancia promedio por Region y Category:')
print(df.groupby(["Region", "Category"])["Profit"].mean().round(2))

print('\nTiempo de envío promedio por Region:')
print(df.groupby("Region")["Tiempo_Envio"].mean().round(2))

print('\nVentas (Region x Category):')
print(pd.pivot_table(df, index="Region", columns="Category", values="Sales", aggfunc="sum").round(2))

print('\nProporción de pedidos con pérdida (Profit < 0) por Category:')
print(df.groupby("Category")["Profit"].apply(lambda x: (x < 0).mean()).round(2))