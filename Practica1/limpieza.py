import pandas as pd
from tabulate import tabulate

df = pd.read_csv("./datos/train.csv")
dfnulls = df[df.isnull().any(axis=1)]
print(f"empezamos con {len(dfnulls)} valores nulos y {df.duplicated().sum()} filas duplicadas")
'''
Descartaremos cieraas columnas que no tienen tanta relevancia para acortar la cantidad de columnas porque hay originalmente 18 columnas
pero las reduje a 12 necesarias para las practicas futuras
'''

limpio = df[['Order ID','Customer ID', 'Order Date', 'Ship Date', 'Region', 'Segment','Category','Sub-Category', 'Product Name', 'Sales']].copy()
#agregamos una columna de margen de ganancia para cada categoria de producto para las practicas futuras y tener variables numericas
limpio['Profit_Margin'] = limpio['Category'].map({
    'Office Supplies': 0.25,
    'Technology': 0.15,
    'Furniture': 0.08
})
limpio['Profit'] = (limpio['Sales'] * limpio['Profit_Margin']).round(2)

#convertimos las columnas de fecha a tipo datetime para poder hacer operaciones de tiempo en el futuro
limpio['Order Date'] = pd.to_datetime(limpio['Order Date'], dayfirst=True)
limpio['Ship Date'] = pd.to_datetime(limpio['Ship Date'], dayfirst=True)

# Seleccionamos todas las columnas de texto
string_cols = limpio.select_dtypes(include=['object', 'string']).columns

# limpiamos espacios en los extremo y convertimos todo a minúsculas para estandraizar el formato de los strings
for col in string_cols:
    limpio[col] = limpio[col].str.strip().str.lower()

#print(limpio.head())
dfnulls = limpio[limpio.isnull().any(axis=1)]

print(f"Terminamos con {len(dfnulls)} valores nulos después de la limpieza y todo estandarizado sin duplicados")
print("Total de filas duplicadas:", limpio.duplicated().sum())
print(limpio.head(5))

limpio.to_csv('limpio.csv', index=False)
print('el archivo limpio.csv se encunetra en la carpeta junto con los demás .py')