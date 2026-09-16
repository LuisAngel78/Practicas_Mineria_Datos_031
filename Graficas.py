import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('limpio.csv')
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Tiempo_Envio"] = (df["Ship Date"] - df["Order Date"]).dt.days

col_numericas = ["Sales", "Profit", "Profit_Margin", "Tiempo_Envio"]
col_categoricas = ["Region", "Segment", "Category"]

# Histograma de profit, sales, tiempo de envio y profit_margin
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(col_numericas):
    axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="black")
    axes[i].set_title(f'Histograma de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig("histogramas.png")
plt.show()

# Diagrama de caja de sales, profit, tiempo_envio y profit_margin 
fig, axes = plt.subplots(1, len(col_numericas), figsize=(14, 5))

for i, col in enumerate(col_numericas):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(f'DiagCaja de {col}')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig("Cajas.png")
plt.show()

# Diagramas de pastel de region, segment y category
fig, axes = plt.subplots(1, len(col_categoricas), figsize=(15, 5))

for i, col in enumerate(col_categoricas):
    conteo = df[col].value_counts()
    axes[i].pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Distribución de {col}')

plt.tight_layout()
plt.savefig("pasteles.png")
plt.show()

# Diagramas de dispersion de pares de sales y profit margin
pares = []
for i in range(len(col_numericas)):
    for j in range(i + 1, len(col_numericas)):
        pares.append((col_numericas[i], col_numericas[j]))

n = len(pares)
cols_grid = 3
filas_grid = n // cols_grid if n % cols_grid == 0 else n // cols_grid + 1

fig, axes = plt.subplots(filas_grid, cols_grid, figsize=(15, 5 * filas_grid))
axes = axes.flatten()

for i, (x, y) in enumerate(pares):
    axes[i].scatter(df[x], df[y], alpha=0.4, s=10, color="darkorange")
    axes[i].set_title(f'{x} vs {y}')
    axes[i].set_xlabel(x)
    axes[i].set_ylabel(y)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("dispersion.png")
plt.show()

# Graficas de barras de ventas totales por categoria
fig, axes = plt.subplots(1, len(col_categoricas), figsize=(15, 5))

for i, col in enumerate(col_categoricas):
    ventas_por_grupo = df.groupby(col)["Sales"].sum().sort_values(ascending=False)
    axes[i].bar(ventas_por_grupo.index, ventas_por_grupo.values, color="seagreen")
    axes[i].set_title(f'Ventas totales por {col}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylabel('Ventas totales')

plt.tight_layout()
plt.savefig("barras.png")
plt.show()