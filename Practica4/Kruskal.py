import pandas as pd
from scipy import stats

df = pd.read_csv('limpio.csv')
'''
Aqui escogi las variables que quiero comparar, una numerica que es profit y una categorica que es region
pero cabe aclarar que se pueden cambiar por cualquier otro par de variable categoria/numerica del dataset
ya sean Sales y Category, Quantity y Sub-Category, etc.
En este caso elegí profit y region para saber si hay diferencias significativas entre las regiones y el profit
'''

variable_numerica = "Profit"
variable_categorica = "Region"

grupos_nombres = df[variable_categorica].unique()
grupos_datos = [df[df[variable_categorica] == g][variable_numerica].dropna() for g in grupos_nombres]

print(f'Comparando "{variable_numerica}" entre grupos de "{variable_categorica}"')
print(f'Número de grupos: {len(grupos_nombres)} \nGrupos: {list(grupos_nombres)}\n')

# Hacemos la prueba de normalidad  por grupo (Shapiro-Wilk)
print('Prueba de normalidad (Shapiro-Wilk):')
todos_normales = True
for nombre, datos in zip(grupos_nombres, grupos_datos):
    stat, p = stats.shapiro(datos.sample(min(len(datos), 5000), random_state=42))
    print(f'  {nombre}: p-valor = {p:.4f} - {"Es normal" if p > 0.05 else "Es no normal"}')
    if p <= 0.05:
        todos_normales = False

'''
ANOVA y la prueba t son pruebas que necesitan que los datos de cada grupo sigan una distribución normal,
porque su p-valor necesita directamente que la distibucion sea normal para dar un resultado confiabel.
Como Shapiro-Wilk detectó que al menos un grupo NO es normal, estuve checando los pares con la prueba 
uno por uno y en todos los casos no hay distribucion normal, que tiene sentido porque en las graficas 
de la practica 3 los datos se inclinaban mucho a ciertos valores, al parecer es porque las ventas reales
siguen una distribucion sesgada a la derecha, pero como no hay distribucion normal, no se puede usar ANOVA
ni prueba t, por eso solo hice la prueba de kruskal que no deopende de una distribucion normal
'''

stat, p = stats.kruskal(*grupos_datos)
print(f'\nKruskal-Wallis: estadístico H = {stat:.4f}, p-valor = {p:.4f}')

#Interpretación
if p < 0.05:
    print(f'\nConclusión: p-valor ({p:.4f}) < 0.05 \nEsto quiere que decir SÍ hay diferencias significativas entre los grupos de {variable_categorica}.')
else:
    print(f'\nConclusión: p-valor ({p:.4f}) >= 0.05 \nEsto quiere decir que NO hay evidencia suficiente de diferencias significativas.')

'''
En este caso la prueba de Kruskal-Willis con el par es de Region/Profit nos indica que si hay una diferencia 
significativa entre las cuatro regiones, o dicho dde ora forma, la region en la que se hace un peido tiene relacion
con las ganacias generadas
'''