import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# Cargamos los datos
df = pd.read_csv('limpio.csv')

# El unico campo de texto libre es Product Name, las demas columnas son categorias y subcategorias
print('Filas:', len(df))
print("Cantidad de productos distintos:", df['Product Name'].nunique())

'''
Aqui vamos a limpiar el nombre de los productos, esto aunque parece que debi haberlo hecho en la practica 1 realmente
está bien asi porque lo que hará esta parte es limpiar los nombres, no descartar productos mal escritos, de cualquier
forma igual usamos stopwords para quitar las palabras que se repiten mucho que no aportan informacion
'''

palabras = []
for nombre in df['Product Name']:
    nombre = re.sub(r'[^a-z\s]', ' ', nombre.lower())
    limpias = []
    for p in nombre.split():
        if len(p) > 2 and p not in STOPWORDS:
            limpias.append(p)
    palabras.append(limpias)

df['palabras'] = palabras
df['num_palabras'] = df['Product Name'].str.split().str.len()

todas = []
for lista in df['palabras']:
    for p in lista:
        todas.append(p)
frecuencia = Counter(todas)

print(f'\nPalabras despues de limpiar: {len(todas)}')
print(f'Palabras distintas: {len(frecuencia)}')
print(f"Palabras por nombre: promedio {df['num_palabras'].mean():.1f}, mediana {df['num_palabras'].median():.0f}")

print('\n20 palabras mas frecuentes:')
for palabra, n in frecuencia.most_common(20):
    print(f'{palabra}: {n}')

# Aqui hacemos la nube de palabras completa sin distinguir entre categorias diferentes
nube = WordCloud(width=1000, height=500, background_color='white', max_words=100,
                 colormap='viridis').generate_from_frequencies(frecuencia)
plt.figure(figsize=(12, 6))
plt.imshow(nube, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras de los nombres de los productos')
plt.tight_layout()
plt.savefig('Texto_nube.png')
plt.show()

'''
Aqui obtenemos el top 15 de las palabras mas frecuentes para mostrarlas en una grafica de barras y tambien sacamos la 
distribucion de cuantas palabras tiene cada nombre de producto para ponerlo en un histograma
'''

top = frecuencia.most_common(15)
# aqui invertimos top para que la palabra más frecuente quede hasta arriba en la grafica de barras
top.reverse()
palabras_top = []
cantidades_top = []
for palabra, n in top:
    palabras_top.append(palabra)
    cantidades_top.append(n)
maximo = df['num_palabras'].max()

# Aqui hacemos la grafica de barras y el histograma
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
ejes[0].barh(palabras_top, cantidades_top, color='steelblue')
ejes[0].set_title('15 palabras mas frecuentes')
ejes[0].set_xlabel('Apariciones')
ejes[1].hist(df['num_palabras'], bins=range(1, maximo + 2), color='steelblue', edgecolor='white')
ejes[1].set_title('Palabras por nombre de producto')
ejes[1].set_xlabel('Numero de palabras')
ejes[1].set_ylabel('Productos')
plt.tight_layout()
plt.savefig('Texto_frecuencias.png')
plt.show()

# Finalmente hacemos una nube por categoria para ver que palabras cambian entre los diferentes grupos
categorias = df['Category'].unique()
fig, ejes = plt.subplots(1, len(categorias), figsize=(6 * len(categorias), 5))
for eje, cat in zip(ejes, categorias):
    f = Counter()
    for lista in df.loc[df['Category'] == cat, 'palabras']:
        f.update(lista)
    nube = WordCloud(width=600, height=400, background_color='white', max_words=60).generate_from_frequencies(f)
    eje.imshow(nube, interpolation='bilinear')
    eje.axis('off')
    eje.set_title(cat)
plt.tight_layout()
plt.savefig('Texto_categorias.png')
plt.show()

'''
Conclusiones:
Los nombres de productos son textos cortos y repetitivos, de las 9800 filas solo hay 1849 productos distintos, lo que
más se repite en las ventas son las marcas como xerox (844 apariciones) y avery (550 apariciones) y entre los productos
los que más aparecen son binders, chair, file, ring y phone, además cada categoria tiene su propia lista de palabras donde
las marcas que mas destacan son distintas en cada categoria, de cualquier forma con textos tan cortos solamente podemos 
conocer superficialmente lo que se vende, para explicar por que se vende más, sus materiales o para que se usan haria 
falta tener reseñas o descripciones más detalladas de los productos
'''