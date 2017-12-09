#------------------------------------------------------------------------------
# 0. SETUP 
#------------------------------------------------------------------------------
# Reiniciamos el nucleo
%reset -f

# Librerias
import os
import pandas as pd
import numpy as np
import gc
import pickle
import functools
import gc
import itertools

# Fijamos el directorio de trabajo
os.chdir('C:/_Proyectos/Challenges/Kaggle/Kaggle_Mercari_Price_Suggestion/')

#------------------------------------------------------------------------------
# 1. LECTURA Y FORMATEO INICIAL
#------------------------------------------------------------------------------
# Leemos las bases de datos
train = pd.read_csv('data/train.tsv', sep='\t')
test = pd.read_csv('data/test.tsv', sep='\t')

# Consolidamos la data en un solo data frame
# Cambiamos el nombre de las variables id
train.columns = ['id'] + train.columns.tolist()[1:len(train.columns.tolist())]
test.columns = ['id'] + test.columns.tolist()[1:len(test.columns.tolist())]
# Modificamos los id poniendole el data set al que pertenecen (si no estarian
# duplicados)
train['id'] = train['id'].map(str) + '-train'
test['id'] = test['id'].map(str) + '-test'
# Añadimos variables que identifiquen cada subset
train['subset'] = 'train'
test['subset'] = 'test'
# Añadimos la variable precio al test con valores nan
test['price'] = np.nan
# Reordenamos las variables
test = test[train.columns.tolist()]
# Unimos los dos data frames en uno
df = train.append(test)
# Reseteamos los indices
df = df.reset_index(drop=True)
# Borramos los df originales
del test, train; gc.collect()
# Reorganizamos el orden de las variables
orden = ['subset', 'id', 'name','item_description', 'brand_name',
         'category_name', 'item_condition_id', 'shipping', 'price']
df = df[orden]
del orden; gc.collect()

#------------------------------------------------------------------------------
# 2. EXPLORATORIO Y TRANSFORMACIONES INICIALES
#------------------------------------------------------------------------------

# 2.1. Vamos a asignar a cada registro la media y la desviacion tipica del
# precio para las combinaciones de las variables brand_name, category_name,
# item_condition_id y shipping
# Primero transformamos en string las variables item_condition_id y shipping
df['item_condition_id'] = df['item_condition_id'].astype(str)
df['shipping'] = df['shipping'].astype(str)
# En la variable brand_name transformamos los nan float en nan string
df['brand_name'] = np.where(df['brand_name'].isnull(), 'nan', df['brand_name'])
# A partir de la variable category_name generamos variables por cada nivel de 
# categoria
categorias = df.category_name.str.split('/', expand=True)
categorias.fillna(value='nan', inplace=True)
categorias.columns = ['cat_lev_1', 'cat_lev_2', 'cat_lev_3',  'cat_lev_4',
                      'cat_lev_5']
# Unimos la info al data frame
df = pd.concat([df.iloc[:,0:5], categorias, df.iloc[:,6:]], axis=1)
# En cada una de las variables a trabajar en este punto si el valor no esta en 
# el train le asignamos nan en el test como string
for column in df.iloc[:,4:10]:
    # Duplicamos la variable
    df[column+'_ori'] = df[column]
    # Nos quedamos con los valores unicos del train
    unicos = df[df.subset=='train'][column].unique()
    # Reemplazamos los nan float por nan string (haciendolo para todo el data
    # frame, train y test, se corregiran los valores del test)
    df[column] = np.where(df[column].isin(unicos), df[column], 'nan')
# Reorganizamos las variables
orden = [list(range(0,4))+list(range(13,19))+list(range(4,13))]
df = df[df.columns.values[orden]]
# Borramos objetos sobrantes
del categorias, column, orden, unicos; gc.collect()



# Obtenemos las combinaciones posibles
variables = df.columns.values.tolist()[10:18]
combinaciones = []
for i in range(0,8):
    for tup  in itertools.combinations(variables, i+1):
        combinaciones.append(list(tup))
# Calculamos el precio medio por cada una de las combinaciones de las variables









# categoria nivel i y marca
# c1
media_c1 = df[df.subset=='train'].groupby(['cat_lev_1'], as_index=False)['price'].mean()
media_c1.columns = ['cat_lev_1', 'price_media_c1']
ds_c1 = df[df.subset=='train'].groupby(['cat_lev_1'], as_index=True)['price'].std()
ds_c1 = ds_c1.reset_index(level=ds_c1.index.names)
ds_c1.columns = ['cat_lev_1', 'price_ds_c1']
dfs = [df, media_c1, ds_c1]
df = functools.reduce(lambda left,right: pd.merge(left,right,on='cat_lev_1'), dfs)
# Borramos objetos sobrantes
del media_c1, ds_c1, dfs; gc.collect()
# c2
media_c2 = df[df.subset=='train'].groupby(['cat_lev_2'], as_index=False)['price'].mean()
media_c2.columns = ['cat_lev_2', 'price_media_c2']
ds_c2 = df[df.subset=='train'].groupby(['cat_lev_2'], as_index=True)['price'].std()
ds_c2 = ds_c2.reset_index(level=ds_c2.index.names)
ds_c2.columns = ['cat_lev_2', 'price_ds_c2']
dfs = [df, media_c2, ds_c2]
df = functools.reduce(lambda left,right: pd.merge(left,right,on='cat_lev_2'), dfs)
# Borramos objetos sobrantes
del media_c2, ds_c2, dfs; gc.collect()
# c3
media_c3 = df[df.subset=='train'].groupby(['cat_lev_3'], as_index=False)['price'].mean()
media_c3.columns = ['cat_lev_3', 'price_media_c3']
ds_c3 = df[df.subset=='train'].groupby(['cat_lev_3'], as_index=True)['price'].std()
ds_c3 = ds_c3.reset_index(level=ds_c3.index.names)
ds_c3.columns = ['cat_lev_3', 'price_ds_c3']
dfs = [df, media_c3, ds_c3]
aux = functools.reduce(lambda left,right: pd.merge(left,right,on='cat_lev_3'), dfs)
# Borramos objetos sobrantes
del media_c3, ds_c3, dfs; gc.collect()
# c4
media_c4 = df[df.subset=='train'].groupby(['cat_lev_4'], as_index=False)['price'].mean()
media_c4.columns = ['cat_lev_4', 'price_media_c4']
ds_c4 = df[df.subset=='train'].groupby(['cat_lev_4'], as_index=True)['price'].std()
ds_c4 = ds_c4.reset_index(level=ds_c4.index.names)
ds_c4.columns = ['cat_lev_4', 'price_ds_c4']
dfs = [df, media_c4, ds_c4]
df = functools.reduce(lambda left,right: pd.merge(left,right,on='cat_lev_4'), dfs)
# Borramos objetos sobrantes
del media_c4, ds_c4, dfs; gc.collect()
# c5
media_c5 = df[df.subset=='train'].groupby(['cat_lev_5'], as_index=False)['price'].mean()
media_c5.columns = ['cat_lev_5', 'price_media_c5']
ds_c5 = df[df.subset=='train'].groupby(['cat_lev_5'], as_index=True)['price'].std()
ds_c5 = ds_c5.reset_index(level=ds_c5.index.names)
ds_c5.columns = ['cat_lev_5', 'price_ds_c5']
dfs = [df, media_c5, ds_c5]
df = functools.reduce(lambda left,right: pd.merge(left,right,on='cat_lev_5'), dfs)
# Borramos objetos sobrantes
del media_c5, ds_c5, dfs; gc.collect()
# brand
media_b = df[df.subset=='train'].groupby(['brand_name'], as_index=False)['price'].mean()
media_b.columns = ['brand_name', 'media_b']
ds_b = df[df.subset=='train'].groupby(['brand_name'], as_index=True)['price'].std()
ds_b = ds_b.reset_index(level=ds_b.index.names)
ds_b.columns = ['brand_name', 'price_ds_b']
dfs = [df, media_b, ds_b]
df = functools.reduce(lambda left,right: pd.merge(left,right,on='brand_name'), dfs)
# Borramos objetos sobrantes
del media_b, ds_b, dfs; gc.collect()
# c1 c2
media_c1c2 = df[df.subset=='train'].groupby(['cat_lev_1', 'cat_lev_2'], as_index=False)['price'].mean()
media_c1c2.columns = ['cat_lev_1', 'cat_lev_2', 'price_media_c1c2']
ds_c1c2 = df[df.subset=='train'].groupby(['cat_lev_1', 'cat_lev_2'], as_index=True)['price'].std()
ds_c1c2 = ds_c1c2.reset_index(level=ds_c1c2.index.names)
ds_c1c2.columns = ['cat_lev_1', 'cat_lev_2', 'price_ds_c1']
dfs = [df, media_c1c2, ds_c1c2]
df = functools.reduce(lambda left,right: pd.merge(left,right,on=['cat_lev_1','cat_lev_2']), dfs)
# Borramos objetos sobrantes
del media_c1c2, ds_c1c2, dfs; gc.collect()
# c1 c2 c3
media_c1c2c3 = df[df.subset=='train'].groupby(['cat_lev_1', 'cat_lev_2', 'cat_lev_3'], as_index=False)['price'].mean()
media_c1c2c3.columns = ['cat_lev_1', 'cat_lev_2', 'cat_lev_3', 'price_media_c1c2']
ds_c1c2c3 = df[df.subset=='train'].groupby(['cat_lev_1', 'cat_lev_2', 'cat_lev_3'], as_index=True)['price'].std()
ds_c1c2c3 = ds_c1c2c3.reset_index(level=ds_c1c2c3.index.names)
ds_c1c2c3.columns = ['cat_lev_1', 'cat_lev_2', 'cat_lev_3', 'price_ds_c1']
dfs = [df, media_c1c2c3, ds_c1c2c3]
df = functools.reduce(lambda left,right: pd.merge(left,right,on=['cat_lev_1','cat_lev_2', 'cat_lev_3']), dfs)
# Borramos objetos sobrantes
del media_c1c2c3, ds_c1c2c3, dfs; gc.collect()










# 2. brand_name
# Obtenemos las frecuencias absolutas y relativas
bn_abs = df[df.subset=='train'].brand_name.value_counts()
bn_rel = round(bn_abs/df[df.subset=='train'].shape[0]*100,4)
# Extraemos los valores unicos
brand_name = df.brand_name.unique().tolist()

# Obtenemos las frecuencias relativas para el par de variables cat_lev_1 y
# brand_name
bc1 = round(pd.crosstab(index=df[df.subset=='train'].brand_name,
                  columns=df[df.subset=='train'].cat_lev_1,
                  dropna=False, normalize='index')*100,4)








# Vamos a usar Google Trends para sacar valores de popularidad para las marcas
# Guardamos el objeto brand_name
with open('brand_name.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(brand_name, f)
# Trabajamos la logica en GT_brand_name.py





