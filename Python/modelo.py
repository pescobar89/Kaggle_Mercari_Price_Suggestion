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
orden = ['subset', 'id', 'price', 'item_condition_id', 'shipping',
         'category_name', 'brand_name', 'name', 'item_description']
df = df[orden]
del orden; gc.collect()

#------------------------------------------------------------------------------
# 2. EXPLORATORIO Y TRANSFORMACIONES INICIALES
#------------------------------------------------------------------------------

# 1. category_name
# A partir de la variable category_name generamos variables por cada nivel de 
# categoria
categorias = df.category_name.str.split('/', expand=True)
categorias.fillna(value='nan', inplace=True)
categorias.columns = ['cat_lev_1', 'cat_lev_2', 'cat_lev_3',
                      'cat_lev_4', 'cat_lev_5']
# Unimos la info al data frame
df = pd.concat([df.iloc[:,0:5], categorias, df.iloc[:,6:]], axis=1)
# Borramos objetos sobrantes
del categorias; gc.collect()
# Obtenemos frecuencias absolutas y relativas
# Para cada variable
n_df_train = df[df.subset=='train'].shape[0]
c1_abs = df[df.subset=='train'].cat_lev_1.value_counts(dropna=False)
c1_rel = round(c1_abs/n_df_train*100,4)
c1_ar = pd.concat([c1_abs, c1_rel], axis=1)
c2_abs = df[df.subset=='train'].cat_lev_2.value_counts(dropna=False)
c2_rel = round(c2_abs/n_df_train*100,4)
c2_ar = pd.concat([c2_abs, c2_rel], axis=1)
c3_abs = df[df.subset=='train'].cat_lev_3.value_counts(dropna=False)
c3_rel = round(c3_abs/n_df_train*100,4)
c3_ar = pd.concat([c3_abs, c3_rel], axis=1)
# Borramos objetos sobrantes
del c1_abs, c1_rel, c2_abs, c2_rel, c3_abs, c3_rel, n_df_train; gc.collect()
# Para el primer par de variables
c12_abs = pd.crosstab(index=df[df.subset=='train'].cat_lev_2,
                      columns=df[df.subset=='train'].cat_lev_1, dropna=False)
c12_abs = c12_abs.reset_index(level=c12_abs.index.names)
c12_ar = pd.melt(c12_abs, id_vars=c12_abs.columns.values[0],
                 value_vars=c12_abs.columns.tolist()[1:])
c12_rel_c1 = round(pd.crosstab(index=df[df.subset=='train'].cat_lev_2,
                               columns=df[df.subset=='train'].cat_lev_1, 
                               dropna=False, normalize='columns')*100,4)
c12_rel_c1 = c12_rel_c1.reset_index(level=c12_rel_c1.index.names)
aux = pd.melt(c12_rel_c1, id_vars=c12_rel_c1.columns.values[0],
                 value_vars=c12_rel_c1.columns.tolist()[1:])
c12_ar = pd.concat([c12_ar, aux.value], axis=1)
c12_rel = round(pd.crosstab(index=df[df.subset=='train'].cat_lev_2,
                            columns=df[df.subset=='train'].cat_lev_1,
                            dropna=False, normalize='all')*100,4)
c12_rel = c12_rel.reset_index(level=c12_rel.index.names)
aux = pd.melt(c12_rel, id_vars=c12_rel.columns.values[0],
              value_vars=c12_rel.columns.tolist()[1:])
c12_ar = pd.concat([c12_ar, aux.value], axis=1)
c12_ar.columns = c12_ar.columns.tolist()[0:2] + ['c12_abs', 'c12_rel_c1', 'c12_rel']
c12_ar = c12_ar.iloc[:,[1,0]+list(range(2,5))]
c12_ar = c12_ar[c12_ar.c12_abs>0]
# Borramos objetos sobrantes
del c12_abs, c12_rel, c12_rel_c1, aux; gc.collect()
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





