#------------------------------------------------------------------------------
# 0. SETUP 
#------------------------------------------------------------------------------
# Reiniciamos el nucleo
%reset -f

# Librerias
import os
import pandas as pd
import numpy as np
import dplython as dp
import gc
import pickle

# Fijamos el directorio de trabajo
os.chdir('D:/pescobar/otros/Kaggle/Kaggle_Mercari_Price_Suggestion/')

#------------------------------------------------------------------------------
# 1. LECTURA Y FORMATEO INICIAL
#------------------------------------------------------------------------------
# Leemos las bases de datos
train = dp.DplyFrame(pd.read_csv('data/train.tsv', sep='\t'))
test = dp.DplyFrame(pd.read_csv('data/test.tsv', sep='\t'))

# Consolidamos la data en un solo data frame
# Cambiamos el nombre de las variables id
train.columns = ['id'] + train.columns.tolist()[1:len(train.columns.tolist())]
test.columns = ['id'] + test.columns.tolist()[1:len(test.columns.tolist())]
# Añadimos variables que identifiquen cada subset
train = dp.mutate(train, subset='train')
test = dp.mutate(test, subset='test')
# Añadimos la variable precio al test con valores nan
test = dp.mutate(test, price=np.nan)
# Reordenamos las variables
test = test[train.columns.tolist()]
# Unimos los dos data frames en uno
df = train.append(test)
# Reseteamos los indices
df = df.reset_index(drop=True)
# Borramos los df originales
del test, train
gc.collect()
# Reorganizamos el orden de las variables
orden = ['subset', 'id', 'price', 'item_condition_id', 'shipping',
         'category_name', 'brand_name', 'name', 'item_description']
df = df[orden]
del orden

#------------------------------------------------------------------------------
# 2. EXPLORATORIO Y TRANSFORMACIONES INICIALES
#------------------------------------------------------------------------------

# 1. category_name
# A partir de la variable category_name generamos variables por cada nivel de 
# categoria
categorias = df.category_name.str.split('/', expand=True)
categorias.fillna(value=np.nan, inplace=True)
categorias.columns = ['cat_level_1', 'cat_level_2', 'cat_level_3',
                      'cat_level_4', 'cat_level_5']
# Unimos la info al data frame
df = pd.concat([df.iloc[:,0:5], categorias, df.iloc[:,6:]], axis=1)
# Obtenemos frecuencias absolutas y relativas
# Para cada variable
n_df_train = df[df.subset=='train'].shape[0]
c1_abs = df[df.subset=='train'].cat_level_1.value_counts(dropna=False)
c1_rel = round(c1_abs/n_df_train*100,4)
c2_abs = df[df.subset=='train'].cat_level_2.value_counts(dropna=False)
c2_rel = round(c2_abs/n_df_train*100,4)
c3_abs = df[df.subset=='train'].cat_level_3.value_counts(dropna=False)
c3_rel = round(c3_abs/n_df_train*100,4)
# Para el primer par de variables
c12_abs = pd.crosstab(index=df[df.subset=='train'].cat_level_2,
                      columns=df[df.subset=='train'].cat_level_1, dropna=False)
c12_abs = c12_abs.reset_index(level=c12_abs.index.names)
c12_ar = pd.melt(c12_abs, id_vars=c12_abs.columns.values[0],
                 value_vars=c12_abs.columns.tolist()[1:])
c12_rel_c1 = round(pd.crosstab(index=df[df.subset=='train'].cat_level_2,
                               columns=df[df.subset=='train'].cat_level_1, 
                               dropna=False, normalize='columns')*100,4)
c12_rel_c1 = c12_rel_c1.reset_index(level=c12_rel_c1.index.names)
aux = pd.melt(c12_rel_c1, id_vars=c12_rel_c1.columns.values[0],
                 value_vars=c12_rel_c1.columns.tolist()[1:])
c12_ar = pd.concat([c12_ar, aux.value], axis=1)
c12_rel = round(pd.crosstab(index=df[df.subset=='train'].cat_level_2,
                            columns=df[df.subset=='train'].cat_level_1,
                            dropna=False, normalize='all')*100,4)
c12_rel = c12_rel.reset_index(level=c12_rel.index.names)
aux = pd.melt(c12_rel, id_vars=c12_rel.columns.values[0],
              value_vars=c12_rel.columns.tolist()[1:])
c12_ar = pd.concat([c12_ar, aux.value], axis=1)
c12_ar.columns = c12_ar.columns.tolist()[0:2] + ['c12_abs', 'c12_rel_c1', 'c12_rel']
c12_ar = c12_ar.iloc[:,[1,0]+list(range(2,5))]
c12_ar = c12_ar[c12_ar.c12_abs>0]


# 2. brand_name
# Obtenemos las frecuencias absolutas y relativas
bn_abs = df[df.subset=='train'].brand_name.value_counts()
bn_rel = round(bn_abs/df[df.subset=='train'].shape[0]*100,4)
# Extraemos los valores unicos
brand_name = df.brand_name.unique().tolist()

# Obtenemos las frecuencias relativas para el par de variables cat_level_1 y
# brand_name
bc1 = round(pd.crosstab(index=df[df.subset=='train'].brand_name,
                  columns=df[df.subset=='train'].cat_level_1,
                  dropna=False, normalize='index')*100,4)








# Vamos a usar Google Trends para sacar valores de popularidad para las marcas
# Guardamos el objeto brand_name
with open('brand_name.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(brand_name, f)
# Trabajamos la logica en GT_brand_name.py





