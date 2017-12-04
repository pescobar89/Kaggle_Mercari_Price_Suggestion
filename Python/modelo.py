#------------------------------------------------------------------------------
# 0. SETUP 
#------------------------------------------------------------------------------
# Librerias
import os
import pandas as pd
import numpy as np
import dplython as dp
import gc

# Fijamos el directorio de trabajo
os.chdir('D:/pescobar/otros/Kaggle/Kaggle_Mercari_Price_Suggestion/')

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
         'brand_name', 'category_name', 'name', 'item_description']
df = df[orden]
del orden

#------------------------------------------------------------------------------
# 2. TRANSFORMACIONES INICIALES DE VARIABLES
#------------------------------------------------------------------------------
# A partir de la variable category_name generamos variables por cada nivel de 
# categoria
categorias = df.category_name.str.split('/', expand=True)
categorias.fillna(value=np.nan, inplace=True)
