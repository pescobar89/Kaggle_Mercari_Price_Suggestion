###############################################################################
###################### KAGGLE - MERCARY PRICE SUGGESTION ######################
###############################################################################


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
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Fijamos el directorio de trabajo
os.chdir('C:/_Proyectos/Challenges/Kaggle/Kaggle_Mercari_Price_Suggestion/')


#------------------------------------------------------------------------------
# 1. LECTURA Y FORMATEO INICIAL
#------------------------------------------------------------------------------
# Leemos las bases de datos
train = pd.read_csv('data/train.tsv', sep='\t')
test = pd.read_csv('data/test.tsv', sep='\t')
# Partimos la data train en train validation
idx = np.random.rand(len(train)) < 0.7
validation = train[~idx]
train = train[idx]

# Consolidamos la data en un solo data frame
# Cambiamos el nombre de las variables id
train.columns = ['id'] + train.columns.tolist()[1:len(train.columns.tolist())]
validation.columns = ['id'] + validation.columns.tolist()[1:len(validation.columns.tolist())]
test.columns = ['id'] + test.columns.tolist()[1:len(test.columns.tolist())]
# Modificamos los id poniendole el data set al que pertenecen (si no estarian
# duplicados)
#train['id'] = train['id'].map(str) + '-train'
#validation['id'] = validation['id'].map(str) + '-validation'
#test['id'] = test['id'].map(str) + '-test'
# Añadimos variables que identifiquen cada subset
train['subset'] = 'train'
validation['subset'] = 'validation'
test['subset'] = 'test'
# Añadimos la variable precio al test con valores nan
test['price'] = np.nan
# Reordenamos las variables
test = test[train.columns.tolist()]
# Unimos los dos data frames en uno
df = train.append(validation)
df = df.append(test)
# Reseteamos los indices
df = df.reset_index(drop=True)
# Borramos los df originales
del train, validation, test, idx; gc.collect()
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
# Borramos objetos sobrantes
del i, tup, variables; gc.collect()
# Calculamos la media y la desviacion tipica del precio por cada una de las 
# combinaciones de las variables
for i, comb in enumerate(combinaciones):
    print('Combinacion ' + str(i) + " - " + '/'.join(comb))
    media = df[df.subset=='train'].groupby(comb, as_index=False)['price'].mean()
    media.columns = comb + ['media-' + '-'.join(comb)]
    ds = df[df.subset=='train'].groupby(comb, as_index=True)['price'].std()
    ds = ds.reset_index(level=ds.index.names)
    ds.columns = comb + ['ds-' + '-'.join(comb)]
    dfs = [df, media, ds]
    df = functools.reduce(lambda left,right: pd.merge(left,right,on=comb), dfs)
    print('Tamaño del data frame: ' + str(df.shape[0]))
# Borramos objetos sobrantes
del comb, combinaciones, dfs, ds, i, media; gc.collect()

# Guardamos df en un csv
#df.to_csv('pkl/df_medias_dss.csv', index=False)

# Formateamos la data para el entrenamiento del xgboost
dtrain = xgb.DMatrix(df[df.subset=='train'].iloc[:,19:90], label=df[df.subset=='train']['price'])
dvalidation = xgb.DMatrix(df[df.subset=='validation'].iloc[:,19:90], label=df[df.subset=='validation']['price'])
dtest = xgb.DMatrix(df[df.subset=='test'].iloc[:,19:90], label=df[df.subset=='test']['price'])
evallist = [(dtrain, 'train'), (dvalidation, 'validation')]


#------------------------------------------------------------------------------
# 3. MODELO XGBOOST
#------------------------------------------------------------------------------
# Fijamos los parametros
param = {
        # General Parameters
        'booster': 'gbtree',
        'silent': 0,
        'nthread': 7,
        # Parameters for Tree Booster
        'eta': 0.3,
        'gamma': 0,
        'max_depth': 6,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # Learning Task Parameters
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 0
        }
num_round = 37
bst = xgb.train(param, dtrain, num_round, evallist)


#------------------------------------------------------------------------------
# 4. VALIDACION
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# 4. PREDICCION
#------------------------------------------------------------------------------
# Obtenemos las predicciones para la data test
pred = bst.predict(dtest)
# Unimos las predicciones a los ids
ids = df[df.subset=='test']['id'].reset_index(drop=True)
pred = pd.concat([ids, pd.Series(pred)], axis=1)
pred.columns = ['test_id', 'price']
# Guardamos el los resultados en un csv
pred.to_csv('results/results_model_hard_1_dos.csv', index=False)




# 2. brand_name
# Obtenemos las frecuencias absolutas y relativas
#bn_abs = df[df.subset=='train'].brand_name.value_counts()
#bn_rel = round(bn_abs/df[df.subset=='train'].shape[0]*100,4)
# Extraemos los valores unicos
#brand_name = df.brand_name.unique().tolist()

# Obtenemos las frecuencias relativas para el par de variables cat_lev_1 y
# brand_name
#bc1 = round(pd.crosstab(index=df[df.subset=='train'].brand_name,
#                  columns=df[df.subset=='train'].cat_lev_1,
#                  dropna=False, normalize='index')*100,4)








# Vamos a usar Google Trends para sacar valores de popularidad para las marcas
# Guardamos el objeto brand_name
#with open('brand_name.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(brand_name, f)
# Trabajamos la logica en GT_brand_name.py





