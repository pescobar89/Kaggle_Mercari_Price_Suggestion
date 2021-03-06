###############################################################################
###################### KAGGLE - MERCARY PRICE SUGGESTION ######################
###############################################################################

# Diferentes pruebas con las variables iniciales 

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
import re
import random
import math
import scipy

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# Fijamos el directorio de trabajo
os.chdir('C:/_Proyectos/Challenges/Kaggle/Kaggle_Mercari_Price_Suggestion/')


#------------------------------------------------------------------------------
# 1. LECTURA Y FORMATEO INICIAL
#------------------------------------------------------------------------------
# Leemos las bases de datos
train = pd.read_csv('data/train.tsv', sep='\t')
test = pd.read_csv('data/test.tsv', sep='\t')
# Partimos la data train en train validation
random.seed(0)
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
train['id'] = train['id'].map(str) + '-train'
validation['id'] = validation['id'].map(str) + '-validation'
test['id'] = test['id'].map(str) + '-test'
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
# 2. EXPLORATORIO Y TRANSFORMACIONES DE VARIABLES
#------------------------------------------------------------------------------
# Eliminamos los registros cuyo precio sea 0
df = df[df.price>0]
# Transformamos la variable dependiente usando el logaritmo
df['price'] = np.log(df['price'])
# Transformamos en string las variables item_condition_id y shipping
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
    # Nos quedamos con los valores unicos del train
    unicos = df[df.subset=='train'][column].unique()
    # Reemplazamos los nan float por nan string (haciendolo para todo el data
    # frame, train y test, se corregiran los valores del test)
    df[column] = np.where(df[column].isin(unicos), df[column], 'nan')
# Borramos objetos sobrantes
del categorias, column, unicos; gc.collect()

# Obtenemos las combinaciones posibles
variables = df.columns.values.tolist()[4:12]
combinaciones = []
for i in range(0,8):
    for tup  in itertools.combinations(variables, i+1):
        combinaciones.append(list(tup))
# Borramos objetos sobrantes
del i, tup, variables; gc.collect()
# Nos quedamos con las variables por separado
combinaciones = combinaciones[0:8]
# Calculamos estadisticos descriptivos del precio por cada una de las 
# combinaciones de las variables
# Revisar la combinacion 9
for i, comb in enumerate(combinaciones):
    print('Combinacion ' + str(i) + " - " + '/'.join(comb))
    media = df[df.subset=='train'].groupby(comb, as_index=False)['price'].mean()
    media.columns = comb + ['media-' + '-'.join(comb)]
    deciles = df[df.subset=='train'].groupby(comb, as_index=False)['price'].quantile([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    deciles.columns = ['D'+str(s) for s in list(range(0,10))] + ['D10']
    deciles.columns = [s+'-'+comb[0] for s in deciles.columns]
    moda = df[df.subset=='train'].groupby(comb, as_index=False)['price'].agg(pd.Series.mode)
    moda['moda_min'+'-'+comb[0]] = [np.min(s.tolist()) for s in moda['price']]
    moda['moda_mediana'+'-'+comb[0]] = [np.median(s.tolist()) for s in moda['price']]
    moda['moda_max'+'-'+comb[0]] = [np.max(s.tolist()) for s in moda['price']]
    moda['moda_n'+'-'+comb[0]] = [np.size(s.tolist()) for s in moda['price']]    
    dsvar = df[df.subset=='train'].groupby(comb, as_index=True)['price'].std()
    dsvar = dsvar.reset_index(level=dsvar.index.names)
    dsvar.columns = comb + ['ds-' + '-'.join(comb)]
    dsvar['var-'+comb[0]] = dsvar['ds-' + '-'.join(comb)]**2
    rango = pd.DataFrame(deciles[deciles.columns[10]] - deciles[deciles.columns[0]])
    rango.columns = ['rango-'+comb[0]]
    iqr = df[df.subset=='train'].groupby(comb, as_index=False)['price'].quantile([.25,.75])
    iqr['iqr-'+comb[0]] = iqr[0.75] - iqr[0.25]
    coefvar = pd.DataFrame(df[df.subset=='train'].groupby(comb, as_index=False)['price'].apply(scipy.stats.variation))
    coefvar.columns = ['coefvar-'+comb[0]]
    skewness = pd.DataFrame(df[df.subset=='train'].groupby(comb, as_index=False)['price'].skew())
    skewness.columns = ['skewness-'+comb[0]]
    kurtosis = pd.DataFrame(df[df.subset=='train'].groupby(comb, as_index=False)['price'].apply(pd.DataFrame.kurt))
    kurtosis.columns = ['kurtosis-'+comb[0]]
    stats = pd.concat([media,
                      deciles,
                      moda[moda.columns[2:7]],
                      dsvar[dsvar.columns[1:3]],
                      rango,
                      iqr[iqr.columns[2]],
                      coefvar,
                      skewness,
                      kurtosis],
                      axis=1)
    df = pd.merge(df, stats, on=comb, how='outer')
    print('Tamaño del data frame: ' + str(df.shape[0]))
# Borramos objetos sobrantes
del i, media, deciles, moda, dsvar, rango, iqr, coefvar, skewness, kurtosis, stats; gc.collect()

# Obtenemos frecuencias absolutas y relativas
# Para cada variable
freqs = []
variables = df.columns.tolist()[4:12]
for i, var in enumerate(variables):
    fabs = df[df.subset=='train'][var].value_counts(dropna=False)
    frel = round(fabs/(df[df.subset=='train'].shape[0])*100,4)
    freq = pd.concat([fabs, frel], axis=1)
    freq.columns = ['absoluta', 'relativa']
    freq = [var, freq]
    freqs.append(freq)
# Borramos objetos sobrantes
del i, var, fabs, frel, freq; gc.collect()

# Reestructuramos las variables categoricas con mas de 10 + 1 (nan) niveles
num_vars = 10
for i, var in enumerate(variables):
    freq = freqs[i][1]
    niveles = freq.index.tolist()
    niveles = [x for x in niveles if x not in 'nan']
    if len(niveles)>num_vars:
        niveles = niveles[0:num_vars]
        df[var] = np.where(df[var].isin(niveles + ['nan']), df[var], 'OTHERS')
# Borramos objetos sobrantes
del i, freq, niveles, var, variables, num_vars, freqs; gc.collect() 

# Generamos dummies para las variables cat_lev_1, item_condition_id y shipping
dummies = pd.get_dummies(df.iloc[:,4:12])
df = pd.concat([df, dummies], axis=1)
# Borramos objetos sobrantes
del dummies; gc.collect()

# Formateamos la data para el entrenamiento del xgboost
ncol = df.shape[1]+1
dtrain = xgb.DMatrix(df[df.subset=='train'].iloc[:,13:ncol], label=df[df.subset=='train']['price'], missing=float('nan'))
dvalidation = xgb.DMatrix(df[df.subset=='validation'].iloc[:,13:ncol], label=df[df.subset=='validation']['price'], missing=float('nan'))
dtest = xgb.DMatrix(df[df.subset=='test'].iloc[:,13:ncol], label=df[df.subset=='test']['price'], missing=float('nan'))


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
        'eta': 0.1,
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
num_round = 100
bst = xgb.train(param, dtrain, num_round, [(dtrain, 'train'), (dvalidation, 'validation')])


#------------------------------------------------------------------------------
# 4. VALIDACION
#------------------------------------------------------------------------------
valpred = bst.predict(dvalidation)
valpred = np.where(valpred<=0,.01,valpred)
valreal = df[df.subset=='validation']['price'].values
rmsle(valreal, valpred)

#------------------------------------------------------------------------------
# 4. PREDICCION
#------------------------------------------------------------------------------
# Obtenemos las predicciones para la data test
pred = bst.predict(dtest)
# Unimos las predicciones a los ids
ids = df[df.subset=='test']['id'].reset_index(drop=True)
pred = pd.concat([ids, pd.Series(pred)], axis=1)
pred.columns = ['test_id', 'price']
pred['price'] = np.where(pred['price']<=0,.01,pred['price'])
pred['test_id'] = pred['test_id'].str.replace('-test','')
# Guardamos el los resultados en un csv
pred.to_csv('results/results_model_easy_tests_2.csv', index=False)