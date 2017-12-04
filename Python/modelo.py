import os
import pandas as pd
import dplython as dp
import gc

# Fijamos el directorio de trabajo
os.chdir('D:/pescobar/otros/Kaggle/Mercari_Price_Suggestion/')

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
test = dp.mutate(test, price=float('nan'))
# Reordenamos las variables
test = test[train.columns.tolist()]
# Unimos los dos data frames en uno
df = train.append(test)
# Borramos los df originales
del test, train
gc.collect()

