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
from pytrends.request import TrendReq

# Cargamos la lista de marcas
with open('brand_name.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    brand_name = pickle.load(f)

# Eliminamos el valor nan
brand_name = brand_name[1:]
ref = brand_name[0]

pytrend = TrendReq()
kw_list = [ref, brand_name[1]]
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')