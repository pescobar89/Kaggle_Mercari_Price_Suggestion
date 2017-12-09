# Codigo desechado

# Obtenemos frecuencias absolutas y relativas
# Para cada variable
n_df_train = df[df.subset=='train'].shape[0]
b_abs = df[df.subset=='train'].brand_name.value_counts(dropna=False)
b_rel = round(b_abs/n_df_train*100,4)
b_ar = pd.concat([b_abs, b_rel], axis=1)
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
del b_abs, b_rel, c1_abs, c1_rel, c2_abs, c2_rel, c3_abs, c3_rel, n_df_train
gc.collect()
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