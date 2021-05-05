# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:15:37 2021

@author: ferra
"""

import pandas as pd
import numpy as np
import statsmodels as sm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# import plotly.graph_objects as go # Für andere Dartstellung (Paket erst installieren)
# from plotly.offline import plot #Für andere Darstellung (Paket erst installieren)
# %%  daten einladen
dfb = pd.read_csv("C:\\Users\D.Albrecht\Documents\Hochschule\Machine Learning\Beispieldatensatz_Beckhoff_SPS.csv",
                  sep=';',
                  low_memory=False
                  ).drop(columns='Unnamed: 54')

dfs = pd.read_csv("C:\\Users\D.Albrecht\Documents\Hochschule\Machine Learning\Beispieldatensatz_Siemens_S7_SPS.csv",
                  sep=';',
                  low_memory=False
                  ).drop(columns='Unnamed: 143')

# %%Entfernen von konstanten Daten
dfb = dfb.drop(dfb.columns[dfb.apply(lambda col: col.nunique() == 1)], axis=1)
dfb = dfb.drop(dfb.columns[dfb.apply(lambda col: col.nunique() == 0)], axis=1)

dfs = dfs.drop(dfs.columns[dfs.apply(lambda col: col.nunique() == 1)], axis=1)
dfs = dfs.drop(dfs.columns[dfs.apply(lambda col: col.nunique() == 0)], axis=1)

# Übrige "Not used" spalten entfernen
del dfs['I200.7']
del dfs['M0.1']
del dfs['M24.4']
del dfs['M24.7']
del dfs['M25.3']
del dfs['M25.4']
# %% True & False zu 1&0
dfb = dfb.applymap(lambda x: 1 if x == True else x)
dfb = dfb.applymap(lambda x: 0 if x == False else x)

dfs = dfs.applymap(lambda x: 1 if x == True else x)
dfs = dfs.applymap(lambda x: 0 if x == False else x)

# %% Merge über über Nearest Neighbour (Timestamp)
dfb['timestamp'] = pd.to_datetime(dfb.timestamp)
dfs['timestamp'] = pd.to_datetime(dfs.timestamp)
dfb.sort_values('timestamp', inplace=True)
dfs.sort_values('timestamp', inplace=True)

dfs_new = dfs.set_index('timestamp').reindex(dfb.set_index('timestamp').index, method='nearest').reset_index()
df = pd.merge(dfb, dfs_new, on='timestamp', suffixes=('_', ''))
df.set_index('timestamp', inplace=True)
dfs.set_index('timestamp', inplace=True)
dfb.set_index('timestamp', inplace=True)
# %% Auteilen in Stationsdatensätze
df_robot = pd.DataFrame(data=dfb,
                        columns=['VG.I1_ref_switch_vertical', 'VG.I2_ref_switch_horizontal',
                                 'VG.B1_encoder_vertical_impulse1', 'VG.B2_encoder_vertical_impulse2',
                                 'VG.B3_encoder_horizontal_impulse1',
                                 'VG.B4_encoder_horizontal_impulse2', 'VG.B5_encoder_rotate_impulse1',
                                 'VG.B6_encoder_rotate_impulse2', 'VG.safeCurtainShort', 'VG.Q1_verUp', 'VG.Q2_verDown',
                                 'VG.Q3_horBack', 'VG.Q4_horForw',
                                 'VG.Q5_rotClockwise', 'VG.Q6_rotCounterClockwise', 'VG.Q7_compressorON',
                                 'VG.Q8_valve'],
                        index=dfb.index)

df_warehouse = pd.DataFrame(data=dfb,
                            columns=['WH.I1_Ref_switch_horizontal', 'WH.I2_Light_barrier_inside',
                                     'WH.I3_Light_barrier_outside', 'WH.I5_Ref_switch_cantilever_front',
                                     'WH.I6_Ref_switch_cantilever_back',
                                     'WH.B1_encoder_horizontal_impuls1', 'WH.B2_encoder_horizontal_impuls2',
                                     'WH.B3_encoder_vertical_impuls1', 'WH.B4_encoder_vertical_impuls2',
                                     'WH.Q3_horizontal_towards_rack', 'WH.Q3_horRack',
                                     'WH.Q4_horBelt', 'WH.Q4_horizontal_towards_conveyor_belt', 'WH.Q5_verDown',
                                     'WH.Q5_vertical_downward', 'WH.Q6_vertical_upward', 'WH.Q6_verUP',
                                     'WH.Q7_cantForw', 'WH.Q7_cantilever_forward',
                                     'WH.Q8_cantBack', 'WH.Q8_cantilever_backward', 'WH.Q1_beltForward',
                                     'WH.Q2_beltBackward', 'WH.isEmpty', 'WH.desiredPalletX', 'WH.desiredPalletY'],
                            index=dfb.index)

df_visual = pd.DataFrame(data=dfb,
                         columns=['Visual.start', 'Visual.Stoplight'],
                         index=dfb.index)

df_sorting = pd.DataFrame(data=dfs,
                          columns=['A0.0', 'A0.1', 'A0.2', 'A0.3', 'A0.4', 'C0', 'C1', 'C2', 'E0.0', 'E0.1', 'E0.2',
                                   'EW288', 'M0.2', 'M0.4', 'M0.6', 'M0.7', 'M1.0', 'M1.1', 'M1.2', 'M1.5', 'M1.6',
                                   'M2.0', 'M2.1',
                                   'M2.2', 'M2.3', 'M2.4', 'M2.6', 'M2.7', 'M3.0', 'M3.1', 'MW12', 'MW14', 'MW16',
                                   'MW18'],
                          index=dfs.index)
df_sorting = df_sorting.drop(df_sorting.columns[df_sorting.apply(lambda col: col.nunique() == 1)], axis=1)
df_sorting = df_sorting.drop(df_sorting.columns[df_sorting.apply(lambda col: col.nunique() == 0)], axis=1)

df_drilling = pd.DataFrame(data=dfs,
                           columns=['A0.5', 'A0.6', 'A0.7', 'A1.0', 'A1.1', 'A1.2', 'A1.3', 'A1.4', 'A1.5', 'A1.6',
                                    'A1.7', 'A4.0', 'A4.1', 'A4.2', 'C4', 'E0.6', 'E0.7', 'E1.0', 'E1.1', 'E1.2',
                                    'E1.3', 'E1.4', 'E1.5', 'E1.6', 'M3.2', 'M3.3', 'M3.4', 'M3.5', 'M3.6', 'M3.7',
                                    'M4.0', 'M4.1', 'M4.2', 'M4.3', 'M4.4', 'M4.5', 'M4.6', 'M4.7', 'M5.0', 'M5.1',
                                    'M5.2', 'M5.3',
                                    'M5.4', 'M5.5', 'M5.6', 'M5.7', 'M6.0', 'M6.1', 'M6.2', 'M6.3', 'M6.4', 'M6.5',
                                    'M6.6', 'M6.7', 'M7.0', 'M7.1', 'M7.2', 'M7.3', 'M7.4', 'M7.5', 'M7.6', 'MD28',
                                    'MW20', 'MW22', 'MW26', 'MW32'],
                           index=dfs.index)
df_drilling = df_drilling.drop(df_drilling.columns[df_drilling.apply(lambda col: col.nunique() == 1)], axis=1)
df_drilling = df_drilling.drop(df_drilling.columns[df_drilling.apply(lambda col: col.nunique() == 0)], axis=1)

df_beckhoff = pd.DataFrame(data=dfs,
                           columns=['A200.0', 'A200.1', 'A200.2', 'A200.3', 'A200.4', 'A200.5', 'A200.6', 'A201.0',
                                    'A201.1', 'E200.0', 'EW256', 'EW258'],
                           index=dfs.index)
df_beckhoff = df_beckhoff.drop(df_beckhoff.columns[df_beckhoff.apply(lambda col: col.nunique() == 1)], axis=1)
df_beckhoff = df_beckhoff.drop(df_beckhoff.columns[df_beckhoff.apply(lambda col: col.nunique() == 0)], axis=1)

df_balluff = pd.DataFrame(data=dfs,
                          columns=['A6.4', 'A6.5', 'AW256', 'M0.3', 'M0.5', 'M1.3', 'M1.4', 'M24.0', 'M24.1', 'M24.2',
                                   'M24.3'],
                          index=dfs.index)

df_balluff = df_balluff.drop(df_balluff.columns[df_balluff.apply(lambda col: col.nunique() == 1)], axis=1)
df_balluff = df_balluff.drop(df_balluff.columns[df_balluff.apply(lambda col: col.nunique() == 0)], axis=1)

df_main = pd.DataFrame(data=dfs,
                       columns=['E0.3', 'E0.4', 'E0.5', 'M38.0', 'M38.1'],
                       index=dfs.index)

df_main = df_main.drop(df_main.columns[df_main.apply(lambda col: col.nunique() == 1)], axis=1)
df_main = df_main.drop(df_main.columns[df_main.apply(lambda col: col.nunique() == 0)], axis=1)

# %%  PCA auf den warehouse arbeitsplatz
x = preprocessing.StandardScaler().fit_transform(df_warehouse.values)
pca = PCA(n_components=1)
pc = pca.fit_transform(x)

warehouse_pc = pca.fit_transform(df_warehouse)

warehouse_pc = pd.DataFrame(data=warehouse_pc,
                            columns=['warehouse_1'],
                            index=df_warehouse.index)

bool_warehouse_plot = plt.figure(figsize=(16, 10))
ax = plt.gca()
warehouse_pc.plot(ax=ax)
plt.show()

print('hallo')