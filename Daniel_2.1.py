# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:21:42 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import statsmodels as sm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go # Für andere Dartstellung (Paket erst installieren)
from plotly.offline import plot #Für andere Darstellung (Paket erst installieren)

 # swgfwef

#%%  daten einladen
dfb = pd.read_csv(r"D:\Studium\Master\2. Semester SS21\54020 WPM1_Machine Learning\Datensätze\Aufgabe 2\Beispieldatensatz Beckhoff SPS.csv",
    sep=';',
    low_memory=False,
    index_col=0
).drop(columns='Unnamed: 54')



dfs = pd.read_csv(r"D:\Studium\Master\2. Semester SS21\54020 WPM1_Machine Learning\Datensätze\Aufgabe 2\Beispieldatensatz Siemens S7 SPS.csv",
    sep=';',
     low_memory=False,
     index_col=0
).drop(columns='Unnamed: 143')

    
#%%
# Datensätze zusammenführen & Spalten mit konstanten Werten entfernen

df = pd.merge_ordered(dfb, dfs, on='timestamp', how='inner')
df.set_index('timestamp', inplace=True)
df = df.drop(df.columns[df.apply(lambda col: col.nunique() == 1)],axis=1)


# Neues Dataframe nur mit Integer
dfint = df.select_dtypes(exclude=['bool', 'object'])
dfint.info()

int_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
dfint.plot(ax=ax)
plt.yscale('log')

plt.show()

# Gesamtplot mit StandardScaler
x = preprocessing.StandardScaler().fit_transform(dfint.values)

dfint.info()
int_sc = pd.DataFrame(data=x,
              columns=['WH.desiredPalletX', 'WH.desiredPalletY', 'MW116', 'MW12', 'MW18', 'MW20', 'MW22', 'MW26', 'AW256'],
              index=dfint.index)


int_plot = plt.figure(figsize=(16,10))
int_sc.plot()


plt.show()

#%%
#PCA  aus StandardScaler Spalten zu zwei Spalten zusammenfassen
pca = PCA(n_components=2)
pc = pca.fit_transform(x)

pc = pd.DataFrame(data=pc,
              columns=['PCA1', 'PCA2'],
              index=dfint.index)

plt.figure(figsize=(16,10))
plt.scatter(pc.iloc[:, 0], pc.iloc[:, 1])
plt.show()


#%%
# PCA1,2 in Scatter 

fig, ax = plt.subplots() 


ax.scatter(pc.index, pc["PCA1"], color="b", label="PCA1")
ax.scatter(pc.index, pc["PCA2"], color = "r",label="PCA2")

ax.set_xlabel("Timestamps")
ax.set_ylabel("PCA1/PCA2")
ax.set_title("Scatterplot PCA integer merged")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()



#%% 
# Für darstellung im Browser -> Pakete von oben installieren
# pc['timestamp'] = df.index
# fig = go.Figure()
# # Add traces
# fig.add_trace(go.Scatter(x=pc['timestamp'], y=pc['PCA1'],
#                     mode='markers',
#                     name='PCA1'))
# fig.add_trace(go.Scatter(x=pc['timestamp'], y=pc['PCA2'],
#                     mode='markers',
#                     name='PCA2'))

# plot(fig)

#%%
#Kmeans auf int.PCA

kmeans = KMeans(
    init="k-means++",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(pc)

int_kmeans = kmeans.predict(pc)

#%%
# Kmeans in Scatter

fig, ax = plt.subplots()

plt.scatter(pc.iloc[:, 0], pc.iloc[:, 1], c=int_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Scatterplot PCA_Kmeans integer merged")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)

#%%




#%% Schaubild PCA mit "Index" Lables
cluster_labels=kmeans.labels_

print("cluster labels of points:", cluster_labels)

fig, ax = plt.subplots()

print("data with two features:\n", pc)
plt.plot(pc.iloc[:,0],pc.iloc[:,1],'ro')
for i in range(pc.shape[0]):        
        plt.text(pc.iloc[i,0],pc.iloc[i,1], str(i), fontsize=16)
        
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Scatterplot PCA_Kmeans integer merged")
 
plt.show()

### Fraglich ob Sinvoll. Eigentlich reicht euch kurz der Blick in den Variable explorer auf pc und nach Größe ordnen.

#%%Beckhof
# Neues Dataframe nur mit Bool
dfb_bool = dfb.select_dtypes(exclude=['int', 'object', 'int64'])
#dfb_bool.info()

# True & False zu 1&0
dfb_bool = dfb_bool.applymap(lambda x: 1 if x == True else x)
dfb_bool= dfb_bool.applymap(lambda x: 0 if x == False else x)

#Löschen der Spalten welche einen Konstanten wert haben
dfb_bool = dfb_bool.drop(dfb_bool.columns[dfb_bool.apply(lambda col: col.nunique() == 1)],axis=1) 
#Löschen der Spalten welche keine Werte haben
dfb_bool = dfb_bool.drop(dfb_bool.columns[dfb_bool.apply(lambda col: col.nunique() == 0)],axis=1) 
#dfb_bool.info()

#%% Stationen Beckhoff
b_bool_robot= pd.DataFrame(data=dfb_bool,
              columns=['VG.I1_ref_switch_vertical', 'VG.I2_ref_switch_horizontal', 'VG.B1_encoder_vertical_impulse1', 'VG.B2_encoder_vertical_impulse2', 'VG.B3_encoder_horizontal_impulse1',
                       'VG.B4_encoder_horizontal_impulse2', 'VG.B5_encoder_rotate_impulse1', 'VG.B6_encoder_rotate_impulse2','VG.safeCurtainShort', 'VG.Q1_verUp', 'VG.Q2_verDown', 'VG.Q3_horBack', 'VG.Q4_horForw',
                       'VG.Q5_rotClockwise', 'VG.Q6_rotCounterClockwise', 'VG.Q7_compressorON', 'VG.Q8_valve' ],
              index=dfb_bool.index)


b_bool_warehouse=pd.DataFrame(data=dfb_bool,
              columns=['WH.I1_Ref_switch_horizontal', 'WH.I2_Light_barrier_inside', 'WH.I3_Light_barrier_outside','WH.I5_Ref_switch_cantilever_front', 'WH.I6_Ref_switch_cantilever_back',
                       'WH.B1_encoder_horizontal_impuls1', 'WH.B2_encoder_horizontal_impuls2', 'WH.B3_encoder_vertical_impuls1', 'WH.B4_encoder_vertical_impuls2', 'WH.Q3_horizontal_towards_rack', 'WH.Q3_horRack',
                       'WH.Q4_horBelt', 'WH.Q4_horizontal_towards_conveyor_belt', 'WH.Q5_verDown', 'WH.Q5_vertical_downward', 'WH.Q6_vertical_upward', 'WH.Q6_verUP', 'WH.Q7_cantForw', 'WH.Q7_cantilever_forward',
                       'WH.Q8_cantBack', 'WH.Q8_cantilever_backward', 'WH.Q1_beltForward', 'WH.Q2_beltBackward', 'WH.isEmpty'], 
              index=dfb_bool.index)

b_bool_visual=pd.DataFrame(data=dfb_bool,
              columns=['Visual.start', 'Visual.Stoplight'], 
              index=dfb_bool.index)

#%%PCA: Daten aus Robot Spalten zusammenfassen zu einer Spalte --1
# Beckhoff Robot
pca_1 = PCA(n_components=1)

b_robot_pc_1 = pca_1.fit_transform(b_bool_robot)

b_robot_pc_1 = pd.DataFrame(data=b_robot_pc_1,
              columns=['robot'],
              index=b_bool_robot.index)

bool_robot_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
b_robot_pc_1.plot(ax=ax)
plt.show()

#%%PCA: Daten aus Robot Spalten zusammenfassen zu zwei Spalten --2

b_robot_pc = pca.fit_transform(b_bool_robot)

b_robot_pc = pd.DataFrame(data=b_robot_pc,
              columns=['robot_1', 'robot_2'],
              index=b_bool_robot.index)

plt.figure(figsize=(16,10))
plt.scatter(b_robot_pc.iloc[:, 0], b_robot_pc.iloc[:, 1])
plt.show() #Noch eine Änderung von Johannes

#Änderung Johannes

#%%robot 1 & 2 in Scatter --3 --Kein bezug zu Timestamp
fig, ax = plt.subplots() 

ax.scatter(b_robot_pc["robot_2"], b_robot_pc["robot_1"], color="b", label="robot_1")
ax.scatter(b_robot_pc["robot_1"], b_robot_pc["robot_2"], color = "r",label="robot_2")

ax.set_xlabel("robot_1")
ax.set_ylabel("robot_2")
ax.set_title("Scatterplot PCA Beckhoff boolean robot")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
#%% scatter bezgl timestamp --3
fig, ax = plt.subplots() 

ax.scatter(b_robot_pc.index, b_robot_pc["robot_1"], color="b", label="robot_1")
ax.scatter(b_robot_pc.index, b_robot_pc["robot_2"], color = "r",label="robot_2")

ax.set_xlabel("timestamp")
ax.set_ylabel("robot_1/robot_2")
ax.set_title("Scatterplot PCA Beckhoff boolean robot")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#%% Kmeans Beckhoff boolean PCA robot_1 &_2 --4

kmeans.fit(b_robot_pc)

b_bool_robot_kmeans = kmeans.predict(b_robot_pc)
fig, ax = plt.subplots()

plt.scatter(b_robot_pc.iloc[:, 0], b_robot_pc.iloc[:, 1], c=b_bool_robot_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("robot_1")
ax.set_ylabel("robot_2")
ax.set_title("Scatterplot PCA_Kmeans Beckhoff boolean robot")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)

#%%PCA: Daten aus Warehoues Spalten zusammenfassen zu einer Spalte
# Beckhoff Warehouse
pca_1 = PCA(n_components=1)

b_warehouse_pc_1 = pca_1.fit_transform(b_bool_warehouse)

b_warehouse_pc_1 = pd.DataFrame(data=b_warehouse_pc_1,
              columns=['warehouse'],
              index=b_bool_warehouse.index)

bool_warehouse_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
b_warehouse_pc_1.plot(ax=ax)
plt.show()

#%%PCA: Daten aus Warehouse Spalten zusammenfassen zu zwei Spalten

b_warehouse_pc = pca.fit_transform(b_bool_warehouse)

b_warehouse_pc = pd.DataFrame(data=b_warehouse_pc,
              columns=['warehouse_1', 'warehouse_2'],
              index=b_bool_warehouse.index)

plt.figure(figsize=(16,10))
plt.scatter(b_warehouse_pc.iloc[:, 0], b_warehouse_pc.iloc[:, 1])
plt.show()
#%%warehouse 1 & 2 in Scatter
fig, ax = plt.subplots() 
ax.scatter(b_warehouse_pc.index, b_warehouse_pc["warehouse_1"], color="b", label="warehouse_1")
ax.scatter(b_warehouse_pc.index, b_warehouse_pc["warehouse_2"], color = "r",label="warehouse_2")

ax.set_xlabel("Timestamp")
ax.set_ylabel("warehouse_1/warehouse_2")
ax.set_title("Scatterplot PCA Beckhoff boolean warehouse")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#%% 
# Für darstellung im Browser -> Pakete von oben installieren
# b_warehouse_pc['timestamp'] = b_bool_warehouse.index
# fig = go.Figure()
# # Add traces
# fig.add_trace(go.Scatter(x=b_warehouse_pc['timestamp'], y=b_warehouse_pc['warehouse_1'],
#                     mode='markers',
#                     name='warehouse_1'))
# fig.add_trace(go.Scatter(x=b_warehouse_pc['timestamp'], y=b_warehouse_pc['warehouse_2'],
#                     mode='markers',
#                     name='warehouse_2'))

# plot(fig)
#%% Kmeans Beckhoff boolean PCA warehouse_1 &_2
kmeans.fit(b_robot_pc)

b_bool_warehouse_kmeans = kmeans.predict(b_warehouse_pc)
fig, ax = plt.subplots()

plt.scatter(b_warehouse_pc.iloc[:, 0], b_warehouse_pc.iloc[:, 1], c=b_bool_warehouse_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("warehouse_1")
ax.set_ylabel("warehouse_2")
ax.set_title("Scatterplot PCA_Kmeans Beckhoff boolean Warehouse")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)

#%%PCA: Daten aus Visualizations Spalten zusammenfassen zu einer Spalte
# Beckhoff Visual
pca_1 = PCA(n_components=1)

b_visual_pc_1 = pca_1.fit_transform(b_bool_visual)

b_visual_pc_1 = pd.DataFrame(data=b_visual_pc_1,
              columns=['visual'],
              index=b_bool_visual.index)

bool_visual_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
b_visual_pc_1.plot(ax=ax)
plt.show()
#%% visual 1 & 2 in Scatter
fig, ax = plt.subplots() 
ax.scatter(b_bool_visual.index, b_bool_visual["Visual.start"], color="b", label="Visual.start")
ax.scatter(b_bool_visual.index, b_bool_visual["Visual.Stoplight"], color = "r",label="Visual.Stoplight")

ax.set_xlabel("Timestamp")
ax.set_ylabel("Visual.start/Visual.Stoplight")
ax.set_title("Scatterplot PCA Beckhoff boolean Visual")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#%% Kmeans Beckhoff boolean PCA Visual_1 &_2
kmeans.fit(b_bool_visual)

b_bool_visual_kmeans = kmeans.predict(b_bool_visual)
fig, ax = plt.subplots()

plt.scatter(b_bool_visual.iloc[:, 0], b_bool_visual.iloc[:, 1], c=b_bool_visual_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("Visual.start")
ax.set_ylabel("Visual.Stoplight")
ax.set_title("Scatterplot PCA_Kmeans Beckhoff boolean Visual")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)
#%%Siemens
# Neues Dataframe nur mit Bool
#Betrachtung des Dfb1 alle Booleons anschauen 
del dfs['I200.7']
del dfs['E6.0']
del dfs['E6.1']
del dfs['E200.1']
del dfs['E200.2']


dfs_bool = dfs.select_dtypes(exclude=['int64', 'object','float64'])   # alle Löschen auser die Bool spalten 
dfs_bool.info()


dfs_bool = dfs_bool.applymap(lambda x: 1 if x == True else x)
dfs_bool = dfs_bool.applymap(lambda x: 0 if x == False else x)


dfs_bool = dfs_bool.drop(dfs_bool.columns[dfs_bool.apply(lambda col: col.nunique() == 1)],axis=1)
dfs_bool = dfs_bool.drop(dfs_bool.columns[dfs_bool.apply(lambda col: col.nunique() == 0)],axis=1)
dfs_bool.info()
#%% Stationen Simens
s_bool_driling= pd.DataFrame(data=dfs_bool,
              columns=['E0.6', 'E0.7', 'E1.0', 'E1.1', ' E1.2',
                       'E1.3', 'E1.4', 'E1.5','E1.6'],
              index=dfs_bool.index)


s_bool_Beckhoff =pd.DataFrame(data=dfs_bool,
              columns=['E200.0'], 
              index=dfs_bool.index)


s_bool_sorting =pd.DataFrame(data=dfs_bool,
              columns=['E0.0', 'E0.1'], 
              index=dfs_bool.index)


s_bool_main =pd.DataFrame(data=dfs_bool,
              columns=['E0.3', 'E0.4','E0.5'], 
              index=dfs_bool.index)

#%%PCA: Daten aus  Spalten zusammenfassen zu einer Spalte --1
# Siemens Sorting
pca_1 = PCA(n_components=1)

s_sorting_pc_1 = pca_1.fit_transform(s_bool_sorting)

s_sorting_pc_1 = pd.DataFrame(data=s_sorting_pc_1,
              columns=['sorting'],
              index=s_bool_sorting.index)

bool_sorting_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
s_sorting_pc_1.plot(ax=ax)
plt.show()
#%% sorting in Scatter
fig, ax = plt.subplots() 
ax.scatter(s_bool_sorting.index, s_bool_sorting["E0.0"], color="b", label="E0.0")
ax.scatter(s_bool_sorting.index, s_bool_sorting["E0.1"], color = "r",label="E0.1")

ax.set_xlabel("Timestamp")
ax.set_ylabel("E0.0/E0.1")
ax.set_title("Scatterplot PCA Siemens boolean Sorting")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
#%% Kmeans Siemens boolean PCA Sorting --> Fraglich wie sinnvoll 
kmeans.fit(s_bool_sorting)

s_bool_sorting_kmeans = kmeans.predict(s_bool_sorting)
fig, ax = plt.subplots()

plt.scatter(s_bool_sorting.iloc[:, 0], s_bool_sorting.iloc[:, 1], c=s_bool_sorting_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("E0.0")
ax.set_ylabel("E0.1")
ax.set_title("Scatterplot PCA_Kmeans Siemens boolean Sorting")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)

#%%PCA: Daten aus  Spalten zusammenfassen zu einer Spalte 
# Siemens Main
pca_1 = PCA(n_components=1)

s_main_pc_1 = pca_1.fit_transform(s_bool_main)

s_main_pc_1 = pd.DataFrame(data=s_main_pc_1,
              columns=['main'],
              index=s_bool_main.index)

s_main_plot = plt.figure(figsize=(16,10))
ax = plt.gca()
s_main_pc_1.plot(ax=ax)
plt.show()

#%% sorting in Scatter
fig, ax = plt.subplots() 
ax.scatter(s_bool_main.index, s_bool_main["E0.3"], color="b", label="E0.3")
ax.scatter(s_bool_main.index, s_bool_main["E0.4"], color = "r",label="E0.4")
ax.scatter(s_bool_main.index, s_bool_main["E0.5"], color = "g",label="E0.5")

ax.set_xlabel("Timestamp")
ax.set_ylabel("E0.3/E0.4/E0.5")
ax.set_title("Scatterplot PCA Siemens boolean Main")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#%%PCA: Daten aus Main Spalten zusammenfassen zu zwei Spalten --> Fraglich wie sinnvoll 

s_main_pc = pca.fit_transform(s_bool_main)

s_main_pc = pd.DataFrame(data=s_main_pc,
              columns=['Main_1', 'Main_2'],
              index=s_bool_main.index)

kmeans.fit(s_main_pc)

s_main_pc_kmeans = kmeans.predict(s_main_pc)
fig, ax = plt.subplots()

plt.scatter(s_main_pc.iloc[:, 0], s_main_pc.iloc[:, 1], c=s_main_pc_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

ax.set_xlabel("Main_1")
ax.set_ylabel("Main_2")
ax.set_title("Scatterplot PCA_Kmeans Siemens boolean Main")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()
print("center of clusters:""\n", kmeans.cluster_centers_)

