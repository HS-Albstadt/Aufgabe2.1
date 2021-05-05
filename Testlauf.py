
import pandas as pd
import numpy as np
import statsmodels as sm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import plotly.graph_objects as go # Für andere Dartstellung (Paket erst installieren)
#from plotly.offline import plot #Für andere Darstellung (Paket erst installieren)
#%%  daten einladen
dfb = pd.read_csv("C:\\Users\D.Albrecht\Documents\Hochschule\Machine Learning\Beispieldatensatz_Beckhoff_SPS.csv",
    sep=';',
    low_memory=False
).drop(columns='Unnamed: 54')



dfs = pd.read_csv("C:\\Users\D.Albrecht\Documents\Hochschule\Machine Learning\Beispieldatensatz_Siemens_S7_SPS.csv",
    sep=';',
     low_memory=False
).drop(columns='Unnamed: 143')
 