import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from util import create_whoop_data
from figs import cluster_viz
    
def cluster_actual(whoop_data):
    features = whoop_data[['HRV', 'RHR']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    cluster_sizes = range(3, 8)
    clustered_data_sets = {}
    for k in cluster_sizes:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        clusters = kmeans.labels_
        whoop_data['cluster_assign'] = clusters
        clustered_data_sets[f'k{k}'] = {cluster: whoop_data[whoop_data['cluster_assign'] == cluster] for cluster in range(k)}

    return clustered_data_sets


if __name__ == '__main__':
    data = pd.read_excel('BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_whoop_data(data)
    cluster_viz(whoop_data)
    ##### FULL ####
    clustered_data = cluster_actual(whoop_data)
    for k, clusters in clustered_data.items():
        print(f'Running random forest on clusters from {k}...')
        acc = []
        for cluster_idx, cluster_data in clusters.items():
            print(f'  Subgroup {cluster_idx + 1}...')
            acc.append(run_rf(cluster_data)[0])
        print(f'Average accuracy on cluster {k} was {np.mean(acc)}')
    
    ### SMALL ####
    small_data = whoop_data[['ScrSubjectID', 'RHR', 'HRV', 'Pre STAI State Anxiety', 'Post STAI State Anxiety']]
    print(small_data.columns)
    cluster_viz(small_data)
    clustered_data = cluster_actual(small_data)
    for k, clusters in clustered_data.items():
        print(f'Running random forest on clusters from {k}...')
        acc = []
        for cluster_idx, cluster_data in clusters.items():
            print(f'  Subgroup {cluster_idx + 1}...')
            acc.append(run_rf(cluster_data)[0])
        print(f'Average accuracy on cluster {k} was {np.mean(acc)}')
        