import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from WhoopNNs import create_data


def cluster_viz(whoop_data):
    features = whoop_data[['HRV', 'RHR']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    cluster_sizes = range(3, 8)

    n_rows = len(cluster_sizes) // 2 + len(cluster_sizes) % 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  
    
    for i, k in enumerate(cluster_sizes):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        clusters = kmeans.labels_
    
        axes[i].scatter(features['HRV'], features['RHR'], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
        axes[i].set_title(f'K-Means Clustering (K={k})')
        axes[i].set_xlabel('HRV')
        axes[i].set_ylabel('RHR')
    
    if len(cluster_sizes) % 2 != 0:
        axes[-1].axis('off')
        plt.tight_layout()
    plt.show()
    
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

def run_rf(data):
    trial_one_data = data
    if 'cluster_assign' in trial_one_data.columns:
        trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety', 'cluster_assign'])
    else:
        trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety'])

    X = trial_one_data.drop('Pre STAI State Anxiety', axis=1)  
    y = trial_one_data['Pre STAI State Anxiety']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_

    print("CV Scores:", cv_scores)
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

    y_pred = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy



if __name__ == '__main__':
    data = pd.read_excel('whoop/BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_data(data)
    
    ##### FULL ####
    # cluster_viz(whoop_data)
    clustered_data = cluster_actual(whoop_data)
    for k, clusters in clustered_data.items():
        print(f'Running random forest on clusters from {k}...')
        acc = []
        for cluster_idx, cluster_data in clusters.items():
            print(f'  Subgroup {cluster_idx + 1}...')
            acc.append(run_rf(cluster_data))
        print(f'Average accuracy on cluster {k} was {np.mean(acc)}')
    
    #### SMALL ####
    # small_data = whoop_data[['ScrSubjectID', 'RHR', 'HRV', 'Pre STAI State Anxiety', 'Post STAI State Anxiety']]
    # print(small_data.columns)
    # cluster_viz(small_data)
    # clustered_data = cluster_actual(small_data)
    # for k, clusters in clustered_data.items():
    #     print(f'Running random forest on clusters from {k}...')
    #     acc = []
    #     for cluster_idx, cluster_data in clusters.items():
    #         print(f'  Subgroup {cluster_idx + 1}...')
    #         acc.append(run_rf(cluster_data))
    #     print(f'Average accuracy on cluster {k} was {np.mean(acc)}')
        