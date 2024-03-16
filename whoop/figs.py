import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from util import create_whoop_data

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

def whoop_scatter(complete_whoop_dataset, num_ids):
    graph_data = complete_whoop_dataset[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']]
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    selected_ids = graph_data['ScrSubjectID'].unique()[:num_ids]
    selected_data = graph_data[graph_data['ScrSubjectID'].isin(selected_ids)]
    color_map = {id: color for id, color in zip(selected_ids, plt.cm.jet(np.linspace(0, 1, len(selected_ids))))}

    anxiety_levels = [(0, 'o'), (1, 'x')]

    for i, (anxiety, marker) in enumerate(anxiety_levels):
        subset = selected_data[selected_data['Pre STAI State Anxiety'] == anxiety]
        for sub_id in subset['ScrSubjectID'].unique():
            sub_subset = subset[subset['ScrSubjectID'] == sub_id]
            axs[i].scatter(sub_subset['RHR'], sub_subset['HRV'], s=100, alpha=0.5, marker=marker, 
                           label=f'ID={sub_id}', c=[color_map[sub_id]])
        axs[i].set_title(f'HRV vs RHR for Anxiety Level {anxiety}')
        axs[i].set_xlabel('Resting Heart Rate (RHR)')
        axs[i].set_ylabel('Heart Rate Variability (HRV)')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def rf_plots(ids, accuracy_scores):
    # Plotting
    plt.figure(figsize=(10, 6))  
    plt.plot(ids, accuracy_scores, marker='o', linestyle='-', color='b') 
    plt.xlabel('ScrSubjectID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by ScrSubjectID')
    plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
    plt.grid(True)  # Adds a grid for easier readability
    plt.show()


if __name__ == '__main__':
    data = pd.read_excel('BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_whoop_data(data)
    whoop_scatter(whoop_data, 100)

