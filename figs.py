import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

if __name__ == '__main__':
    data = pd.read_excel('whoop/BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')

    # List of columns we want to keep
    list_of_columns = ['ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
                       'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
                       'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
                       'Pre STAI State Anxiety', 'Post STAI State Anxiety',
                       'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
                       'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
                       'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
                       'Respiration Rate', 'Sleep T Score']

    whoop_dataset = data[list_of_columns]
    whoop_dataset = whoop_dataset[whoop_dataset['Days from  Round1 Day1'] >= 0]
    # One-hot encoding of Round 1 Exercises
    whoop_dataset = pd.get_dummies(whoop_dataset, columns=["Round 1 Exercise"])

    # Replace entries with '.' with 'NaN'
    whoop_dataset = whoop_dataset.replace('.', np.nan)

    # Remove columns that are mostly empty
    percent_missing = whoop_dataset.isnull().mean() * 100
    threshold = 50
    columns_to_drop = percent_missing[percent_missing > threshold].index.tolist()
    whoop_dataset_cleaned = whoop_dataset.drop(columns=columns_to_drop)

    # Check completeness
    complete_rows = whoop_dataset_cleaned.dropna().shape[0]

    # Drop incomplete rows
    complete_whoop_dataset = whoop_dataset_cleaned.dropna()

    # Check completeness per person
    grouped_counts = complete_whoop_dataset.groupby('ScrSubjectID').apply(lambda group: group.dropna().shape[0])

    # Want to test on Pre STAI State Anxiety. If it is above 38, say "stressed" (1), else "not stressed" (0). 
    complete_whoop_dataset.loc[:, 'Pre STAI State Anxiety'] = np.where(complete_whoop_dataset['Pre STAI State Anxiety'] > 38, 1, 0)

    print(complete_whoop_dataset.columns)
    
    ## GRAPH ##
    
    graph_data = complete_whoop_dataset[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']]
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    selected_ids = graph_data['ScrSubjectID'].unique()[:10]
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