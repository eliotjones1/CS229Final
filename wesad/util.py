import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def expand_sensor_data(data, sensor_prefix):
  """Expand multi-dimensional sensor data into a flat DataFrame with appropriate column names."""
  df_expanded = pd.DataFrame()

  for sensor, values in data.items():
    print(sensor)
    if isinstance(values, np.ndarray) and values.ndim > 1:
      for i in range(values.shape[1]):
        df_expanded[f'{sensor_prefix}{sensor}_{i}'] = values[:, i]
    else:
      df_expanded[f'{sensor_prefix}{sensor}'] = values

    return df_expanded
  
def create_wesad_data(dir):
  chest_dfs = []
  wrist_dfs = []
  combined_dfs = []
  pkl_files = [file for file in os.listdir(dir) if file.endswith('.pkl')]
  for file in pkl_files:
    file_path = os.path.join(dir, file)
    print(f'Loading data from {file_path}')
    data = pd.read_pickle(file_path)
    
    subject = [data['subject']] * len(data['label'])
    label = data['label']
    
    chest_df = expand_sensor_data(data['signal']['chest'], 'chest_')
    wrist_df = expand_sensor_data(data['signal']['wrist'], 'wrist_')
    
    print(chest_df.shape)
    print(wrist_df.shape)
    print(len(label))

    chest_df['subject'] = subject
    chest_df['label'] = label
    wrist_df['subject'] = subject
    wrist_df['label'] = label
    
    combined_df = pd.concat([chest_df, wrist_df], axis=1)
    chest_dfs.append(chest_df)
    wrist_dfs.append(wrist_df)
    combined_dfs.append(combined_df)
  
  final_chest_df = pd.concat(chest_dfs, ignore_index=True)
  final_wrist_df = pd.concat(wrist_dfs, ignore_index=True)
  final_combined_df = pd.concat(combined_dfs, ignore_index=True)

  return final_chest_df, final_wrist_df, final_combined_df

  