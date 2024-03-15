import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from gcn import GCN

def create_data(data):
    '''
    Columns: 
    'ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
           'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
           'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
           'Pre STAI State Anxiety', 'Post STAI State Anxiety',
           'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
           'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
           'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
           'Respiration Rate', 'Sleep T Score',
           'Overall, how easy was it to use the instructions/videos for doing the intervention?',
           'Overall, how easy was it do do the daily intervention?',
           'subjective experience: grounding', 'subjective experience: focusing',
           'subjective experience: calming', 'subjective experience: energizing',
           'subjective experience: brief and simple',
           'subjective experience: other positive',
           'subjective experience: some challenges'
    
           Want to keep:
           ['ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
           'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
           'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
           'Pre STAI State Anxiety', 'Post STAI State Anxiety',
           'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
           'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
           'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
           'Respiration Rate', 'Sleep T Score']
        '''
        
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
    return complete_whoop_dataset
    
# def basic_nn(data):
#     X = data.drop(['Pre STAI State Anxiety', 'ScrSubjectID'], axis=1)  # Features
#     y = data['Pre STAI State Anxiety']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     
#     # norm
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     
#     # model def
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
# 
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     # training
#     model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
#     
#     loss, accuracy = model.evaluate(X_test_scaled, y_test)
#     print(f'Loss: {loss}, Accuracy: {accuracy}')
    
def create_graph_data(data):
    node_features = data.drop(['Pre STAI State Anxiety', 'ScrSubjectID'], axis=1)
    for column in node_features.columns:
        node_features[column] = node_features[column].astype(float)
    numpy_array = node_features.to_numpy()
    node_features = torch.from_numpy(numpy_array)    
    
    node_labels = torch.tensor(data['Pre STAI State Anxiety'].values, dtype=torch.long)
    
    subj_to_index = {subj: np.flatnonzero(data['ScrSubjectID'] == subj) for subj in data['ScrSubjectID'].unique()}
    
    edge_index = []
    for indices in subj_to_index.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_index.append((indices[i], indices[j]))
                edge_index.append((indices[j], indices[i]))  # Add both directions for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Convert to PyTorch tensor
    
    num_negative_samples = edge_index.size(1)  
    negative_edge_index = []
    all_indices = set(range(len(data)))
    while len(negative_edge_index) < num_negative_samples:
        rnd_indices = np.random.choice(len(data), 2, replace=False)
        if rnd_indices[0] not in subj_to_index[data.iloc[rnd_indices[1]]['ScrSubjectID']] and \
           rnd_indices[1] not in subj_to_index[data.iloc[rnd_indices[0]]['ScrSubjectID']] and \
           (rnd_indices[0], rnd_indices[1]) not in negative_edge_index:
            negative_edge_index.append((rnd_indices[0], rnd_indices[1]))
    negative_edge_index = torch.tensor(negative_edge_index, dtype=torch.long).t().contiguous()
    
    out = Data(x=node_features, edge_index=edge_index, y=node_labels, num_nodes=node_features.size(0))
    return out, all_indices, negative_edge_index
    
    
def gnn(input_data):
    data, all_indices, negative_edge_index = create_graph_data(input_data)
    
    # model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=data.num_features, num_classes=2).to(device)
    data = data.to(device)
    data.x = data.x.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # train/test
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(0.8 * data.num_nodes)] = True
    data.test_mask = ~data.train_mask
    
    model.train()
    for epoch in range(200):  # Number of epochs can be adjusted
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    data = pd.read_excel('whoop/BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_data(data)
    
    # These suuuuuck
    basic_nn(whoop_data)
    basic_nn(whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']])
    
    # Per person full is better
    gnn(whoop_data)
    
    # Per person small is very bad 
    gnn(whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']])
    
    
    