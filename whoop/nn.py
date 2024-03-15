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
from util import create_whoop_data, create_whoop_graph_data


def basic_nn(data):
    X = data.drop(['Pre STAI State Anxiety', 'ScrSubjectID'], axis=1)  # Features
    y = data['Pre STAI State Anxiety']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # norm
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model def
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # training
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

def gnn(input_data):
    data, all_indices, negative_edge_index = create_whoop_graph_data(input_data)
    
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
    data = pd.read_excel('BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_whoop_data(data)
    
    # These suuuuuck
    basic_nn(whoop_data)
    basic_nn(whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']])
    
    # Per person full is better
    gnn(whoop_data)
    
    # Per person small is very bad 
    gnn(whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']])
    
    
    