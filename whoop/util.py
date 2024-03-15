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

def create_whoop_data(data):
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

def create_whoop_graph_data(data):
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

def run_rf(data, print=False, full_metrics=False, verbose=0):
    trial_one_data = data
    if 'cluster_assign' in trial_one_data.columns:
        trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety', 'cluster_assign'])
    elif 'Post STAI State Anxiety' in trial_one_data.columns:
        trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety'])
    else:
        trial_one_data = trial_one_data.drop(columns=['ScrSubjectID'])

    X = trial_one_data.drop('Pre STAI State Anxiety', axis=1)  
    y = trial_one_data['Pre STAI State Anxiety']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if len(X_train) < 10:
        return -1, 0, 0, 0
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42, verbose=verbose)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_
    
    
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    y_pred = rf.predict(X_test_scaled)
    
    if full_metrics:
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        precision, recall, f1 = 0, 0, 0
    accuracy = accuracy_score(y_test, y_pred)
    

    # Print metrics
    if print:
        print("CV Scores:", cv_scores)
        print("Feature ranking:")
        for f in range(X_train.shape[1]):
            print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
    return accuracy, precision, recall, f1

def run_xgb(xg1_data):
    X = xg1_data.drop('Pre STAI State Anxiety', axis=1)  
    y = xg1_data['Pre STAI State Anxiety']

    # Model #1: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #GridSearch CV
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
    }

    # 10-fold CV
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    xgb_reg = xgb.XGBRegressor(objective='binary:logistic',
                               eval_metric = 'logloss',
                               eta = 0.1,
                               subsample = 0.3)

    # Perform GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy score found: ", grid_search.best_score_)

    #Train XGBoost model
    best_params = grid_search.best_params_
    best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **best_params)
    best_xgb_model.fit(X_train_scaled, y_train)

    #Predictions
    y_pred = best_xgb_model.predict(X_test_scaled)

    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", accuracy)
    return accuracy

