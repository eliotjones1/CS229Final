import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
    print(percent_missing)
    threshold = 50
    columns_to_drop = percent_missing[percent_missing > threshold].index.tolist()
    whoop_dataset_cleaned = whoop_dataset.drop(columns=columns_to_drop)
    
    # Check completeness
    complete_rows = whoop_dataset_cleaned.dropna().shape[0]
    print(f"Number of rows with full information: {complete_rows}")
    
    # Drop incomplete rows
    complete_whoop_dataset = whoop_dataset_cleaned.dropna()
    
    # Check completeness per person
    grouped_counts = complete_whoop_dataset.groupby('ScrSubjectID').apply(lambda group: group.dropna().shape[0])
    
    # Want to test on Pre STAI State Anxiety. If it is above 38, say "stressed" (1), else "not stressed" (0). 
    complete_whoop_dataset.loc[:, 'Pre STAI State Anxiety'] = np.where(complete_whoop_dataset['Pre STAI State Anxiety'] > 38, 1, 0)
    print(complete_whoop_dataset.columns)
    
    ######## TRIAL ONE: Using full dataset, without focusing on SubjectID ########
    print('TRIAL ONE: Full dataset')
    trial_one_data = complete_whoop_dataset
    trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety'])
    
    X = trial_one_data.drop('Pre STAI State Anxiety', axis=1)  
    y = trial_one_data['Pre STAI State Anxiety']
    
    # Model #1: RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameters according to the paper
    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)
    
    # 10-fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)
    
    # Fit the model
    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_
    
    print("CV Scores:", cv_scores)
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")
        
    # Evaluate the model
    y_pred = rf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Print metrics
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    ## SMALL MODEL, RHR, HRV, Sleep Score, Hours of Sleep, Sleep Efficiency, Respiration Rate ##
    print('TRIAL TWO: Only Whoop data as features')
    small_trial_data = complete_whoop_dataset
    small_trial_data = small_trial_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency', 'Respiration Rate']]

    X = small_trial_data.drop('Pre STAI State Anxiety', axis=1)  
    y = small_trial_data['Pre STAI State Anxiety']

    # Model #1: RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameters according to the paper
    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)

    # 10-fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    # Fit the model
    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_

    print("CV Scores:", cv_scores)
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

    # Evaluate the model
    y_pred = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Print metrics

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

     ## LIMITED MODEL, RHR, HRV ##
    print('TRIAL THREE: Only RHR and HRV features')
    limited_trial_data = complete_whoop_dataset
    limited_trial_data = limited_trial_data[['Pre STAI State Anxiety', 'RHR', 'HRV']]

    X = limited_trial_data.drop('Pre STAI State Anxiety', axis=1)  
    y = limited_trial_data['Pre STAI State Anxiety']

    # Model #1: RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameters according to the paper
    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)

    # 10-fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    # Fit the model
    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_

    print("CV Scores:", cv_scores)
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

    # Evaluate the model
    y_pred = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Print metrics

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    
'''
    Model on RHR, HRV only - Sylvie attempted, check my work pls
    Begin person-specific trials
    Hybrid model
    -- over time? -- 
    '''


