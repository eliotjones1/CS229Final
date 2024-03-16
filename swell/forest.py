from util import create_swell_data, run_rf, run_xgb
import pandas as pd
from figs import rf_plots
import numpy as np

if __name__ == '__main__':
    data = create_swell_data('classification/combined-swell-classification-hrv-dataset.csv')
    print('TRIAL ONE: SWELL Full dataset RF')
    trial_1_acc, trial_1_precision, trial_1_recall, trial_1_f1 = run_rf(data, print_metrics=True, full_metrics=True, verbose=1)
    ### FULL DATASET XGB (3-task classification) ####
    print('TRIAL TWO: SWELL Full dataset XGB')
    trial_2_acc = run_xgb(data)

    # #### PER-PERSON RF ####
    print('TRIAL THREE: SWELL Person-specific random forest (full dataset)')
    persons_data = data['subject_id'].unique()

    accuracies = {}
    list_of_accuracies = []
    for person in persons_data:
        person_data = data[data['subject_id'] == person]
        if len(person_data) < 10:
            continue
        indiv_acc, _, _, _ = run_rf(person_data)
        if indiv_acc != -1:
            accuracies[person] = indiv_acc
            list_of_accuracies.append(indiv_acc)

    average_acc = np.mean(list_of_accuracies)
    sd_acc = np.std(list_of_accuracies)

    print(f'Accuracy average: {average_acc}, standard dev: {sd_acc}')
    ids = list(accuracies.keys())
    accuracy_scores = list(accuracies.values())
    rf_plots(ids, accuracy_scores)
    