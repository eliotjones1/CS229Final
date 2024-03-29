from util import create_wesad_data, run_rf, run_xgb
import pandas as pd
from figs import rf_plots
import numpy as np

if __name__ == '__main__':
    data = create_wesad_data('classification/wesad-chest-combined-classification-hrv.csv')

    #### FULL DATASET RF (3-task classification) ####
    print('TRIAL ONE: WESAD Full dataset RF')
    trial_1_acc, trial_1_precision, trial_1_recall, trial_1_f1 = run_rf(data, verbose=1, full_metrics=True, print_metrics=True)
    #### FULL DATASET XGB (3-task classification) ####
    print('TRIAL TWO: WESAD Full dataset XGB')
    trial_2_acc = run_xgb(data)

    #### PER-PERSON RF ####
    print('TRIAL THREE: WESAD Person-specific random forest (full dataset)')
    persons_data = data['subject id'].unique()

    accuracies = {}
    list_of_accuracies = []
    for person in persons_data:
        person_data = data[data['subject id'] == person]
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
