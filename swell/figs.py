import matplotlib.pyplot as plt


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