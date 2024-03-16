import matplotlib.pyplot as plt

def rf_plots(ids, accuracy_scores):
    # Plotting
    plt.figure(figsize=(10, 6))  
    plt.plot(ids, accuracy_scores, marker='o', linestyle='-', color='b') 
    plt.xlabel('subject id')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by subject id')
    plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
    plt.grid(True)  # Adds a grid for easier readability
    plt.show()