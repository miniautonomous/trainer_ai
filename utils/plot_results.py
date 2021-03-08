import matplotlib.pyplot as plt


def plot_results(history, history_keys, training_dictionary):
    """
        Plot the results of a simulation.

    Parameters
    ----------
    history: (tf history) result of a Keras model fit
    history_keys: (list) list of losses and metrics related to the triaining fit
    training_dictionary: (dict) configuration of training options

    """
    # Plot loss
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=0.6)
    ax.plot(history.history[history_keys[0]], color='b', linestyle='-', linewidth=5)
    ax.plot(history.history[history_keys[2]], color='r', linestyle='--', linewidth=5)
    plt.title('Loss', fontsize=18)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['train', 'validation'], loc='best')

    # Plot accuracy
    ax = plt.subplot(2, 1, 2)
    ax.plot(history.history[history_keys[1]], color='b', linestyle='-', linewidth=5)
    ax.plot(history.history[history_keys[3]], color='r', linestyle='--', linewidth=5)
    plt.title('Accuracy', fontsize=18)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Mean Absolute Error', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['train', 'validation'], loc='best')
    if training_dictionary['save_curve']:
        fig.savefig('loss_and_accuracy.png')
    plt.show()
