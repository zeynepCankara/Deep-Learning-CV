import os
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(X, y, training_mode, save_name):
    """
        Reference: https://github.com/NaJaeMin92/pytorch_DANN/
        Gets the t-sne output and actions label encodings plot T-SNE
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)


    plt.figure(figsize=(10, 10))
    for i in range(len(y)):  # X.shape[0] : 1024
    # plot colored number
      if y[i] == 0:
          colors = (0.0, 0.0, 1.0, 1.0)
      elif y[i] == 1:
          colors = (1.0, 0.0, 0.0, 1.0)
      elif y[i] == 2:
          colors = (1.0, 1.0, 0.0, 1.0)
      elif y[i] == 3:
          colors = (1.0, 1.0, 1.0, 1.0)
      elif y[i] == 4:
          colors = (1.0, 0.5, 0.0, 1.0)
      elif y[i] == 5:
          colors = (1.0, 0.0, 0.5, 1.0)
      elif y[i] == 6:
          colors = (1.0, 1.0, 0.0, 0.0)
      elif y[i] == 7:
          colors = (1.0, 0.0, 1.0, 1.0)
      elif y[i] == 8:
          colors = (0.5, 0.5, 0.5, 0.5)
      elif y[i] == 9:
          colors = (0.5, 0.2, 0.2, 0.2)
      elif y[i] == 10:
          colors = (1.0, 0.5, 0.2, 1.0)
      else:
          colors = (1.0, 0.2, 0.5, 1.0)

      plt.text(X[i, 0], X[i, 1], str(y[i]),
                color=colors,
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if save_name is not None:
        plt.title(save_name)

    save_folder = 'saved_plot'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = 'saved_plot/' + str(training_mode) + '_' + str(save_name) + '.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))


def plot_p1_train_info(training_loss, val_accuracy, save_dir = "./saved_plot/problem1_loss_acc.png"):
    """ 
        Plots training Loss and Validation Acc
    """
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.plot(training_loss, color = 'red')
    plt.title("Training Loss vs # Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Number of Epochs")
    
    plt.subplot(1,2,2)
    plt.plot(val_accuracy, color = 'blue')
    plt.title("Validation Accuracy vs # Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.savefig(save_dir)
    plt.show()


