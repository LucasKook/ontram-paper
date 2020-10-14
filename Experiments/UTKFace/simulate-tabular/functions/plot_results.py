import matplotlib.pyplot as plt
import numpy as np

def plot_results(dat):
    plt.plot(dat.train_acc, 'blue')
    plt.plot(dat.test_acc, 'magenta')
    plt.ylim(0, 1)
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc = 'lower right')
    plt.show()
    plt.plot(dat.train_loss, 'blue')
    plt.plot(dat.test_loss, 'magenta')
    plt.ylim(0, np.max(dat.test_loss))
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc = 'lower right')
    plt.show()
    print("Max. validation accuracy: ", np.max(dat.test_acc))
    print('In epoch: ', np.where(dat.test_acc == np.max(dat.test_acc)))
    print("Min. validation loss: ", np.min(dat.test_loss))
    print('In epoch: ', np.where(dat.test_loss == np.min(dat.test_loss)))