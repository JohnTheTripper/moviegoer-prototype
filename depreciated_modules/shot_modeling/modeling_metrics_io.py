import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == 0:
            plt.text(j-0.1, i+0.3, format(cm[i, j], fmt), color="white" if cm[i, j] > thresh else "black")
        if i == 1:
            plt.text(j-0.1, i-0.2, format(cm[i, j], fmt), color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', weight='bold')
    plt.xlabel('Predicted Label', weight='bold')
    plt.show()


def evaluate_test(model, history, class_labels, train_X, test_X, train_y, test_y):
    """Return evaluation metrics for given model on given train and test sets."""
    train_loss, train_acc = model.evaluate(train_X, train_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
    print('Accuracy \n Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    print('Loss \n Train: %.3f, Test: %.3f \n' % (train_loss, test_loss))
    # plot loss during training
    plt.subplots_adjust(hspace=.5, wspace=0.5)
    plt.subplot(211)
    plt.title('Loss', weight='bold')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy', weight='bold')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.legend()
    plt.show()
    print('\n')
    # predict probabilities for test set
    yhat_probs = model.predict(test_X, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(test_X, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # calculate metrics
    report = metrics.classification_report(test_y, yhat_classes, target_names=class_labels)
    confusion_matrix = metrics.confusion_matrix(test_y, yhat_classes)
    plot_confusion_matrix(confusion_matrix, class_labels)
    print('\n')

    return report
