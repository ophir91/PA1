import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from PartA import read_mnist, print_multi_confusion_matrix


def softmax(wx):
    e_x = np.exp(wx)
    return e_x / e_x.sum(axis=0)


def get_probs_preds(data, w):
    probs = softmax(np.matmul(w.T,data))
    preds = np.argmax(probs, axis=0)
    return probs, preds


def check_accuracy(data, true, w):
    prob, pred = get_probs_preds(data, w)
    accuracy = print_multi_confusion_matrix(pred, np.argmax(true, axis=0))
    print("The accuracy for the Softmax regression is {}".format(accuracy))
    print_confusion_tables(pred, np.argmax(true, axis=0))
    return accuracy


def softmax_regression(train_data, test_data, epochs=100, lr=1):
    x = train_data['data'].T
    y = train_data['target'].T
    N = x.shape[1]
    losses = []
    w_multi = np.zeros((785, 10), dtype=np.float32)
    for epoch in range(epochs):
        wx = np.matmul(w_multi.T, x)
        y_pred = softmax(wx)
        loss = (-1 / N) * np.sum(y * np.log(y_pred))
        div_loss = (-1 / N) * np.matmul(x,(y - y_pred).T)
        losses.append(loss)
        w_multi = w_multi - (lr * div_loss)
    plt.plot(losses)
    return w_multi


def print_confusion_tables(pred, true):
    fig_tc, axs_tc = plt.subplots(2, 5, sharex=True)
    for label in range(10):
        target = (true == label)
        predict = (pred == label)
        tp = np.sum(predict * target)
        fp = np.sum(predict * np.logical_not(target))
        fn = np.sum(np.logical_not(predict) * target)
        tn = np.sum(np.logical_not(predict) * np.logical_not(target))

        axs_tc[label // 5][label % 5].set_title('Label {} TP={}, FP={}, FN={}, TN={},\n TPR={}'
                                                .format(label, tp, fp, fn, tn, (tp/(tp+fn))))
        axs_tc[label // 5][label % 5].matshow([[tp,fp],[fn,tn]])


def to_one_hot(y):
    _y = np.expand_dims(y, axis=1).astype(np.int)
    y_one_hot = np.zeros((_y.size, _y.max()+1))
    y_one_hot[np.arange(_y.size), _y.T] = 1
    return y_one_hot


def _main():
    start_train_time = time.time()
    train_data = {}
    test_data = {}
    mnist = read_mnist()
    train_data['data'], test_data['data'], train_data['target'], test_data['target'] = \
        train_test_split(mnist['data'], mnist['target'], test_size=int(1e4), random_state=42)
    train_data['target'] = to_one_hot(train_data['target'])
    test_data['target'] = to_one_hot(test_data['target'])
    w_multi = softmax_regression(train_data,test_data)
    print('Train finished in {} sec'.format(int(time.time() - start_train_time)))
    check_accuracy(test_data['data'].T, test_data['target'].T, w_multi)
    plt.show()


if __name__ == '__main__':
    _main()



