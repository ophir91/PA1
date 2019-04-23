import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import urllib
import time

def read_mnist():
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat?raw=true"
    mnist_path = "./mnist-original.mat"

    if not os.path.exists(mnist_path):
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
    mnist_raw = loadmat(mnist_path)
    x_with_bais = np.ones((mnist_raw["data"].T.shape[0], mnist_raw["data"].T.shape[1]+1))
    x_with_bais[:,1:] = mnist_raw["data"].T
    mnist = {
    "data": x_with_bais/255,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")
    return mnist


def print_multi_confusion_matrix(preds, targets):
    # calculate the confusion matrix; labels is numpy array of classification labels
    cm = np.zeros((10, 10))
    for t, p in zip(targets, preds):
        cm[int(t)][int(p)] += 1
    # also get the accuracy
    accuracy = (preds == targets).sum() / float(len(targets))
    # fig2 = plt.figure()
    fig2, ax2 = plt.subplots(1,1)
    # ax2.axis(list(range(10)),list(range(10)))
    ax2.set_title('Confusion Matrix, The accuracy is {}'.format(accuracy))
    ax2.matshow(cm)
    return accuracy


def check_loss(w, data, label, need_table_confusion=False):
    def TP():
        return np.sum(predict * target)
    def FP():
        return np.sum(predict * np.logical_not(target))
    def FN():
        return np.sum(np.logical_not(predict) * target)
    def TN():
        return np.sum(np.logical_not(predict) * np.logical_not(target))
    predict = (np.matmul(w, data['data'].T) > 0)
    target = (data['target'] == label)
    tp = TP()
    fp = FP()
    fn = FN()
    tn = TN()
    acc = (tp+tn)/(tp+fp+fn+tn)
    loss = (fp+fn)/(len(data['target']))
    if need_table_confusion:
        global axs_tc
        axs_tc[label // 5][label % 5].set_title('Label {} TP={}, FP={}, FN={}, TN={},\n TPR={}'
                                                .format(label, tp, fp, fn, tn, (tp/(tp+fn))))
        axs_tc[label // 5][label % 5].matshow([[tp,fp],[fn,tn]])
    return acc, loss


def save_best_w(w, w_best):
    pass


def PLA_binary(train_data, tested_label, test_data, num_of_epochs=1):
    """

    :param train_data:
    :param tested_label: the multi label that will labeled as 1
    :param test_data:
    :param num_of_epochs:
    :return:
    """
    global axs
    # init
    w = np.zeros((785, ), dtype=np.float32)
    w_best = np.copy(w)
    best_loss = 1
    flag = True  # indicate that we not classify all good
    epoch = 0
    test_acc_list = list()
    train_acc_list = list()
    test_loss_list = list()
    train_loss_list = list()
    x_tick = list()
    num_of_train_data = len(train_data['target'])
    while flag:
        flag = False
        epoch += 1
        if epoch > num_of_epochs:
            break
        for i, (x, y) in enumerate(zip(train_data['data'], train_data['target'])):
            if y == tested_label:  # -> that mean y is 1
                if float(np.matmul(w.T, x)) <= 0:
                    w = w + x
                    flag = True
            else:  # -> that mean y is -1
                if float(np.matmul(w.T, x)) > 0:
                    w = w - x
                    flag = True
            if i % 1000 == 0:
                test_acc, test_loss = check_loss(w, test_data, tested_label)
                train_acc, train_loss = check_loss(w, train_data, tested_label)
                test_acc_list.append(test_acc)
                train_acc_list.append(train_acc)

                test_loss_list.append(test_loss)
                train_loss_list.append(train_loss)

                x_tick.append(i + num_of_train_data*(epoch-1))

                # pocket algo:
                if best_loss > train_loss:
                    w_best = np.copy(w)
                    best_loss = train_loss

    # for the last iteration:
    x_tick.append(i + num_of_train_data * (epoch - 2))
    test_acc, test_loss = check_loss(w, test_data, tested_label, need_table_confusion=True)
    train_acc, train_loss = check_loss(w, train_data, tested_label)

    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)

    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)

    # pocket algo:
    if best_loss > train_loss:
        w_best = np.copy(w)
        best_loss = train_loss

    axs[tested_label][0].set_title('Accuracy for label {}'.format(tested_label))
    axs[tested_label][0].plot(x_tick, test_acc_list, label='Test accuracy')
    axs[tested_label][0].plot(x_tick, train_acc_list, label='Train accuracy')
    axs[tested_label][0].legend()

    axs[tested_label][1].set_title('Loss for label {}'.format(tested_label))
    axs[tested_label][1].plot(x_tick, test_loss_list, label='Test loss')
    axs[tested_label][1].plot(x_tick, train_loss_list, label='Train loss')
    axs[tested_label][1].legend()


    return w_best


def PLA_multi(train_data, test_data):
    w_multi = np.zeros((785, 10), dtype=np.float32)
    for label in range(0, 10):
        w_multi[:, label] = PLA_binary(train_data, label, test_data)
    return w_multi


def predict(w_multi, input):
    return np.argmax(np.matmul(w_multi.T, input.T), axis=0)


def _main():
    start_train_time = time.time()
    train_data = {}
    test_data = {}
    mnist = read_mnist()
    train_data['data'], test_data['data'], train_data['target'], test_data['target'] = \
        train_test_split(mnist['data'], mnist['target'], test_size=int(1e4), random_state=42)
    w_multi = PLA_multi(train_data, test_data)
    print('Train finished in {} sec'.format(int(time.time() - start_train_time)))
    res = predict(w_multi, test_data['data'])
    accuracy = print_multi_confusion_matrix(res, test_data['target'])

    print('The accuracy for the multi_PLA is {}'.format(accuracy))
    pass


if __name__ == '__main__':
    fig, axs = plt.subplots(10, 2)
    fig_tc, axs_tc = plt.subplots(2, 5, sharex=True)
    _main()
    plt.show()
    pass