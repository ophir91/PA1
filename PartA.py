import numpy as np
from scipy.io import loadmat
import urllib

def read_mnist():
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat?raw=true"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    mnist_raw = loadmat(mnist_path)
    x_with_bais = np.ones((mnist_raw["data"].T.shape[0], mnist_raw["data"].T.shape[1]+1))
    x_with_bais[:,1:] = mnist_raw["data"].T
    mnist = {
    "data": x_with_bais,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")
    return mnist


def check_loss(w, data):
    pass


def PLA_binary(train_data, tested_label, test_data, num_of_epochs=5):
    """

    :param train_data:
    :param tested_label: the multi label that will labeled as 1
    :param test_data:
    :param num_of_epochs:
    :return:
    """
    # init
    w = np.zeros((785, 1), dtype=np.float32)
    flag = True  # indicate that we not classify all good
    epoch = 0

    while flag:
        flag = False
        epoch += 1
        if epoch >= num_of_epochs:
            break
        for i, x, y in enumerate(zip(train_data['data'], train_data['target'])):
            if y == tested_label:
                if np.dot(x, w) > 0:
                    w = w + y*x
                    flag = True
            else:
                if np.dot(x, w) <= 0:
                    w = w + y*x
                    flag = True
            if i % 1000 == 0:
                check_loss(w, test_data)
    return w


def PLA_multi(train_data, test_data):
    w_multi = np.zeros((785, 10), dtype=np.float32)
    for label in range(0, 10):
        w_multi[:, label] = PLA_binary(train_data, label, test_data)
    return w_multi


def _main():
    mnist = read_mnist()

pass