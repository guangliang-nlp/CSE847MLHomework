import numpy as np
import random


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


def logistic_train(data, labels, epsilon, maxiter):
    """
    lr training with gradient descent
    :param data:
    :param labels:
    :param epsilon:
    :param maxiter:
    :return:
    """
    X, y = data, labels
    (m, n) = X.shape
    train_X, train_y = X[:int(m * 0.8), :], labels[:int(m * 0.8), :]
    vad_X, vad_y = X[int(m * 0.8):, :], labels[int(m * 0.8):, :]
    theta = np.zeros((n, 1))

    for i in range(maxiter):
        pred_x = sigmoid(np.dot(X, theta))
        loss = y - pred_x
        theta = theta + epsilon * np.dot(X.transpose(), loss) / m
        train_score = score(train_X, train_y, theta)
        valid_score = score(vad_X, vad_y, theta)
        #print(i, train_score, valid_score)

    return theta


def score(Xtest, Ytest, theta):
    """
    calculate the accuracy score of test data
    :param Xtest:
    :param Ytest:
    :param theta:
    :return:
    """
    pred_x = sigmoid(np.dot(Xtest, theta))
    acc = 0
    for i, j in zip(pred_x, Ytest):
        pred, label = float(i[0]), float(j[0])
        if label == 0 and pred < 0.5:
            acc += 1
        elif label == 1 and pred >= 0.5:
            acc += 1
    # print(acc, Xtest.shape[0])
    return acc / Xtest.shape[0]


def load_data(data_file, label_file):
    """
    load data from local files
    :param data_file:
    :param label_file:
    :return:
    """
    train_data_matrix, train_label = [], []
    test_data_matrix, test_label = [], []
    line_idx = 0
    for data_line, label_line in zip(open(data_file), open(label_file)):
        if 0 <= line_idx <= 2000:
            train_data_matrix.append(list(map(float, data_line.strip().split("  "))))
            train_label.append(int(float(label_line.strip())))
        else:
            test_data_matrix.append(list(map(float, data_line.strip().split("  "))))
            test_label.append(int(float(label_line.strip())))
        line_idx += 1
    return np.array(train_data_matrix), np.expand_dims(np.array(train_label), -1), np.array(
        test_data_matrix), np.expand_dims(np.array(test_label), -1)


if __name__ == "__main__":
    Xtrain, Ytrain, Xtest, Ytest = load_data("data.txt", "labels.txt")

    for n in [200,500,800,1000,1500,2000]:  # 200,500,800,1000,1500,2000
        for epsilon in [0.0001, 0.001, 0.01, 0.1, 0.2]:
            lr_weight = logistic_train(Xtrain[:n, :], Ytrain[:n], epsilon=epsilon, maxiter=1000)
            test_score = score(Xtest, Ytest, lr_weight)
            print(n,"\t",epsilon, "\t",test_score)
