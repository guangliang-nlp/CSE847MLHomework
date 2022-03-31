from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import linear_model
from sklearn import metrics

data = scio.loadmat("alzheimers/ad_data.mat")
features = scio.loadmat("alzheimers/feature_name.mat")

print(data.keys())

x_train, y_train, x_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

for i in range(y_train.shape[0]):
    if y_train[i,0] == -1: y_train[i,0] = 0

for i in range(y_test.shape[0]):
    if y_test[i,0] == -1: y_test[i,0] = 0

l1_list, auc_list = [],[]
non_zero = []
for l1_reg in [1e-8,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    l1_list.append(l1_reg)
    l1_lr = linear_model.Lasso(alpha=l1_reg)
    l1_lr.fit(x_train,y_train)
    pred_test = l1_lr.predict(x_test)
    #print(y_test)
    fpr, tpr, _ = metrics.roc_curve(y_test,pred_test,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    auc_list.append(auc)
    non_zero_w = 0
    #print(l1_lr.coef_)
    for idx in range(l1_lr.coef_.shape[0]):
        if l1_lr.coef_[idx] != 0:non_zero_w+=1
    non_zero.append(non_zero_w)
    print(l1_reg,"\t",non_zero_w)

"""
plt.plot(l1_list, auc_list, color="r")
plt.xlabel("l1 regularization parameter")
plt.ylabel("auc")

plt.show()
"""
plt.plot(l1_list, non_zero, color="g")
plt.xlabel("l1 regularization parameter")
plt.ylabel("num of selected features")

plt.show()

