import numpy as np

import operator
import sys
import gc
import re
# listdir
from os import listdir

from fileinput import filename
import math
import matplotlib
import matplotlib.pyplot as plt
import os

from config import *


table_X = {}

def get_index(userT):
    userT = list(set(userT))
    User_List = userT
    return User_List


def read_train_data():
    """
    只读入原始训练数据，数据并未经过embedding
    :return: 分割好的train和test
    """
    test_T = list()
    test_UserT = list()
    test_lens = list()
    ftraindata = open('../../data/TEST/data/gowalla_scopus_1104.dat', 'r')  # gowalla_scopus_1006.dat
    tempT = list()  #
    pointT = list()  #
    userT = list()  # User ID
    seqlens = list()  #
    item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(int(i))
        tempT.append(X)
        userT.append(X[0])
        pointT.append(X[1:])
        seqlens.append(len(X) - 1)  #
        item += 1
    # Test 98481
    Train_Size = 20000
    pointT = pointT[:Train_Size]
    userT = userT[:Train_Size]
    seqlens = seqlens[:Train_Size]
    User_List = get_index(userT)
    print('Count average length of trajectory')
    avg_len = 0
    for i in range(len(pointT)):
        avg_len += len(pointT[i])

    print('--------Average Length=', avg_len / (len(pointT)))
    print(User_List)
    print("Index numbers", len(User_List))
    print("point T", pointT[Train_Size - 1])
    flag = 0
    count = 0
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  #
    rate = 0.1  # 10% for test

    # 对于一个用户的轨迹，找到最后的10%作为test，剩下的作为train，用户id为70的只有一条轨迹，无法分割test，因此只有train
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):  # 当前用户不是flag 或者 找到pointT的最后一个，进入
            User += 1
            # split data
            if (count > 1):  # 之前一直连续匹配到flag的数目，如果这个用户只有一条轨迹的话，只有train，没有test
                # print "count",count," ",index
                test_T += (pointT[int((index - math.ceil(count * rate))):index])  # test point
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])  # test user
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])  # test seq len
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1  #
            flag = userT[index]  #
        else:
            count += 1

    pointT = temp_pointT
    userT = temp_userY
    seqlens = temp_seqlens
    print('training Numbers=', item - 1)
    print('div number=', Train_Size)
    print('Train Size=', len(userT), ' Test Size=', len(test_UserT), "User numbers=", User)
    print(test_T[-1])

    return tempT, pointT, userT, seqlens, test_T, test_UserT, test_lens, User_List  #


def get_xs():  #
    """
    相当于读取point的embedding，存到table_X中进行查找
    :return: 读取到的embedding向量
    """
    fpointvec = open('../../data/TEST/data/gowalla_vector_new250d.dat', 'r')  #gowalla_vector_250d.dat  gowalla_em_250.dat
    #     table_X={}  #
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()
        if len(lineArr) < 250 or lineArr[0] == '</s>':
            continue
        item += 1  #
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  #
        table_X[int(lineArr[0])] = X
    print("point number item=", item)

    return table_X


def get_one_hot(i):
    x = [0] * n_classes
    x[i] = 1
    return x


def get_mask_index(value, User_List):
    #     print User_List #weikong
    return User_List.index(value)


def get_pvector(i):  #
    return table_X[i]


def get_true_index(index, User_List):
    return User_List[index]



if __name__ == "__main__":
    tempT, pointT, userT, seqlens, test_T, test_UserT, test_lens, User_List = read_train_data()

    print(pointT[0])

    get_xs()

    print(table_X[pointT[0][0]])
