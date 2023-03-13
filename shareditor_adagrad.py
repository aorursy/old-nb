#!/usr/bin/env python
# coding: utf-8



# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import sys


base_learning_rate = 0.01


def rand_list(dim):
    return np.random.rand(dim)


def zero_list(dim):
    return np.linspace(0, 0, dim)


def predict_y(row, norm_avg, norm_max, w, b):
    # 不做正则会发散
    row = normalize(row, norm_avg, norm_max)
    x = row[0:3]
    y = row[3]
    y_ = np.dot(w, x) + b
    return x, y, y_


def normalize(row, norm_avg, norm_max):
    return [(row[i]-norm_avg[i])/(norm_max[i]-norm_avg[i]) for i in range(4)]


def reverse(y_, norm_avg, norm_max):
    return y_ * (norm_max[3] - norm_avg[3]) + norm_avg[3]


def train():
    train_df = pd.read_csv('../input/train_data.csv')
    norm_avg = [np.average(train_df.ix[:, col_name]) for col_name in
                ['total_rooms', 'population', 'median_income', 'median_house_value']]
    norm_max = [np.max(train_df.ix[:, col_name]) for col_name in
                ['total_rooms', 'population', 'median_income', 'median_house_value']]

    w = rand_list(3)
    b = rand_list(1)

    eps = 1e-7
    history_gradient_square_w = zero_list(3)
    history_gradient_square_b = zero_list(1)

    for i in range(10):
        total_err_w = zero_list(3)
        total_err_b = zero_list(1)
        cost = 0.0
        sample_num = 0

        for row in train_df.values:
            x, y, y_ = predict_y(row, norm_avg, norm_max, w, b)
            total_err_w += x * (y_ - y)
            total_err_b += y_ - y
            cost += np.power((y_ - y), 2)
            sample_num += 1
        cost /= sample_num
        gradient_w = total_err_w / sample_num
        gradient_b = total_err_b / sample_num
        history_gradient_square_w += np.square(gradient_w)
        history_gradient_square_b += np.square(gradient_b)
        learning_rate_w = base_learning_rate / np.sqrt(history_gradient_square_w + eps)
        learning_rate_b = base_learning_rate / np.sqrt(history_gradient_square_b + eps)
        w = w - learning_rate_w * gradient_w
        b = b - learning_rate_b * gradient_b

        if i % 10 == 0:
            print 'step=', i, 'cost=', cost[0]
            # print 'w=', w
            # print 'b=', b
            # print 'norm_avg=', norm_avg
            # print 'norm_max=', norm_max
            test(w, b, norm_avg, norm_max)

        # plt.plot(i, cost[0], 'ro')

    test(w, b, norm_avg, norm_max)
    # plt.show()


def test(w, b, norm_avg, norm_max):
    test_df = pd.read_csv('../input/test_predict.csv')
    save = pd.DataFrame(columns=['Prediction'])
    save.index.name = 'Id'
    for row in test_df.values:
        id = int(row[0])
        row = [row[1], row[2], row[3], 0]
        x, y, y_ = predict_y(row, norm_avg, norm_max, w, b)
        real_y = reverse(y_, norm_avg, norm_max)
        save.loc[id] = [real_y[0]]
    save.to_csv('result.csv')


if __name__ == '__main__':
    train()
    # test()






