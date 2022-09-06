import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from data.Preprocesser import get_data


class MyPlot:
    def __init__(self, X, Y, predict_Y):
        self.num_points = len(X)
        self.original_Y = Y
        self.predict_Y = predict_Y
        if isinstance(X[0], int) or isinstance(X[0], float):
            self.original_X = X
        else:
            self.original_X = range(len(Y))
        self.ticks = (self.original_X, X)
        self.model = make_interp_spline(self.original_X, self.original_Y)
        self.predict_model = make_interp_spline(self.original_X, self.predict_Y)
        self.show_window = None

    def simple_draw(self, x_limit=None, y_limit=None, save_name=None):
        plt.plot(self.original_X, self.original_Y)
        if x_limit:
            plt.xlim(x_limit)
        if y_limit:
            plt.ylim(y_limit)
        plt.show()

    def model_draw(self, x_limit=None, y_limit=None, save_name=None):
        pass

    def one_window_animation(self, current_window_X, current_window_Y, predict_window_Y, window_size, window_id):
        one_window_pic_nums = len(current_window_X) - window_size
        for i in range(one_window_pic_nums):
            plt.grid(ls="--", lw=0.5, color="#4E616C")
            plt.title(u"服务器负载数据与预测")
            plt.plot(current_window_X[i:i + window_size], current_window_Y[i:i + window_size])
            plt.plot(current_window_X[i:i + window_size], predict_window_Y[i:i + window_size])
            plt.fill_between(x=current_window_X[i:i + window_size], y1=current_window_Y[i:i + window_size], y2=0,
                             alpha=0.5)
            plt.scatter(self.original_X[window_id + 1:window_id + self.show_window],
                        self.original_Y[window_id + 1:window_id + self.show_window], color='blue')
            plt.scatter(self.original_X[window_id + 1:window_id + self.show_window],
                        self.predict_Y[window_id + 1:window_id + self.show_window])
            plt.ylim(0.1, 0.2)
            print(self.ticks[0][window_id:window_id + self.show_window])
            print(self.ticks[1][window_id:window_id + self.show_window])
            plt.xticks(self.ticks[0][window_id + 1:window_id + self.show_window],
                       self.ticks[1][window_id + 1:window_id + self.show_window], rotation=90, fontsize=10)
            plt.legend(["真实数据", "预测数值"], loc="upper left")
            plt.pause(0.01)
            plt.clf()

    def overall_animation(self, show_window, smooth):
        self.show_window = show_window
        animation_start_x = self.original_X[0]
        animation_end_x = self.original_X[-1]
        total_num_points = (self.num_points - 1) * smooth + 1
        animation_X = np.linspace(animation_start_x, animation_end_x, total_num_points)
        animation_Y = self.model(animation_X)
        animation_predict_Y = self.predict_model(animation_X)
        actual_show_num_points = (show_window - 1) * smooth + 1
        for i in range(0, total_num_points - actual_show_num_points - 1, smooth):
            current_window_X = animation_X[i:i + actual_show_num_points + smooth]
            current_window_Y = animation_Y[i:i + actual_show_num_points + smooth]
            predict_window_Y = animation_predict_Y[i:i + actual_show_num_points + smooth]
            self.one_window_animation(current_window_X, current_window_Y, predict_window_Y, actual_show_num_points,
                                      i // smooth)


# X = np.random.randint(0,10,size=100)
# Y = np.random.randint(1, 10, size=1000)
# predict_Y = np.random.randint(1, 8, size=1000)
Y = get_data('data/SMD_test')[105:,0]
predict_Y = np.load('data/prediction_for_384 e81 W50 D0.3 620.npy')[105:,0]
X = ['2022-02-07T22:' + str(i // 60) + ':' + str(i % 60) for i in range(len(Y))]
myplot = MyPlot(X, Y, predict_Y)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
myplot.overall_animation(10, 20)
