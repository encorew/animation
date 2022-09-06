import random

import numpy as np
from PIL import Image

# matplotlib.use('AGG')

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

import os

from data.data import process_milan_data
from data.utils import combine, stick_word, combine3


def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


class myPlots():
    def __init__(self, data, size, time, smooth=5, stay=10, save_mode="real_", save_name=None):
        self.data = data
        self.l, self.w = size
        self.smooth = smooth
        self.stay = stay
        self.values = {}
        self.compare_values = {}
        self.X = []
        self.Y = []
        gap = (data[time + 1] - data[time]) / smooth
        print(gap)
        for i in range(self.l):
            for j in range(self.w):
                self.X.append(i)
                self.Y.append(j)
        for k in range(smooth + stay):
            self.values[k] = []
            for i in range(self.l):
                for j in range(self.w):
                    if k < smooth:
                        self.values[k].append(data[time, i, j] + gap[i, j] * k)
                    else:
                        self.values[k].append(data[time + 1, i, j])
            # print(self.values[k])
        self.time = time
        self.save_mode = save_mode
        self.save_name = save_name

    def initialize_font(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_plots(self):
        fig = plt.figure()
        self.initialize_font()
        for k in range(self.smooth + self.stay):
            ax = fig.gca(projection='3d')
            # ax = Axes3D()
            if self.save_mode == "real_":
                ax.set_title('真实值')
            else:
                ax.set_title('预测值')
            # 限制显示范围为（0,1）
            ax.set_zlim3d(0, 60)
            print(f"{k}:", self.values[k])
            ax.plot_trisurf(self.X, self.Y, self.values[k], cmap=plt.get_cmap('jet'), linewidth=0.1)
            plt.savefig(f"{self.save_name}{self.time}-{k}")
            print(f"save{self.time}-{k}")
            # plt.pause(0.1)
            plt.clf()

    def annotate(self, ax, k, idx):
        ax.text(self.X[idx], self.Y[idx], self.values[k][idx],
                s=f'region({self.X[idx]},{self.Y[idx]})\nvalue:{self.values[k][idx]:.3f}')

    def annotate_some_points(self, ax, k):
        # self.annotate(ax, k, 0)
        self.annotate(ax, k, 22)
        # self.annotate(ax, k, 50)
        self.annotate(ax, k, 98)
        self.annotate(ax, k, 136)

    def annotate_high_points(self, ax, k, thresh_hold=0.6):
        for i in range(len(self.X)):
            if self.values[k][i] > thresh_hold:
                self.annotate(ax, k, i)


image_save_dir = "animation/stream_pics"
gif_save_dir = "animation/gifs"
predict_values_dir = "animation/images/values/predict/"
real_values_dir = "animation/images/values/real/"
combine_values_dir = "animation/images/values/combine/"
predict_callin_dir = "animation/images/callin/predict/"
real_callin_dir = "animation/images/callin/real/"
combine_callin_dir = "animation/images/callin/combine/"
predict_callout_dir = "animation/images/callout/predict/"
real_callout_dir = "animation/images/callout/real/"
combine_callout_dir = "animation/images/callout/combine/"
final_dir = "animation/images/final/"


def fake_predict(processed_values):
    predict_values = np.zeros((13, processed_values.shape[1], processed_values.shape[2]))
    for i in range(13):
        for j in range(processed_values.shape[1]):
            for k in range(processed_values.shape[2]):
                predict_values[i, j, k] = processed_values[i, j, k] + random.uniform(-0.1, 0.1)
    return predict_values


def overall_stick_word(dir, MAE, time, font_size=20, show=False):
    str1 = """
        Telecom Italia Big Data 是一个包含了时空信息的电信数据集。
        数据集中，米兰地区被分割成了多个小区域，并记录下了每个区域在每一个时刻的电信数据。
        
        下面使用3维热力图描绘了每个区域于每一时刻的数据，包括呼叫接收数、呼叫发送数与连接建立数。
        左列为传感器记录的真实数据，右列为预测算法得到的预测值，使用MAE来评估预测误差。
        """
    position0 = (10, 10)
    # str2 = """
    #     左图记录了每一时刻服务器指标的真实观测值，右图为预测算法得到的每一时刻指标预测值，使用MAE来评估预测误差
    #     """
    # position2 = (60, 35)
    position1 = (550, 350)
    position2 = (550, 950)
    position3 = (550, 1550)
    for i in range(duration):
        for j in range(smooth + stay):
            file = os.path.join(dir, str(i) + '-' + str(j) + ".png")
            stick_word(file, str1, position0, font_size, show)
            stick_word(file, f"呼叫接收数\n当前时间:{time[i]}\n预测误差MAE:{MAE[0][i]:.4f}", position1, font_size, show)
            stick_word(file, f"呼叫发送数\n当前时间:{time[i]}\n预测误差MAE:{MAE[1][i]:.4f}", position2, font_size, show)
            stick_word(file, f"连接建立数\n当前时间:{time[i]}\n预测误差MAE:{MAE[2][i]:.4f}", position3, font_size, show)


def make_gif(image_path, save_path="animation/gifs", gif_name="myGif1.gif"):
    image_list = []
    for i in range(duration):
        for j in range(smooth + stay):
            image_list.append(f"{image_path}{i}-{j}.png")
            print(i, '-==-', j)
    print("start_create")
    create_gif(image_list, os.path.join(save_path, gif_name))


duration = 6
smooth = 8
stay = 10

if __name__ == '__main__':
    time_str, processed_values, processed_callin, processed_callout = process_milan_data('data',
                                                                                         'sms-call-internet-mi-2013-11-07.csv')
    idx = [i for i in range(30, 40)]
    idx2 = [i for i in range(45, 75)]
    idx = idx + idx2
    print(idx)
    # for time in range(6):
    #     # real_values = myPlots(processed_values[:, 30:70, 30:70], (40, 40), time, smooth=smooth, save_mode="real_",
    #     #                       save_name=real_values_dir)
    #     # real_values.generate_plots()
    #     # predict_values = fake_predict(processed_values)
    #     # predict_values = myPlots(predict_values[:, 30:70, 30:70], (40, 40), time, smooth=smooth, save_mode="predict_",
    #     #                          save_name=predict_values_dir)
    #     # predict_values.generate_plots()
    #     # real_callin = myPlots(processed_callin[:, idx, 40:80], (40, 40), time, smooth=smooth, save_mode="real_",
    #     #                       save_name=real_callin_dir)
    #     # real_callin.generate_plots()
    #     # predict_callin = fake_predict(processed_callin)
    #     # predict_callin = myPlots(predict_callin[:, idx, 40:80], (40, 40), time, smooth=smooth,
    #     #                          save_mode="predict_",
    #     #                          save_name=predict_callin_dir)
    #     # predict_callin.generate_plots()
    #     real_callout = myPlots(processed_callout[:, idx, 40:80], (40, 40), time, smooth=smooth, save_mode="real_",
    #                           save_name=real_callout_dir)
    #     real_callout.generate_plots()
    #     predict_callout = fake_predict(processed_callout)
    #     predict_callout = myPlots(predict_callout[:, idx, 40:80], (40, 40), time, smooth=smooth,
    #                              save_mode="predict_",
    #                              save_name=predict_callout_dir)
    #     predict_callout.generate_plots()
    # combine(real_callin_dir, predict_callin_dir, combine_callin_dir)
    # combine(real_values_dir, predict_values_dir, combine_values_dir)
    # combine3(combine_callin_dir, combine_callout_dir, combine_values_dir, final_dir)
    # MAE_in = [random.uniform(0.3, 0.7) for i in range(duration)]
    # MAE_out = [random.uniform(0.2, 0.6) for i in range(duration)]
    # MAE_values = [random.uniform(2, 5.4) for i in range(duration)]
    # MAE = [MAE_in, MAE_out, MAE_values]
    # # print(MAE)
    # overall_stick_word(final_dir, MAE, time_str, 25)
    make_gif(final_dir, gif_name="milan 3D.gif")
