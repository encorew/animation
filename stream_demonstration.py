import os
from time import sleep

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import cm
from data.Preprocesser import get_data, normalization
from data.utils import create_gif, myPlots, join

sample_step = 100


class streamPlot:
    def __init__(self, raw_data, time_stamp=None, smooth=5, sample=False, model=False,
                 save_dir=None):
        raw_data = raw_data.transpose()
        self.dimension = raw_data.shape[0]
        self.MAE = None
        if not sample:
            self.raw_data = raw_data
        else:
            idx = [a for a in range(0, len(raw_data[0]), sample_step)]
            self.raw_data = raw_data[:, idx]
        self.length = self.raw_data.shape[1]
        self.model = [make_interp_spline(range(self.length), self.raw_data[i]) for i in range(self.dimension)]
        if not time_stamp.any():
            time_stamp = ['T22:' + str(i // 60) + ':' + str(i % 60) for i in range(self.length)]
        self.time_ticks = (range(self.length), time_stamp)
        # self.time_ticks = [(i, time_stamp[i]) for i in range(self.length)]
        # print(self.time_ticks)
        self.smooth = smooth
        self.num_total_points = (self.length - 1) * smooth + 1
        self.X = np.linspace(0, self.length - 1, self.num_total_points)
        self.Y = np.arange(0, self.dimension, 1)
        self.angle1 = 25
        self.angle2 = -25
        self.rotation_status = 0
        self.values = np.array([self.model[i](self.X) for i in range(self.dimension)])
        self.save_dir = save_dir
        # plt.plot(self.X,self.values[0])
        # plt.show()

    def get_current_window_information(self, window_size, window_idx):
        current_window_point_X = self.X[window_idx * smooth:(window_idx + window_size) * smooth]
        current_window_point_Z = self.values[:, window_idx * smooth:(window_idx + window_size) * smooth]
        current_time_ticks = (self.time_ticks[0][window_idx + 1:window_idx + window_size],
                              self.time_ticks[1][window_idx + 1:window_idx + window_size])
        return current_window_point_X, current_window_point_Z, current_time_ticks

    def initialize_font(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def adjust_angle(self):
        if self.rotation_status == 0:
            self.angle2 += 1
            if self.angle2 >= 70:
                self.rotation_status = 1
        elif self.rotation_status == 1:
            self.angle2 -= 1
            self.angle1 += 1
            if self.angle1 >= 85:
                self.rotation_status = 2
        elif self.rotation_status == 2:
            self.angle2 -= 1
            self.angle1 -= 1
            if self.angle2 <= -70 or self.angle1 <= 20:
                self.rotation_status = 0

    def construct_figure_details(self, ax):
        ax.set_ylabel('性能指标')
        ax.set_xlabel('时间')
        self.adjust_angle()
        ax.view_init(self.angle1, self.angle2)
        # plt.show()
        if self.save_dir == real_data_dir:
            #     f'timestamp:{self.time_ticks[1][idx + window_size - 1]}  预测误差MAE:{self.MAE[idx] / 10:.4f}\n\n服务器负载观测值')
            ax.set_title(f'\n服务器负载观测值')
        elif self.save_dir == predict_data_dir:
            ax.set_title(f'\n服务器负载预测值')
        ax.set_zlim3d(0, 1)

    def rotation_view(self, ax, num_frames=18, initial_angle1=32, initial_angle2=-32, rotation_angle1=0.1,
                      rotation_angle2=-0.1):
        stay = 13
        rotation_gap1 = (rotation_angle1 - initial_angle1) / num_frames
        rotation_gap2 = (rotation_angle2 - initial_angle2) / num_frames
        for i in range(num_frames):
            ax.view_init(initial_angle1 + rotation_gap1 * i, initial_angle2 + rotation_gap2 * i)
            plt.savefig(f"{self.save_dir}rotation-{i}")
        for i in range(stay):
            plt.savefig(f"{self.save_dir}rotation-{i + num_frames}")
        for i in range(num_frames):
            ax.view_init(rotation_angle1 - rotation_gap1 * i, rotation_angle2 - rotation_gap2 * i)
            plt.savefig(f"{self.save_dir}rotation-{i + stay + num_frames}")

    def animation(self, window_size):
        self.initialize_font()
        num_one_window_point = (window_size - 1) * self.smooth + 1
        fig = plt.figure()
        for idx in range(duration):
            current_sliding_window_point_X, current_sliding_window_point_Z, current_time_ticks = self.get_current_window_information(
                window_size, idx)
            for i in range(self.smooth):
                current_animation_show_X = current_sliding_window_point_X[i:i + num_one_window_point]
                current_animation_show_Z = current_sliding_window_point_Z[:, i:i + num_one_window_point]
                ax = fig.gca(projection='3d')
                current_animation_show_X, current_animation_show_Y = np.meshgrid(current_animation_show_X, self.Y)
                ax.plot_surface(current_animation_show_X, current_animation_show_Y, current_animation_show_Z,
                                cmap=cm.coolwarm, shade=False, alpha=0.8)
                # ax.plot3D(current_animation_show_X, current_animation_show_Y, current_animation_show_Z, 'gray')
                self.construct_figure_details(ax)
                ax.set_xticks(current_time_ticks[0])
                ax.set_xticklabels(current_time_ticks[1])
                print(f'save-{idx}-{i}')
                plt.savefig(f"{self.save_dir}{idx}-{i}")
                # 这行代码很重要，plt.clf()
                # if idx == 0 and i == 0:
                #     self.rotation_view(ax)
                plt.clf()


def easy_make_gif(image_path, save_path="animation/gifs", gif_name="myGif0.gif"):
    # print(os.listdir(image_path))
    names = os.listdir(image_path)
    names = [os.path.join(image_path, name) for name in names]
    print(names)
    print(os.path.join(save_path, gif_name))
    create_gif(names, os.path.join(save_path, gif_name))


def make_gif(image_path, save_path="animation/gifs", gif_name="myGif0.gif"):
    image_list = []
    # for i in range(num_rotation_frames):
    #     file = os.path.join(image_path, f"rotation-{i}.png")
    #     image_list.append(file)
    for i in range(duration):
        for j in range(smooth):
            file = os.path.join(image_path, f"{i}-{j}.png")
            image_list.append(file)
    print("start create")
    create_gif(image_list, os.path.join(save_path, gif_name))


def combine(image_path1, image_path2, save_path):
    names = os.listdir(image_path1)
    for name in names:
        print(f'join{name}')
        png1_path = os.path.join(image_path1, name)
        png2_path = os.path.join(image_path2, name)
        join(png1_path, png2_path, save_path, name)


def join(png1_path, png2_path, save_path, save_name, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = Image.open(png1_path), Image.open(png2_path)
    # # 统一图片尺寸，可以自定义设置（宽，高）
    # img1 = img1.resize((1500, 1000), Image.ANTIALIAS)
    # img2 = img2.resize((1500, 1000), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1] + 100))
        loc1, loc2 = (0, 0), (size1[0], 0)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
    white_img = Image.new("RGB", (size1[0] + size2[0], 100), (255, 255, 255))
    joint.paste(white_img, (0, 0))
    joint.paste(img2, (0, 100))
    joint.paste(img1, (size1[0], 100))
    joint.save(os.path.join(save_path, save_name))


def stick_word(file, str, position, show=False):
    img = Image.open(file)
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
    draw = ImageDraw.Draw(img)
    draw.text(position, str, fill='black', font=font)
    if show:
        img.show()
    # sleep(10000)
    print(file)
    img.save(file)


def overall_stick_word(dir, MAE, time):
    str1 = """
        数据集Server Machine Data是一个大型互联网公司的数据集，在每个时刻记录了服务器的38维性能指标
        """
    position1 = (60, 10)
    str2 = """
        左图记录了每一时刻服务器指标的真实观测值，右图为预测算法得到的每一时刻指标预测值，使用MAE来评估预测误差
        """
    position2 = (60, 35)
    position3 = (590, 260)
    for i in range(num_rotation_frames):
        file = os.path.join(dir, f"rotation-{i}.png")
        stick_word(file, str1, position1)
    for i in range(duration):
        for j in range(smooth):
            file = os.path.join(dir, str(i) + '-' + str(j) + ".png")
            stick_word(file, str1, position1)
            stick_word(file, str2, position2)
            stick_word(file, f"当前时间:{time[i + 4]}\n预测误差MAE:{MAE[i + 4]:.4f}", position3)


num_rotation_frames = 49
duration = 60
image_save_dir = "animation/stream_pics"
gif_save_dir = "animation/gifs"
predict_data_dir = "animation/stream_pics/predict/"
real_data_dir = "animation/stream_pics/real/"
combine_dir = "animation/stream_pics/combine/"
business_data_dir = "animation/business_pics/real/"
smooth = 10

if __name__ == '__main__':
    # real_data = get_data('data/SMD_test')
    business_data = get_data('data/integrated', ".csv")
    time, raw_data = business_data[:, 0].astype(int), normalization(business_data[:, 1:])
    my_plot = streamPlot(raw_data, time_stamp=time, sample=False, smooth=smooth, save_dir=business_data_dir)
    my_plot.animation(5)
    # print(time, raw_data)
    # predict_data = np.load('data/prediction_for_384 e81 W50 D0.3 620.npy')
    # abnormal_index = np.array(np.where(real_data[0, :] > 0.25))
    # real_data[:, abnormal_index] /= 10
    # predict_data[:, abnormal_index] /= 10
    # time_stamp = ['T22:' + str(i // 60) + ':' + str(i % 60) for i in range(len(real_data))]
    # idx = [a for a in range(0, real_data.shape[0], sample_step)]
    # MAE = np.sum(np.absolute(real_data[idx, :] - predict_data[idx, :]), axis=1) / 38
    # my_plot = streamPlot(real_data, sample=True, smooth=smooth, save_dir=real_data_dir)
    # my_plot.animation(5)
    # predict_plot = streamPlot(predict_data, sample=True, smooth=smooth, save_dir=predict_data_dir)
    # predict_plot.animation(5)
    # combine(predict_data_dir, real_data_dir, combine_dir)
    # overall_stick_word(combine_dir, MAE, time_stamp)
    # make_gif(combine_dir, gif_name="version2.gif")
