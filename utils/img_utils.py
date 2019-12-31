#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/22
"""
import cv2
import os
import json
import numpy as np

from utils.project_utils import check_np_empty


def convert_transparent_png(image_4channel):
    """
    将PNG图像转换为白底图像
    :param image_4channel: 4通道的PNG图像
    :return: 白底图像
    """
    # image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 128

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def add_alpha_channel(img_opencv):
    """
    添加alpha通道
    """
    rgba_alpha = cv2.cvtColor(img_opencv, cv2.COLOR_RGB2RGBA)
    h, w, _ = img_opencv.shape
    alpha_channel = np.ones((h, w)) * 255
    rgba_alpha[:, :, 3] = alpha_channel
    return rgba_alpha


def init_vid(vid_path):
    """
    初始化视频
    """
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, n_frame, w, h


def decode_gif_file(filename):
    """
    解码gif文件
    """
    frames = []

    if not os.path.exists(filename) or not filename.endswith('gif'):  # 文件不存在
        print('[Warning] Gif文件不存在: {}'.format(filename))
        return frames

    cap, fps, w, h = init_vid(filename)
    print('[Info] Gif 视频尺寸: {}'.format((h, w)))

    n_frame = 1000

    for i in range(0, n_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if check_np_empty(frame):
            print('[Info] Gif 读取停止, 帧数: {}'.format(i))
            break

        # print('[Info] frame shape: {}'.format(frame.shape))

        frames.append(frame)

    cap.release()
    return frames


def get_min_max_points(x_list, y_list):
    """
    获取像素点的边界，where的输出，左上角和右下角
    :param x_list: x序列
    :param y_list: y序列
    :return: 左上角和右下角
    """
    x_min = np.min(x_list)
    x_max = np.max(x_list)
    y_min = np.min(y_list)
    y_max = np.max(y_list)

    return (x_min, y_min), (x_max, y_max)


def remove_up_down_trans(img_png, img_draw):
    h, w, _ = img_draw.shape
    img_alpha = img_draw[:, :, 3]
    x_list, y_list = np.where(img_alpha > 20)
    (x_min, y_min), (x_max, y_max) = get_min_max_points(x_list, y_list)
    img_draw_crop = img_draw[x_min:x_max, :]
    img_crop = img_png[x_min:x_max, :]

    return img_crop, img_draw_crop


def show_png(img_png):
    """
    展示PNG图像
    """
    import matplotlib.pyplot as plt

    img_show = cv2.cvtColor(img_png, cv2.COLOR_BGRA2RGBA)
    plt.imshow(img_show)
    plt.show()


def show_img_opencv(img_opencv):
    """
    显示OpenCV图像
    """
    import matplotlib.pyplot as plt

    img_show = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show)
    plt.show()


def read_config(img_config_path):
    """
    读取Json配置，包含图像的点信息
    :param img_config_path: 图像配置路径
    :return: 图像配置Dict
    """
    with open(img_config_path) as json_file:
        data = json.load(json_file)
    # print('[Info] head_point: {}'.format(data['head_point'])) # 测试
    return data


def get_skt_parameters(head_r, neck_p, rhip_p, lhip_p):
    """
    获取身体参数
    """
    body_ex = int(min(rhip_p[0], lhip_p[0]) + abs(rhip_p[0] - lhip_p[0]) / 2)
    body_ey = int(min(rhip_p[1], lhip_p[1]) + abs(rhip_p[1] - lhip_p[1]) / 2)
    head_p = neck_p[0], neck_p[1] - head_r
    return head_p, head_r, (body_ex, body_ey)


def get_skt_min_and_max(o_skt, head_r):
    """
    获取骨骼的小坐标和最大坐标
    :param skt: 骨骼
    :param head_r: 脑袋半径
    :return: 最小坐标骨骼和最大坐标骨骼
    """
    skt = []
    for s in o_skt:
        if s == (0, 0):  # 特殊点不做处理
            continue
        skt.append(s)

    # 头部的最小值和最大值
    head_p, head_r, body_ep = get_skt_parameters(head_r, skt[1], skt[2], skt[3])
    head_min = (head_p[0] - head_r, head_p[1] - head_r)
    head_max = (head_p[0] + head_r, head_p[1] + head_r)

    skt_all = skt + [head_min, head_max]

    x_list = np.array([s[0] for s in skt_all])
    y_list = np.array([s[1] for s in skt_all])

    x_min, y_min = np.min(x_list), np.min(y_list)
    x_max, y_max = np.max(x_list), np.max(y_list)

    skt_min, skt_max = (x_min, y_min), (x_max, y_max)  # 骨骼的最大值和最小值

    return skt_min, skt_max


def get_template_min_and_max(skt_list, head_r):
    """
    获取一组骨骼的最小值和最大值，半径
    :param skt_list: 一组骨骼
    :param head_r: 头部半径
    :return: 最小值点和最大值点
    """
    min_x, min_y = [], []
    max_x, max_y = [], []
    for skt in skt_list:
        skt_min, skt_max = get_skt_min_and_max(skt, head_r)  # 获取骨骼的最小值和最大值
        min_x.append(skt_min[0])
        min_y.append(skt_min[1])
        max_x.append(skt_max[0])
        max_y.append(skt_max[1])

    min_p = (int(np.min(np.array(min_x))), int(np.min(np.array(min_y))))
    max_p = (int(np.max(np.array(max_x))), int(np.max(np.array(max_y))))
    return min_p, max_p


def draw_box(img_bkg, p_min, p_max, color=(0, 0, 255), thickness=5):
    """
    绘制正方形
    :param img_bkg: 背景
    :param p_min: 左上点
    :param p_max: 右下点
    :param color: 默认颜色
    :param thickness: 默认线宽
    :return: None
    """
    p_lu = p_min
    p_ru = (p_max[0], p_min[1])
    p_ld = (p_min[0], p_max[1])
    p_rd = p_max

    cv2.line(img_bkg, p_lu, p_ru, color=color, thickness=thickness)
    cv2.line(img_bkg, p_lu, p_ld, color=color, thickness=thickness)
    cv2.line(img_bkg, p_ru, p_rd, color=color, thickness=thickness)
    cv2.line(img_bkg, p_ld, p_rd, color=color, thickness=thickness)
