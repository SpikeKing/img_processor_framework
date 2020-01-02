#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/12/27
"""
import copy
import os
import sys
import numpy as np
import torch

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from demo_dir.demo import get_parser, setup_cfg
from demo_dir.predictor import VisualizationDemo
from root_dir import DATA_DIR, ROOT_DIR
from utils.img_utils import init_vid, show_img_opencv
from utils.project_utils import get_current_time_str


class VideoProcessor(object):
    """
    视频处理类
    """

    def __init__(self):
        self.demo, self.cfg = self.init_demo()

    def init_demo(self):
        """
        初始化模型
        """
        args = get_parser().parse_args()
        args.config_file = os.path.join(ROOT_DIR, "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        args.opts = [
            "MODEL.WEIGHTS",
            "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
            "MODEL.DEVICE",
            "cpu"]
        # "cuda"]

        cfg = setup_cfg(args)
        model = VisualizationDemo(cfg)
        return model, cfg

    def analyze_predictions(self, predictions, img_opencv):
        """
        处理mask
        """
        cpu_device = torch.device("cpu")

        instances = predictions["instances"].to(cpu_device)
        img_size = instances.image_size
        # print('[Info] 图像尺寸: {}'.format(img_size))

        # img_black = np.zeros((*img_size, 3), dtype=np.uint8)
        img_cartoon = self.img_cartoonize(img_opencv)

        pred_masks = instances.pred_masks
        pred_classes = instances.pred_classes

        # 合并所有人的Mask
        all_persons_mask = np.zeros(img_size, dtype=bool)
        for p_class, p_mask in zip(pred_classes, pred_masks):
            p_mask_np = p_mask.numpy()
            if p_class == 0:  # 人的类别
                all_persons_mask = all_persons_mask | p_mask_np

        # Mask转换为图像Mask
        all_persons_mask = np.expand_dims(all_persons_mask, axis=2)
        img_persons_mask = np.tile(all_persons_mask, 3)

        # 只包含人的图像
        img_persons = np.where(img_persons_mask == 1, img_cartoon, img_opencv)

        # show_img_opencv(img_persons)
        # img_path = os.path.join(DATA_DIR, 'videos', 'test.xxx.jpg')
        # cv2.imwrite(img_path, img_persons)

        return img_persons

    def img_cartoonize(self, image_opencv):
        """
        卡通化
        """
        num_down = 1
        num_bilateral = 1

        img_color = copy.deepcopy(image_opencv)

        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)

        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=3, sigmaColor=9, sigmaSpace=7)

        # upsample image to original size
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        # STEP 2 & 3
        # Use median filter to reduce noise
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        # STEP 4
        # Use adaptive thresholding to create an edge mask
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=3,
                                         C=2)

        # Step 5
        # Combine color image with edge mask & display picture
        # convert back to color, bit-AND with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)  # 线条
        # img_cartoon = cv2.bitwise_and(img_color, img_edge)   # 颜色

        # show_img_opencv(image_opencv)
        # show_img_opencv(img_cartoon)
        return img_edge

    def process_img(self, img_opencv):
        """
        处理图像
        """
        predictions, visualized_output = self.demo.run_on_image(img_opencv)
        img_out = self.analyze_predictions(predictions, img_opencv)
        # img_rgb = visualized_output.get_image()
        return img_out

    def process_video(self, vid_path, out_vid_path):
        """
        读取视频
        """
        print('[Info] 输入视频路径: {}'.format(vid_path))
        cap, fps, n_frame, w, h = init_vid(vid_path)
        print('[Info] 输入视频 PFS: {}, 帧数: {}, 宽高: {} {}'.format(fps, n_frame, w, h))

        n_vid_fps = 25
        n_vid_frame = 1000

        print('[Info] 输出帧数 PFS: {}, 帧数: {}'.format(n_vid_fps, n_vid_frame))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vw = cv2.VideoWriter(filename=out_vid_path, fourcc=fourcc, fps=n_vid_fps, frameSize=(w, h), isColor=True)

        for i in range(0, n_frame):
            if i == n_vid_frame:  # 特定帧数停止
                break

            print('[Info] frame processed: {} / {}'.format(i, n_vid_frame))

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            img_bgr = self.process_img(frame)
            # img_bgr = img_rgb[:, :, ::-1]

            vw.write(img_bgr)

        vw.release()
        print('[Info] 处理视频完成!')


def img_processor_test():
    vp = VideoProcessor()

    # 处理视频
    vid_path = os.path.join(DATA_DIR, 'videos', 'test.mp4')
    out_vid_path = os.path.join(DATA_DIR, 'videos', 'test.out.{}.mp4'.format(get_current_time_str()))
    vp.process_video(vid_path, out_vid_path)

    # 处理图像
    # tmp_img = os.path.join(DATA_DIR, 'videos', 'test.jpg')
    # out_img = os.path.join(DATA_DIR, 'videos', 'test.out.jpg')
    # img_opencv = cv2.imread(tmp_img)
    # img_rgb = vp.process_img(img_opencv)
    # img_opencv = img_rgb[:, :, ::-1]
    # cv2.imwrite(out_img, img_opencv)


def main():
    img_processor_test()


if __name__ == '__main__':
    main()
