#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/12/27
"""

import os

import cv2
import numpy as np

from demo.demo import get_parser, setup_cfg
from demo.predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from root_dir import DATA_DIR, ROOT_DIR
from utils.img_utils import init_vid


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
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        return demo, cfg

    def process_img(self, img_opencv):
        """
        处理图像
        """
        predictions, visualized_output = self.demo.run_on_image(img_opencv)
        img_rgb = visualized_output.get_image()
        return img_rgb

    def process_video(self, vid_path, out_vid_path):
        """
        读取视频
        """
        cap, fps, n_frame, w, h = init_vid(vid_path)

        n_vid_frame = 10
        n_vid_fps = 5

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vw = cv2.VideoWriter(filename=out_vid_path, fourcc=fourcc, fps=n_vid_fps, frameSize=(w, h), isColor=True)

        for i in range(0, n_frame):

            if i == n_vid_frame:
                break
            print('[Info] frame processed: {}'.format(i))

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            img_rgb = self.process_img(frame)
            img_bgr = img_rgb[:, :, ::-1]

            vw.write(img_bgr)

        vw.release()

        print('[Info] 处理视频完成!')


def img_processor_test():
    vid_path = os.path.join(DATA_DIR, 'videos', 'test.mp4')
    out_vid_path = os.path.join(DATA_DIR, 'videos', 'test.out.mp4')
    # tmp_img = os.path.join(DATA_DIR, 'videos', 'test.jpg')
    # out_img = os.path.join(DATA_DIR, 'videos', 'test.out.jpg')

    vp = VideoProcessor()
    vp.process_video(vid_path, out_vid_path)


def main():
    img_processor_test()


if __name__ == '__main__':
    main()
