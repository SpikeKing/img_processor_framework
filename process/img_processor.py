#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/12/27
"""

import os
import sys

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from demo_dir.demo import get_parser, setup_cfg
from demo_dir.predictor import VisualizationDemo
from root_dir import DATA_DIR, ROOT_DIR
from utils.img_utils import init_vid
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
            # "cpu"]
            "cuda"]

        cfg = setup_cfg(args)
        model = VisualizationDemo(cfg)
        return model, cfg

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
        print('[Info] 输入视频路径: {}'.format(vid_path))
        cap, fps, n_frame, w, h = init_vid(vid_path)
        print('[Info] 输入视频 PFS: {}, 帧数: {}, 宽高: {} {}'.format(fps, n_frame, w, h))

        n_vid_fps = 25
        n_vid_frame = 4000

        print('[Info] 输出帧数 PFS: {}, 帧数: {}'.format(n_vid_fps, n_vid_frame))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vw = cv2.VideoWriter(filename=out_vid_path, fourcc=fourcc, fps=n_vid_fps, frameSize=(w, h), isColor=True)

        for i in range(0, n_frame):
            if i == n_vid_frame:  # 特定帧数停止
                break

            print('[Info] frame processed: {} / {}'.format(i, n_vid_frame))

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            img_rgb = self.process_img(frame)
            img_bgr = img_rgb[:, :, ::-1]

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
