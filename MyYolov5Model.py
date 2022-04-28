import os
import time
from pathlib import Path
import ntpath

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from DenseNet.MyDensenetModel import *

class MyYolov5Model:
    def __init__(self):
        # select model
        self.weights = "./weights/fp60_best.pt"
        self.img_size = 640
        # set iou and conf thres
        self.iou_thres = 0.45
        self.conf_thres = 0.65

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # one stage - Detect the insects in the picture and assign them to the biology department
    def one_stage_image_detect(self, image_path, save_dir_name="temp", save_image=False, save_label=False):
        imgsz = self.img_size

        if save_image or save_label:
            save_dir = "./result/" + save_dir_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # load the model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 models
        stride = int(model.stride.max())  # models stride
        # check img size
        imgsz = check_img_size(imgsz, s=stride)
        dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=True)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=True)
            insect_list = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                        insect_list.append([x1, y1, x2, y2, conf.item(), cls.item()])

                    if save_label:
                        # Write results
                        label_dir = "./result/" + save_dir_name + "/labels/"
                        os.mkdir(label_dir)
                        for *xyxy, conf, cls in reversed(det):
                            # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf)  # label format
                            with open(label_dir + save_dir_name + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                if save_image:
                    # Save results (image with detections)
                    image_save_path = save_dir + "/" + p.name
                    cv2.imwrite(image_save_path, im0)
        return insect_list