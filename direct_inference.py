#!/usr/bin/env python

import os
import time
from loguru import logger
import importlib
import pickle
import cv2

import torch

# import rospy
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from vision_msg.msg import human_detection_result
import sys
sys.path.append("..")

from exps.default import yolox_nano
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

# # CV bridge : OpenCV 와 ROS 를 이어주는 역할 
# bridge = CvBridge()

# # initialize result publisher
# result_pub = rospy.Publisher('human_detection_result', human_detection_result, queue_size=10)


class Predictor(object):
    def __init__(self, model, exp, class_names = COCO_CLASSES, fp16 = False,
    device = 'cpu', legacy = False):
        self.model = model
        self.cls_name = class_names
        self.num_class = exp.num_classes
        self.confthre = 0.5
        self.nmsthre = 0.45
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy = legacy)
        self.test_size = exp.test_size


    def inference(self, img):
        # img_info dictionary 작성 

        img_info = {'id' : 0}
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        ratio = min(self.test_size[0] / img.shape[0] , 
                self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0) # batch 차원 추가
        img = img.float()
        img.cpu() # cpu로 처리
        ### 참고 : apex를 이용해서 fp16을 사용하면 연산속도가 빨라진다고 함 -> 나중에 공부하기### 

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_class, self.confthre, self.nmsthre,class_agnostic=True
            )
        return outputs, img_info


    def visual(self, output, img_info, cls_conf = 0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_name)
        return vis_res



def web_cam(predictor):
    

    t0 = time.time()
    c = cv2.VideoCapture(0)
    c.set(3,640)
    c.set(4,480)
    
    # for i in range(4):
    #     c.grab()
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    rett, frame = c.read()
    
    


    outputs, img_info = predictor.inference(frame)

    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    
    



    cv2.putText(frame, "fps : {}".format(1.0 / (time.time()-t0)), (10,20), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255,0,0), 2)
    cv2.imshow('YOLOX_NANO', result_image)

    cv2.waitKey(1)


if __name__=='__main__':

    print('start')

    # Set model 
    
    exp = yolox_nano.Exp()
    model = exp.get_model()
    model.eval()
    ckpt = torch.load('/home/seojungin/Desktop/YOLOX/90epoch_ckpt.pth', map_location = 'cpu')
    model.load_state_dict(ckpt["model"])
    
    # create object predicting results
    predictor = Predictor(model, exp, COCO_CLASSES)
 
    
    while True:
        web_cam(predictor)
        
    cv2.destroyAllWindows()
    
