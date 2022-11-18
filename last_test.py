#!/usr/bin/env python

import os
import time
from loguru import logger
import importlib

import cv2

import torch
import socket
import sys

import pickle
import numpy as np
import struct
import zlib


import sys
sys.path.append("..")

from exps.default import yolox_nano
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis


HOST='10.42.0.215'      # set your own IP
PORT=9998               # set your own PORT

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')

s.listen(10)
print('Socket now listening')





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
    conn,addr=s.accept()   


    data = b""      # b"" : byte 문자열 지정

    
    payload_size = struct.calcsize(">L") # struct.calcsize 설정
    num = 1
    
    while True:
        
        t1 = time.time()
        while len(data) < payload_size:
            
            data += conn.recv(1024)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += conn.recv(1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        outputs, img_info = predictor.inference(frame)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        t2 = time.time()
        sendresult, sendframe = cv2.imencode('.jpg', result_image)
        senddata = pickle.dumps(sendframe,0)
        size = len(senddata)
        conn.sendall(struct.pack(">L", size)+senddata)
        period = time.time()-t1
        print("Period : {}".format(period))
        print("Period : {}".format(time.time()-t2))
        num += 1
        

        


if __name__=='__main__':

    print('start')

    # Set model 
    
    exp = yolox_nano.Exp()
    model = exp.get_model()
    model.eval()
    ckpt = torch.load('/home/kist/YOLOX/90epoch_ckpt.pth', map_location = 'cpu')   # set your 90epoch_ckpt path
    model.load_state_dict(ckpt["model"])
    
    # create object predicting results
    predictor = Predictor(model, exp, COCO_CLASSES)
 
    
    while True:
        web_cam(predictor)
        
    cv2.destroyAllWindows()
    
