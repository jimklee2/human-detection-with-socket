import cv2
import io
import socket
import struct
import time
import pickle

HOST = '10.42.0.215'
PORT = 9999


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('wb')


cam = cv2.VideoCapture(0)
cam.set(3, 640);
cam.set(4, 480);
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)



img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

recvdata = b""




while True:
    t2 = time.time()

    for i in range(6):
        cam.grab()
        
    # flg, frame1 = cam.retrieve()
    flg, frame1 = cam.read()

    
    result, frame = cv2.imencode('.jpg', frame1, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
    print("Img_counter : {}".format(img_counter))
    print(frame1.shape)
    
        

    
    recvdata += client_socket.recv(1024)
    packed_msg_size = recvdata[:4]
    recvdata = recvdata[4:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(recvdata) < msg_size:
        recvdata += client_socket.recv(1024)
    frame_data = recvdata[:msg_size]
    recvdata = recvdata[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('YOLOX_NANO', frame)
    cv2.waitKey(1)
    print('Time : {}'.format(time.time() - t2))




cam.release()