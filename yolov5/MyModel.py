import time
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device

import logging
import zipfile
import paho.mqtt.client as paho
from uuid import uuid4
import numpy as np
import json
from threading import Thread
import base64


class MyModel:
    def __init__(self, app):

        self.app = app

        self.fps = -1
        self.current_frame_time = -1

        self.source = None
        self.cap = None
        self.thread = None

        self.skip_frames = False
        self.continue_running = False
        self.id = str(uuid4())

        self.weights = 'best2.pt'
        self.imgsz = 512
        self.conf_thres = 0.4
        self.iou_thres = 0.45


        logging.info('creating session')



    def load_ai_model(self, filename):
        #tole je brezveze
        logging.info('File downloaded, extacting...')
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall("/")
        zip_ref.close()

        # Initialize
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

        return 0

    def ai_thread(self, *args):
        def on_connect(client, userdata, flags, rc):
            print("CONNECT RECEIVED with code %d." % (rc))

        def on_publish(client, userdata, mid):
            print("PUBLISHED")

        def on_message(client, userdata, message):
            print("message received ", str(message.payload.decode("utf-8")))
            print("message topic=", message.topic)
            print("message qos=", message.qos)
            print("message retain flag=", message.retain)

        client = paho.Client(transport="websockets")
        client.on_connect = on_connect
        client.on_publish = on_publish
        # client.on_message = on_message
        client.connect(self.app.appconfig.get_destination()['mqtt'].hostname,
                                self.app.appconfig.get_destination()['mqtt'].port)
        client.loop_start()

        # if you need to check the mqtt data send to prometheus you need to uncomment this an on_message line
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_fps")
        # client.subscribe("prometheus/job/AI_metrics/instance/yolov3/monitoring_video_delay")
        logging.info('opening source at: ' + self.source)

        # AXIS ip camera MJPEG
        self.cap = cv2.VideoCapture(self.source)
        print(self.cap)
        self.set_minimum_fps(args[0])
        self.current_frame_time = time.time()
        prometheus_time = time.time()

        while self.continue_running:
            # skip frames until the delay is smaller than the value specified
            if self.skip_frames:
                self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)

                if self.get_current_video_delay() < 0.2:
                    self.skip_frames = False
                    self.current_frame_time = time.time()
            else:
                try:
                    ret, frame_ori = self.cap.read()
                    start_time = time.time()  # start time of the loop
                    print()

                    if ret:

                        # Padded resize
                        img = letterbox(frame_ori, self.imgsz, stride=self.stride)[0]

                        # Convert
                        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        img = np.ascontiguousarray(img)

                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        # Inference
                        pred = self.model(img)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                        # Process detections
                        detections = []
                        for i, det in enumerate(pred):  # detections per image
                            gn = torch.tensor(frame_ori.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_ori.shape).round()

                                # Write results
                                for *xyxy, conf, cls in reversed(det):

                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh

                                    detections.append({'class': self.names[int(cls)],
                                                       'location': xywh,
                                                       'score': float(conf)})

                                    # plot on image
                                    c = int(cls)  # integer class
                                    label =  f'{self.names[c]} {conf:.2f}'
                                    plot_one_box(xyxy, frame_ori, label=label, color=colors(c, True),
                                                 line_thickness=3)

                        # show image with predictions
                        # cv2.imshow("image", frame_ori)
                        # cv2.waitKey(1)

                        retval, buffer = cv2.imencode('.jpg', frame_ori)
                        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                        message = {'ai_id': self.id,
                                   'fps': self.fps,
                                   'delay': self.get_current_video_delay(),
                                   'timestamp': str(self.current_frame_time),
                                   'detections': detections,
                                   'encoded_image': jpg_as_text}
                        message = json.dumps(message)
                        print(message)

                        """
                        for i in range(len(boxes_)):
                            x0, y0, x1, y1 = boxes_[i]
                            plot_one_box(frame_ori, [x0, y0, x1, y1], label=self.classes[labels_[i]], color=self.color_table[labels_[i]])

                        frame_out = cv2.imencode('.jpg', frame_ori)[1].tostring()
                        """

                        infot = client.publish(str(self.app.appconfig.get_destination()["mqtt"].path)[1:], message, qos=0)
                        infot.wait_for_publish()

                        # send data to prometheus every 2 seconds
                        if time.time() - prometheus_time > 1:
                            print("publishing to prometeus current FPS")
                            infot = client.publish("prometheus/job/AI_metrics/instance/yolov5/monitoring_fps",
                                                   self.get_current_fps())
                            infot.wait_for_publish()
                            infot = client.publish("prometheus/job/AI_metrics/instance/yolov5/monitoring_video_delay",
                                                   self.get_current_video_delay())
                            infot.wait_for_publish()
                            prometheus_time = time.time()

                        # yield(b'--frame\r\n'
                        #   b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')

                        self.fps = 1.0 / (time.time() - start_time)
                        self.reset_video_feed()
                        if self.cap.isOpened():
                            self.current_frame_time += 1 / self.cap.get(cv2.CAP_PROP_FPS)
                    else:
                        logging.info("NO DATA!")
                        break
                except Exception as ex:
                    print(f"An exception occured while processing a frame: {type(ex)}: {ex}")

        client.disconnect()
        self.cap.release()
        return "DONE"

    def get_current_fps(self):
        if not self.continue_running:
            return -1
        return self.fps

    def get_current_video_delay(self):
        if not self.continue_running:
            return -1
        return time.time() - self.current_frame_time

    def set_minimum_fps(self, minimum_fps):
        if self.continue_running and 60 >= minimum_fps != self.cap.get(cv2.CAP_PROP_FPS) and minimum_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, minimum_fps)
            return True
        return False

    def reset_video_feed(self):
        if self.get_current_video_delay() > 1:
            print("Delay was grater then the limit, so video feed was reset to align with the current stream")
            self.skip_frames = True

    def start_thread(self, fps):
        if not self.continue_running:
            print("starting AI computations")
            self.thread = Thread(target=self.ai_thread, args=[fps])
            self.continue_running = True
            self.thread.start()
            return "STARTED"
        return "ALREADY RUNNING"

    def stop_thread(self):
        print("Stopping AI computations")
        if self.continue_running:
            self.continue_running = False
            self.fps = -1
            self.current_frame_time = - 1
            self.cap.release()
            return "AI STOPPED"
        return "AI WAS NOT RUNNING"

    def compute_ai(self, *args):
        self.source = ''.join(args[0])
        # if we would need auto start of the AI model
        self.start_thread(30)
        return {}


