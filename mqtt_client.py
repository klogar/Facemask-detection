import random
import time

from paho.mqtt import client as mqtt_client
import base64
import json
import cv2
import numpy as np


# broker = '7203864f936b45518a401c7e7c24dbf3.s1.eu.hivemq.cloud'
# port = 8883
broker = 'broker.emqx.io'
port = 1883
topic = "yolov5/general"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
#username = 'testuser'
#password = 'Test1234'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    #client.username_pw_set(username, password)
    client.on_connect = on_connect
    print(broker, port)
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        img = json.loads(msg.payload.decode())["encoded_image"]
        img = base64.b64decode(img)
        #img = np.asarray(bytearray(img, encoding='utf8'), dtype="uint8")
        img = np.fromstring(img, dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        cv2.imshow("detection", img)
        cv2.waitKey(1)  # 1 millisecond

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()