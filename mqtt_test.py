import random
import time

from paho.mqtt import client as mqtt_client


# broker = '7203864f936b45518a401c7e7c24dbf3.s1.eu.hivemq.cloud'
# port = 8883
broker = 'broker.emqx.io'
port = 8083
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

    client = mqtt_client.Client(client_id, transport="websockets")
    #client.username_pw_set(username, password)
    client.on_connect = on_connect
    print(broker, port)
    client.connect(broker, port)
    return client


def publish(client):
    msg_count = 0
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1


def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    run()