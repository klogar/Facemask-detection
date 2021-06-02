from decenter.ai.baseclass import BaseClass
from decenter.ai.appconfig import AppConfig
from decenter.ai.requesthandler import AIReqHandler
from decenter.ai.flask import init_handler
import decenter.ai.utils.model_utils as model_utils

import logging
import sys
import os
import json
from flask import request

from yolov5.MyModel import MyModel


def main():
    # set logger config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # localhost:8081/mask.webm
    # http://193.2.72.90:30156/construction.webm
    # config =    {"input":
    #                 {"url": "http://localhost:8081/short.webm"},
    #             "output":
    #                 {"url": {"mqtt": "mqtt://broker.emqx.io:1883/yolov5/general"}},
    #             "ai_model":
    #                 {"url": "http://182.252.132.39:5000",
    #                  "model_name": "decenter_yolov3",
    #                  "model_version": "0.1"	},
    #             "autostart":
    #                 { "value": "True" }
    #     }
    # app = BaseClass(json.loads(json.dumps(config)))

    # Init BaseClass
    if os.getenv('MY_APP_CONFIG') is None:
        app = BaseClass()
    else:
        app = BaseClass(json.loads(os.getenv('MY_APP_CONFIG')))


    my_model = MyModel(app)

    app.start(my_model)

    # start Flask message handler here
    msg_handler = init_handler(app)

    flaskapp = msg_handler.get_flask_app()

    @flaskapp.route('/getCurrentFPS', methods=['GET'])
    def getFPS():
        return str(my_model.get_current_fps())

    @flaskapp.route('/getCurrentDelay', methods=['GET'])
    def getDelay():
        return str(my_model.get_current_video_delay())

    @flaskapp.route('/setMinimumFPS', methods=['GET'])
    def setMinimumFPS():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        if my_model.set_minimum_fps(minimum_fps):
            return "setting minimum fps to " + str(request.args.get('minimum_fps', default=30, type=int))
        else:
            return "did not change fps. They have to between 60 and 1"

    @flaskapp.route('/startAI', methods=['GET'])
    def startAI():
        minimum_fps = request.args.get('minimum_fps', default=30, type=int)
        return my_model.start_thread(minimum_fps)

    @flaskapp.route('/stopAI', methods=['GET'])
    def stopAI():
        return my_model.stop_thread()

    flaskapp.run(host="0.0.0.0", threaded=True)


main()
