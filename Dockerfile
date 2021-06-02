FROM klogar/nvidia:latest

ADD ./decenter_yolov3_0.1.zip /
ADD ./main.py /
ADD ./best2.pt /
ADD yolov5 /yolov5/

ENV MY_APP_CONFIG="{\
	\"input\": \
	{ \"url\": \"http://83.212.126.92:8081/folder/short.webm\" },\
	\"output\": \
		{ \"url\": {\
			\"mqtt\":\"mqtt://broker.emqx.io:8083/yolov5\"}\
 		},\
	\"ai_model\":\
		{ \"url\": \"http://182.252.132.39:5000\",\
		  \"model_name\": \"decenter_yolov3\",\
		  \"model_version\": \"0.1\"	 },\
	\"autostart\": \
		{ \"value\": \"True\" } \
	}"

WORKDIR /

# Expose ports
# for flask
EXPOSE 5000

ENV PYTHONPATH /yolov5

CMD ["python", "main.py"]