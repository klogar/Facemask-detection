# Facemask-detection

This project focuses on face mask detection in video streams.

# Setting up the environment and running the code

The input to the container, set through the environment variable is the following:

```
{
  "input":{
    "url":"http://193.2.72.90:30156/mask.webm/"
  },
  "output":{
    "url":{
      "mqtt":"mqtt://194.249.2.112:30/jobs/67965ae8-d5e3-4e82-a123-151aae8bbc5e"
    }
  },
  "ai_model":{
    "url":"",
    "model_name":"facemask_model",
    "model_version":"0.1"
  },
  "autostart":{
    "value":"True"
  }
}
```

The output of the container is:
```
{
  "ai_id":"asfewwerfd-asd-4875-asdafdsf",
  "fps":0.39501934,
  "delay":2.33180904,
  "timestamp":"1620236707",
  "detections":[
    {
      "class":"with_mask",
      "location":[
        "274.6901",
        "664.73906",
        "437.6706",
        "719.99426"
      ],
      "score":0.75641185
    },
    {
      "class":"without_mask",
      "location":[
        "274.6901",
        "664.73906",
        "437.6706",
        "719.99426"
      ],
      "score":0.8971
    }
  ],
  "encoded_image": "123abc..."
}
```

For building nad running container you need to execute the following commands (this creates the base image and the image with the model):
```
docker build -f Dockerfile_nvidia -t klogar/nvidia .
docker build --no-cache -t klogar/facemask:0.0.7 .
docker run -d --name fm klogar/facemask:0.0.7
```
