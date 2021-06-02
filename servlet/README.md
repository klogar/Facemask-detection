# Servlet
This is the servlet that serves the video files in .webm format. If you prefer to run it locally from file, use ```servlet.py```. Otherwise you can build and run the container using ```Dockerfile```:

```
docker build -t klogar/servlet .
docker run -d --name servlet -p 8081:8081 klogar/servlet
```
