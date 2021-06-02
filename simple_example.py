import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Images
img1 = Image.open('image.jpg')  # PIL image
img2 = cv2.imread('image.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images

# Inference
results = model(imgs, size=512)  # includes NMS

# Results
results.print()
results.show()  # or .show()

#print(results.xyxy[0] ) # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
print(results.pandas().xyxy)  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
