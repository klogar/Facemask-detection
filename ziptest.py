import zipfile

zip_ref = zipfile.ZipFile("decenter_yolov3_0.1.zip", 'r')
zip_ref.extractall("/")
zip_ref.close()