from ultralytics import YOLO

model = YOLO("D:/123main/ultralytics/cfg/models/v8/leyolomedium.yaml").Load("D:/123main/123main/LeYOLOMedium.pt") 

model.train(data="D:/123main/ultralytics/cfg/datasets/mydata.yaml", epochs=200, workers=1,imgsz=96,device=3,batch=64,resume=True)

model.val()
