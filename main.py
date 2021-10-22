# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import tensorflow as tf

from keras.preprocessing import image
#use this latter for bounding box
resMap = {
        1 : 'Mask On',
        0 : 'Kindly Wear Mask'
    }

colorMap = {
        1 : (0,255,0),
        0 : (0,0,255)
    }


def prepImg(pth):
    return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0



#defining model paths
caffe_model = "Face detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
prototext_path = "Face detector/deploy.prototxt.txt"
mask_path = 'mask_model/face_mask_model.h5'

#Load Models
print("Loading models...................\n")

#Loading mask classification model
print("[INFO] Loading model 1...\n")
model = tf.keras.models.load_model(mask_path)
# model.load_weights(mask_path)

#Loading face detction model
print("[INFO] Loading model 2...\n")
net = cv2.dnn.readNetFromCaffe(prototext_path,caffe_model)

# initialize the video stream to get the video frames
vs_num = input('[INPUT] Enter your video streaming devices unique id: ')
print("[INFO] starting video stream...")
vs = VideoStream(src=int(vs_num)).start()
time.sleep(2.0)


#loop the frames from the  VideoStream
while True :
    #Get the frames from the video stream and resize to 400 px
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h, w) = frame.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence and prediction
        
        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum confidence
        if confidence > 0.8 :
            
            # Determine the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY, endX, endY) = (startX-15, startY-10, endX+15, endY+15)

            #Determining bounding box color and text
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            #get the image to pass to the mask detction model
            img = frame[startX:endX, startY:endY]

            pred = model.predict(prepImg(img))

            pred = np.round(pred)     

            #draw the bounding box of the face along with the associated text
            cv2.rectangle(frame,(startX,startY),(endX,endY),
                            colorMap[pred[0][0]],2)
            cv2.putText(frame, resMap[pred[0][0]],(startX,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
