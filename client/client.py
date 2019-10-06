import cv2
import io
import socket
import struct
import time
import pickle
import zlib


import os
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import numpy as np
import argparse


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from firebase import firebase



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True,help="host IP of server")
ap.add_argument("--port", required=True,help="port of host IP")
args = vars(ap.parse_args())

HOST=args['host']
PORT=int(args['port'])


#讀取模型設定
useNeuralStick=False
useTensor=False
face_detection_model="frozen_inference_graph.pb"
frozen_inference_graph="graph_ssd.pbtxt"
filter_confidence = 0.5

#firebase
firebase = firebase.FirebaseApplication('https://ntnu-skyeyes.firebaseio.com/', authentication=None) 
num_registered_local=pickle.loads(open("firebase/num_registered_local.pickle", "rb").read())


detector=None
embedder=None
recognizer=None
le=None


def reload_model():
    #load face dection model
    print("[INFO] loading face detector...")

    global detector
    global embedder
    global recognizer
    global le
    
    if useTensor:
        protoPath = os.path.sep.join(["face_detection_model", frozen_inference_graph])
        modelPath = os.path.sep.join(["face_detection_model",face_detection_model])
        detector = cv2.dnn.readNetFromTensorflow(modelPath,protoPath)
    else:
        protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    if useNeuralStick:
        print("[INFO]setting a neural stick is activated")
        detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    print("[INFO]completed")



def training():
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = firebase.get('/trainingData','')

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

    #紀錄現在有多少人註冊過(local)
    names=list(set(data["names"]))
    f = open("firebase/num_registered_local.pickle", "wb")
    f.write(pickle.dumps(len(names)))
    f.close()

    global num_registered_local
    num_registered_local=len(names)

    print("[INFO]completed")


reload_model()

#socket connect...
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

cam.set(3, 320);
cam.set(4, 240);

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:

    #確認client端的模型是否是最新的 不是就更新
    num_registered_server=int(client_socket.recv(1024))
    if(num_registered_server!=num_registered_local):
        print("*****model synchronization*****")
        training()
        reload_model()
        print("*****model synchronization completed*****")

    ret, frame = cam.read()

    #人臉辨識
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image

    
    detector.setInput(imageBlob)
    detections = detector.forward()

    name="not"
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > filter_confidence:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #傳送辨識人名
    client_socket.sendall(bytes(str(name), 'utf8'))

    #辨識完把辨識好的圖片送到server
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)

    #print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    #img_counter += 1

    

cam.release()
