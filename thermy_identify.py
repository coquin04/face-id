# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:50:17 2020

@author: Coquin
"""
# face_id detection
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import load, expand_dims, array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, Normalizer
from cv2 import cvtColor, resize, COLOR_BGR2RGB
from cv2.dnn import blobFromImage, readNet
from imutils.video import VideoStream
from src.help_func import *
import imutils
import time
import joblib
import cv2
import os
# extract a single face_id from a given photograph


# load our serialized face_id detector model_id from disk
print("[INFO] loading face_id detector model_id...")
prototxtPath_id = os.path.sep.join(['face_detector', "deploy.prototxt"])
weightsPath_id = os.path.sep.join(['face_detector',
                                   "res10_300x300_ssd_iter_140000.caffemodel"])
net_id = readNet(prototxtPath_id, weightsPath_id)
# load model_id for build embeddings
model_keras_id = load_model(os.path.sep.join(['model', 'facenet_keras.h5']))
#---------------------------------------#
# load dataset
embeddings_dir = os.path.sep.join(['model', "deploy.prototxt"])
data_id = load(embeddings_dir)
trainy_id = data_id['arr_1']
# label_id encode targets
out_encoder_id = LabelEncoder()
out_encoder_id.fit(trainy_id)
trainy_id = out_encoder_id.transform(trainy_id)
# load trained SVC model_id from disk
filename_id = 'last_model2.sav'
model_skl_id = joblib.load(filename_id)
#---------------------------------------#
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs_id = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frame_ids from the video stream
while True:
    # grab the frame_id from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame_id = vs_id.read()
    frame_id = imutils.resize(frame_id, width=400)

    # detect faces_id in the frame_id and determine if they are wearing a
    # face_id mask or not
    (locs_id, names_id) = extract_reco_face(frame_id, net_id,
                                            model_keras_id, model_skl_id, required_size_id=(160, 160), confi_id=0.5)
    # loop over the detected face_id locations and their corresponding
    # locations
    for (box_id, name_id) in zip(locs_id, names_id):
        # unpack the bounding box and predictions
        (startX_id, startY_id, endX_id, endY_id) = box_id
        # determine the class label_id and color_id we'll use to draw
        # the bounding box and text
        label_id = name_id
        color_id = (0, 255, 0)
        # display the label_id and bounding box rectangle on the output
        # frame_id
        cv2.putText(frame_id, label_id, (startX_id, startY_id - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_id, 2)
        cv2.rectangle(frame_id, (startX_id, startY_id),
                      (endX_id, endY_id), color_id, 2)

    # show the output frame_id
    cv2.imshow("Thermy ID", frame_id)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs_id.stop()
vs_id.stream.release()
