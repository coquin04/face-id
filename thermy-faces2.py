# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:48:07 2020

@author: Coquin
"""
# face detection
from os import listdir
from os.path import isdir
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import savez_compressed, asarray, load, expand_dims, array
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from joblib import dump
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from cv2.dnn import blobFromImage, readNet
from time import time
import os
from src.help_func import *


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(['model', "deploy.prototxt"])
weightsPath = os.path.sep.join(['model',
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = readNet(prototxtPath, weightsPath)

t0 = time()
#---------------------------------------#
# load train dataset
trainX, trainy = load_dataset(
    'C:/Users/Thermy/Google Drive (thermy.hc.ai@gmail.com)/Jorge/Thermy-ID/dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(
    'C:/Users/Thermy/Google Drive (thermy.hc.ai@gmail.com)/Jorge/Thermy-ID/dataset/val/')
# save arrays to one file in compressed format
tf = time()
print('Tiempo de deteccion de rostros:', tf-t0)
savez_compressed('thermy-dataset2.npz', trainX, trainy, testX, testy)
#---------------------------------------#
# load the face dataset
data = load('thermy-dataset2.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
model_keras = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model_keras, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model_keras, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('thermy-dataset-embeddings2.npz',
                 newTrainX, trainy, newTestX, testy)
#---------------------------------------#
# load dataset
data = load('thermy-dataset-embeddings2.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model_skl = SVC(kernel='linear', probability=True)
model_skl.fit(trainX, trainy)
# predict
yhat_train = model_skl.predict(trainX)
yhat_test = model_skl.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
#---------------------------------------#
# save the model to disk
filename = 'last_model2.sav'
dump(model_skl, filename)
