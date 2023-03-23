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

# extract a single face from a given photograph


def extract_face(filename, required_size=(160, 160), confi=0.5):
    # load image from file
    image = imread(filename)
    # retrive spatial dimensions
    (h, w) = image.shape[:2]
    # construct a blob from the image
    blob = blobFromImage(image, 1.0, (300, 300),
                         (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confi:
            # compute the (x,y)-coordinates of the boundung box for
            # the object
            box = detections[0, 0, i, 3:7] * array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cvtColor(face, COLOR_BGR2RGB)
            face = resize(face, required_size)
            face = img_to_array(face)
    return face

# load images and extract faces for all images in a directory


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):

        # path
        if '.jpg' in filename:

            path = directory + filename
            # get face
            face = extract_face(path)
            # store
            faces.append(face)
        else:
            continue
    return faces

# load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        print(path)
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# get the face embedding for one face


def get_embedding(model_keras, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model_keras.predict(samples)
    return yhat[0]


def extract_reco_face(frame_id, net_id, model_keras_id, model_skl_id, required_size_id=(160, 160), confi_id=0.5):
    # retrive spatial dimensions
    (h_id, w_id) = frame_id.shape[:2]
    # switch channels from BGR to RGB
    #frame_id = cvtColor(frame_id, COLOR_BGR2RGB)
    # construct a blob_id from the frame_id
    blob_id = blobFromImage(frame_id, 1.0, (300, 300),
                            (104.0, 177.0, 123.0))
    # pass the blob_id through the net_idwork and obtain the face_id detections_id
    net_id.setInput(blob_id)
    detections_id = net_id.forward()
    # initialize our list of faces_id, their corresponding locations,
    # and the list of predictions from our face_id mask net_idwork
    faces_id = []
    locs_id = []
    # loop over the detections_id
    for i_id in range(0, detections_id.shape[2]):
        # extract the confi_iddence (i.e., probability) associated with
        # detection
        confidence_id = detections_id[0, 0, i_id, 2]
        # filter out weak detections_id by ensuring the confidence_id is
        # greater than the minimum confidence_id
        if confidence_id > confi_id:
            # compute the (x,y)-coordinates of the boundung box for
            # the object
            box_id = detections_id[0, 0, i_id, 3:7] * \
                array([w_id, h_id, w_id, h_id])
            (startX_id, startY_id, endX_id, endY_id) = box_id.astype('int')
            # ensure the bounding box_ides fall within the dimensions of
            # the frame_id
            (startX_id, startY_id) = (max(0, startX_id), max(0, startY_id))
            (endX_id, endY_id) = (min(w_id-1, endX_id), min(h_id-1, endY_id))
            # extract the face_id ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face_id = frame_id[startY_id:endY_id, startX_id:endX_id]
            face_id = cvtColor(face_id, COLOR_BGR2RGB)
            face_id = resize(face_id, required_size_id)
            face_id = img_to_array(face_id)
            #face_id = expand_dims(face_id, axis=0)
            # add face_id and bounding box_id to their respective
            # lists
            faces_id.append(face_id)
            locs_id.append((startX_id, startY_id, endX_id, endY_id))
    # only make a predictions if at least one face_id was detected
    if len(faces_id) > 0:
        names_id = []
        for j_id in range(0, len(faces_id)):
            # get embeding of the current frame_id
            embedding_id = get_embedding(model_keras_id, faces_id[j_id])
            # normalize embedding_id
            in_encoder_id = Normalizer(norm='l2')
            # add dimension at index 0
            embedding_id = expand_dims(embedding_id, axis=0)
            embedding_id = in_encoder_id.transform(embedding_id)
            # prediction for the face_id
            yhat_class_id = model_skl_id.predict(embedding_id)
            yhat_prob_id = model_skl_id.predict_proba(embedding_id)
            # get name_id
            class_index_id = yhat_class_id[0]
            class_proba_id = yhat_prob_id[0, class_index_id] * 100
            predict_names_id = out_encoder_id.inverse_transform(yhat_class_id)
            print(predict_names_id[0])
            if class_proba_id > 50:
                names_id.append(predict_names_id[0])
            else:
                names_id.append('Desconocido')
            # names_id.append(predict_names_id[0])
            print(class_proba_id)
    else:
        names_id = []

    return (locs_id, names_id)

# get the face_id embedding_id for one face_id


def get_embedding(model_id, face_pixels_id):
    # scale pixel values
    face_pixels_id = face_pixels_id.astype('float32')
    # standardize pixel values across channels (global)
    mean_id, std_id = face_pixels_id.mean(), face_pixels_id.std()
    face_pixels_id = (face_pixels_id - mean_id) / std_id
    # transform face_id into one sample
    samples_id = expand_dims(face_pixels_id, axis=0)
    # make prediction to get embedding_id
    yhat_id = model_id.predict(samples_id)
    return yhat_id[0]
