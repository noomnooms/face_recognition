import cv2
import matplotlib.pyplot as plt
import keras_vggface as kv
import modules.utils as utils
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import nmslib

# Declare a FacePreprocess instance.
from modules.FacePreprocess import FacePreprocess
ssd_model = r'./models/ssd/deploy.prototxt.txt'
ssd_weights = r'./models/ssd/res10_300x300_ssd_iter_140000.caffemodel'
processor = FacePreprocess(ssd_model, ssd_weights)

# Use the facial embedding model you want to use
model = kv.VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)
input_size = (224, 224)

# importing our nmslib index tree
nmslib_path = './output/large_scale_face_recognition/'

# load id_list
id_list = pd.read_csv(nmslib_path + '/IDlist.csv')

# Euclidean distance
index_l2 = nmslib.init(method='hnsw', space='l2',
                       data_type=nmslib.DataType.DENSE_VECTOR)
index_l2.loadIndex(nmslib_path + 'index_l2.bin')

# Cosine similarity
index_cos = nmslib.init(method='hnsw', space='cosinesimil',
                        data_type=nmslib.DataType.DENSE_VECTOR)
index_cos.loadIndex(nmslib_path + 'index_cos.bin')

video_path = './dataset/test/test_2.mp4'
cap = cv2.VideoCapture(video_path)

results = pd.DataFrame(
    columns=['count', 'irene', 'seulgi', 'wendy', 'joy', 'yeri'],
    index=['l2', 'cosinesimil']
)
results.fillna(0, inplace=True)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid = './output/real_time_face_recognition/test.mp4v'
videoWriter = cv2.VideoWriter(vid, fourcc, 12.0, (640, 480))
dist = 'l2'
while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        img = frame.copy()

        faces = processor.preproc(img)

        if len(faces) > 0:
            for face in faces:
                results['count'][dist] += 1

                # target embeddings
                target = model.predict(resize(face[0], input_size))[0, :]
                target = np.array(target, dtype='f')
                target = np.expand_dims(target, axis=0)

                # predict
                neighbors, distances = index_l2.knnQueryBatch(
                    target, k=1, num_threads=4)[0]

                # results
                name = id_list['name'][neighbors[0]]
                results[name][dist] += 1

                # add video frame
                top, bottom, left, right = face[1][0], face[1][1], face[1][2], face[1][3]
                cv2.putText(frame, str(name), (int(left), int(top-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (255, 255, 255), 1)
                videoWriter.write(frame)
    except:
        break
videoWriter.release()
cap.release()
