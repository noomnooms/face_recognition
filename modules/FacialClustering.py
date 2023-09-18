from sklearn.cluster import DBSCAN
import shutil
from datetime import datetime
from random import shuffle
import networkx as nx
import pandas as pd
import keras_vggface as kv
from modules.FacePreprocess import FacePreprocess
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FacialClustering():
    def __init__(self, pathlist: list, processor: FacePreprocess, out_path: str, preprocess: bool = False):
        '''
        Given a list of paths, cluster all images inside the path by identity

        ----
        Parameters
            - pathlist: list of paths containing images (assumes each face is already cropped)
            - processor: FacePreprocess object 
            - out_path: output directory path
            - preprocess: `False` default, but if your data needs to be preprocess use `True`
        '''
        # initialize face preprocessor
        self.processor = processor

        # output data
        try:
            os.makedirs(out_path)
        except:
            pass
        self.out_path = out_path
        self.filename = self.out_path+'\\log.txt'

        # face clustering report
        file = open(self.filename, 'w')
        file.write('Path list:\n')
        for path in pathlist:
            file.write(' - '+path+'\n')
        file.write('\nNo faces detected in these files:\n')

        # list all image paths
        self.img_list = pd.DataFrame(columns=['path', 'image'])
        cnt = 0
        img_ext = ['.jpg', '.png']
        for path in pathlist:  # iterate through every path
            # look for every image inside the path
            for root, dirs, files in os.walk(path):
                for name in files:
                    add = os.path.join(root, name)
                    if add.endswith(tuple(img_ext)):  # image preprocessing
                        try:
                            if preprocess:
                                img = processor.preproc(cv2.imread(add))[0][0]
                            else:
                                img = cv2.imread(add)
                            self.img_list.loc[self.img_list.shape[0]] = [
                                add, img]
                        except:
                            file.write(' - '+add+'\n')
                        cnt += 1

        file.write(
            '\n({}/{}) images processed.\n'.format(self.img_list.shape[0], cnt))
        file.write('='*50+'\n')
        file.close()

    def resize(self, image, size):
        '''
        Resize and reshape image for model prediction (keras format)

        ----
        Parameters
            - image: single image 
            - size: tuple (x, y)
        '''
        img = cv2.resize(image, size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def extract_features(self, FE: str):
        '''
        Returns list of facial encodings from the model output.

        ----
        Parameters
            - FE: 'kv-resnet50', 'kv-senet50', 'kv-vgg16', 'arcface', 'deepface', 'facenet'
        '''
        facial_encodings = []

        if FE in ['kv-resnet50', 'kv-senet50', 'kv-vgg16']:
            # keras_vggface
            FE = FE.replace('kv-', '')
            model = kv.VGGFace(model=FE, include_top=False,
                               input_shape=(224, 224, 3), pooling='avg')

            for index, row in self.img_list.iterrows():
                encodings = model.predict(self.resize(
                    row['image'], (224, 224)), verbose=0)[0, :]
                facial_encodings.append(np.array(encodings))
        elif FE in ['arcface']:
            model = keras.models.load_model(
                './models/arcface/model.keras', compile=False)

            for index, row in self.img_list.iterrows():
                encodings = model.predict(self.resize(
                    row['image'], (112, 112)), verbose=0)[0, :]
                facial_encodings.append(np.array(encodings))
        elif FE in ['deepface']:
            model = keras.models.load_model(
                './models/deepface/model.keras', compile=False)

            for index, row in self.img_list.iterrows():
                encodings = model.predict(self.resize(
                    row['image'], (152, 152)), verbose=0)[0, :]
                facial_encodings.append(np.array(encodings))
        elif FE in ['facenet']:
            model = keras.models.load_model(
                './models/facenet/model.keras', compile=False)

            for index, row in self.img_list.iterrows():
                encodings = model.predict(self.resize(
                    row['image'], (160, 160)), verbose=0)[0, :]
                facial_encodings.append(np.array(encodings))

        return facial_encodings

    def chinese_whispers(self, FE: str = 'kv-resnet50', threshold: float = 0.75, iterations: int = 20, saveas: bool = True):
        '''
        Clustering with chinese whispers method
        source: https://github.com/zhly0/facenet-face-cluster-chinese-whispers-/blob/master/clustering.py#L12

        ----
        Parameters
            - FE: 'kv-resnet50', 'kv-senet50', 'kv-vgg16', 'arcface', 'deepface', 'facenet'
            - threshold: min distance between clusters
            - iterations: number of iterations
            - saveas: save a copy of the clustered faces
        '''
        file = open(self.filename, 'a')
        file.write('\n{}\n'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        file.write('Chinese whispers\nFE: {}\tThreshold: {}\tIterations: {}\n'.format(
            FE, threshold, iterations))

        # extract facial encodings
        facial_encodings = self.extract_features(FE)
        if len(facial_encodings) == 0:
            file.write('FE is not supported\n\n')
            file.write('='*50+'\n')
            return

        # create graph
        nodes = []
        edges = []

        for index, row in self.img_list.iterrows():
            # add node of facial encoding
            node_id = index+1

            # initialize cluster to unique value (cluster of itself)
            node = (node_id, {'cluster': row['path'], 'path': row['path']})
            nodes.append(node)

            # facial encodings to compare
            if node_id >= self.img_list.shape[0]:
                # node is last element, don't create edge
                break

            compare_encodings = facial_encodings[node_id:]
            if len(compare_encodings) == 0:
                distances = np.empty((0))
            else:
                distances = np.sum(compare_encodings *
                                   facial_encodings[index], axis=1)
            encoding_edges = []

            for i, distance in enumerate(distances):
                if distance > threshold:
                    edge_id = index+i+2
                    encoding_edges.append(
                        (node_id, edge_id, {'weight': distance}))

            edges = edges + encoding_edges

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # Iterate
        for _ in range(iterations):
            cluster_nodes = list(G.nodes())
            shuffle(cluster_nodes)
            for node in cluster_nodes:
                neighbors = G[node]
                clusters = {}

                for ne in neighbors:
                    if isinstance(ne, int):
                        if G.nodes[ne]['cluster'] in clusters:
                            clusters[G.nodes[ne]['cluster']
                                     ] += G[node][ne]['weight']
                        else:
                            clusters[G.nodes[ne]['cluster']
                                     ] = G[node][ne]['weight']

                # find the class with the highest edge weight sum
                edge_weight_sum = 0
                max_cluster = 0
                for cluster in clusters:
                    if clusters[cluster] > edge_weight_sum:
                        edge_weight_sum = clusters[cluster]
                        max_cluster = cluster
                # set the class of target node to the winning local class
                G.nodes[node]['cluster'] = max_cluster

        clusters = {}
        # Prepare cluster output
        for (_, data) in G.nodes.items():
            cluster = data['cluster']
            path = data['path']

            if cluster:
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(path)

        # Sort cluster output
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        file.write('{} clusters found\n'.format(len(sorted_clusters)))

        # output
        output_path = os.path.join(
            self.out_path, 'CW_{}_th{}_it{}'.format(FE, threshold, iterations))
        try:
            os.makedirs(output_path)
        except:
            pass
        writer = pd.ExcelWriter(
            output_path+'\\output.xlsx', engine='xlsxwriter')
        for i, pathlist in enumerate(sorted_clusters):
            df = pd.DataFrame(pathlist, columns=['path'])
            df.to_excel(writer, sheet_name=str(i), index=False)

            if saveas:
                cluster_dir = output_path+'\\'+str(i)
                try:
                    os.makedirs(cluster_dir)
                except:
                    pass
                for path in pathlist:
                    shutil.copy(path, cluster_dir+'\\'+os.path.basename(path))

        writer.close()

        file.write('Output path: {}\n\n'.format(output_path))
        file.write('='*50+'\n')

        file.close()

    def DBSCAN(self, FE: str = 'kv-resnet50', eps: float = 0.5, metric: str = 'euclidean', min_samples: int = 3, saveas: bool = True):
        '''
        Clustering with DBSCAN method
        source: https://github.com/AsutoshPati/Face-Clustering-using-DBSCAN 

        ----
        Parameters
            - FE: 'kv-resnet50', 'kv-senet50', 'kv-vgg16', 'arcface', 'deepface', 'facenet'
            - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            - metric:  From scikit-learn -> ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
            - saveas: save a copy of the clustered faces
        '''
        file = open(self.filename, 'a')
        file.write('\n{}\n'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        file.write('DBSCAN\nFE: {}\teps: {}\tmetric: {}\tmin_samples: {}\n'.format(
            FE, eps, metric, min_samples))

        # extract facial encodings
        facial_encodings = self.extract_features(FE)
        if len(facial_encodings) == 0:
            file.write('FE is not supported\n\n')
            file.write('='*50+'\n')
            return

        clt = DBSCAN(eps=eps, metric=metric,
                     min_samples=min_samples, n_jobs=-1)
        clt.fit(facial_encodings)
        labels_id = np.unique(clt.labels_)
        file.write('{} clusters found\n'.format(len(labels_id)-1))

        # output
        output_path = os.path.join(
            self.out_path, 'DBSCAN_{}_eps{}_{}_min{}'.format(FE, eps, metric, min_samples))
        try:
            os.makedirs(output_path)
        except:
            pass
        writer = pd.ExcelWriter(
            output_path+'\\output.xlsx', engine='xlsxwriter')
        for label in labels_id:
            idxs = np.where(clt.labels_ == label)[0]
            pathlist = [self.img_list['path'][i] for i in idxs]

            sheet_name = str(label) if label != -1 else 'no_class'
            df = pd.DataFrame(pathlist, columns=['path'])
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if saveas:
                cluster_dir = output_path+'\\'+sheet_name
                try:
                    os.makedirs(cluster_dir)
                except:
                    pass
                for path in pathlist:
                    shutil.copy(path, cluster_dir+'\\'+os.path.basename(path))

        writer.close()

        file.write('Output path: {}\n\n'.format(output_path))
        file.write('='*50+'\n')

        file.close()
