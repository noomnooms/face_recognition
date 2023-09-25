# Facial Recognition Compilations

This repository is the result of a project, where the task is to perform facial recognition for surveillance. It was proven difficult because we only have one image/person with a total of 1200+ subjects. Some of the images were ID pictures taken years ago, and the surveillance camera feed is dark and blurry. In my efforts to improve the accuracy, I tried out different models, algorithms, and methods, which I compile in this repository. I hope you find this useful!

The original dataset is confidential, so for this repository I will use a sample dataset with 5 subjects and 5 images each.

---

## How to use

TBD

## Topics Covered

1. [Image preprocessing for Facial Recognition](1.%20image%20preprocessing%20for%20facial%20recognition.ipynb)
2. [Intro to Facial Recognition](2.%20face%20recognition%20basics.ipynb)
3. [Large-scale Face Recognition](3.%20large%20scale%20face%20recognition.ipynb)
4. Real-time Face Recognition
5. Improving prediction accuracy
6. [Unsupervised learning: Face Clustering](6.%20face%20clustering.ipynb)
7. 3D Pose Generation from single image

TBD

---

## Project Description

### System Overview

The task of facial recognition can be simplified into 3 parts

1. [Preprocess image](./modules/FacePreprocess.py) &rarr; includes face detection, facial alignment, and facial normalization. Going through this step will help increase the accuracy, as we will only focus on the necessary part of the image for facial recognition, which is the face.

2. Compute Facial Embeddings &rarr; in this step, we will use a deep learning model that has been pre-trained by others. We will input our preprocessed image into their model and the last layer of the model will output a vector. This vector is what we call <b>embeddings</b>. The models that I tested are:

   - [Keras VGGFace](https://github.com/rcmalli/keras-vggface): `resnet50`, `senet50`, `vgg16`
   - [ArcFace](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/)
   - [Facebook DeepFace](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/)
   - [Google Facenet](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/)

3. Compare Embeddings &rarr; after we have embeddings of every image in our dataset, and our target image, we can compare the vector distances between these images, and find the closest distance. The closest distanced image in our dataset, is the identity of our target image.

### Experiment 1:
