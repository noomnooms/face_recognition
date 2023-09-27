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

## System Overview

The task of facial recognition can be simplified into 3 parts

1. [Preprocess image](./modules/FacePreprocess.py) &rarr; includes face detection, facial alignment, and facial normalization. Going through this step will help increase the accuracy, as we will only focus on the necessary part of the image for facial recognition, which is the face. The models used are:

   - [OpenCV SSD](https://learnopencv.com/tag/ssd/)
   - [MediaPipe](https://developers.google.com/mediapipe)

2. Compute Facial Embeddings &rarr; in this step, we will use a deep learning model (FE/Feature Extractor) that has been pre-trained by others. We will input our preprocessed image into their model and the last layer of the model will output a vector. This vector is what we call <b>embeddings</b>. The models tested are:

   - [Keras VGGFace](https://github.com/rcmalli/keras-vggface) &rarr; `resnet50`, `senet50`, `vgg16`
   - [ArcFace](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/)
   - [Facebook DeepFace](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/)
   - [Google Facenet](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/)

3. Compare Embeddings &rarr; after we have embeddings of every image in our dataset, and our target image, we can compare the vector distances between these images, and find the closest distance. The closest distanced image in our dataset, is the identity of our target image. The model used is:
   - [nmslib](https://github.com/nmslib/nmslib) &rarr; `l2` and `cosinesimil` distance

## Experiments

This is a report on the experiments I did to improve accuracy of the system. All the experiments use the same private dataset.

For our dataset, we find that the three `VGGFace` models (`resnet50`, `senet50`, and `vgg16`) perform better than the other models, so other models aren't used for the experiments.

Dataset details:

- IDs: 1284
- Image/ID: 1
- note: at least 70% of these images are recent passport ID pictures, but the rest are outdated or blurry pictures.

For the benchmark, we evaluate on two different scenarios:

1. Webcam &rarr; recorded video of 10 subjects, 30 seconds each. Video quality is clear and camera angle is similar to dataset.
2. Onsite &rarr; CCTV top-down angle in basement (dark and blurry) entrance, results are compared with building entrance records.

### <b>Experiment 1:</b> Basic Algorithm

For this experiment we use the basic pipeline of

<!-- img tbd -->

<u>Webcam results:</u>

| Model      | Distance      | `True`                                 | `False`                                | Accuracy                                  | `avg 'True'` distance | `stdev 'True'` distance |
| ---------- | ------------- | -------------------------------------- | -------------------------------------- | ----------------------------------------- | --------------------- | ----------------------- |
| `resnet50` | `cosinesimil` | 778                                    | 187                                    | 0.8062                                    | 0.2342                | 0.0662                  |
| `resnet50` | `l2`          | 757                                    | 208                                    | 0.7844                                    | 6621.3303             | 1470.4977               |
| `senet50`  | `cosinesimil` | 791                                    | 174                                    | 0.8196                                    | 0.1900                | 0.0470                  |
| `senet50`  | `l2`          | <code style="color : LightSkyBlue">801 | <code style="color : LightSkyBlue">164 | <code style="color : LightSkyBlue">0.8300 | 14349.9026            | 3123.0065               |
| `vgg16`    | `cosinesimil` | 353                                    | 612                                    | 0.3658                                    | 0.0907                | 0.0525                  |
| `vgg16`    | `l2`          | 337                                    | 628                                    | 0.3492                                    | 2054.4873             | 2205.3795               |

<i>\*note: `True` and `False` denotes the sum of `True` and `False` predictions of all the Webcam test data.</i>

### <b>Experiment 2:</b> Blur Dataset Images

Since the webcam and onsite image quality is worse than the dataset, we blur our dataset images so that it closely resembles the target.

<u>Webcam results:</u>

| Model      | Distance      | `True`                                 | `False`                                | Accuracy                                  | `avg 'True'` distance | `stdev 'True'` distance |
| ---------- | ------------- | -------------------------------------- | -------------------------------------- | ----------------------------------------- | --------------------- | ----------------------- |
| `resnet50` | `cosinesimil` | 770                                    | 195                                    | 0.7979                                    | 0.2344                | 0.0652                  |
| `resnet50` | `l2`          | 762                                    | 203                                    | 0.7896                                    | 6642.1061             | 1471.2529               |
| `senet50`  | `cosinesimil` | 791                                    | 174                                    | 0.8196                                    | 0.1905                | 0.0471                  |
| `senet50`  | `l2`          | <code style="color : LightSkyBlue">802 | <code style="color : LightSkyBlue">163 | <code style="color : LightSkyBlue">0.8310 | 14386.1649            | 3118.8102               |
| `vgg16`    | `cosinesimil` | 667                                    | 637                                    | 0.3398                                    | 0.0909                | 0.0529                  |
| `vgg16`    | `l2`          | 328                                    | 626                                    | 0.3512                                    | 2654.7033             | 2371.8272               |

With this method, we increased the accuracy by 0.1%.

### <b>Experiment 3:</b> Double Verification

In this experiment, we use both distance metrics at the same time. When both metrics return the same ID, we will judge for `True` or `False`. Otherwise we label it as `unknown`.

<u>Webcam results:</u>

| Model      | `True`                                 | `False`                               | `unknown`                             | Accuracy                                 | `avg 'True'` distance | `stdev 'True'` distance |
| ---------- | -------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------- | ----------------------- |
| `resnet50` | 729                                    | <code style="color : LightSkyBlue">81 | 155                                   | <code style="color : LightSkyBlue">0.900 | 3284.6284             | 710.7444                |
| `senet50`  | <code style="color : LightSkyBlue">775 | 98                                    | <code style="color : LightSkyBlue">92 | 0.8877                                   | 7134.1112             | 1469.9975               |
| `vgg16`    | 271                                    | 273                                   | 421                                   | 0.4981                                   | 1295.1459             | 1295.1459               |

With this method, we increased the accuracy by 6.9%, up to 90%.

<u>Onsite results:</u>

TBD

<!--
| Model      | Distance      | `True` | `False` | Accuracy | `avg` distance | `stdev` distance |
| ---------- | ------------- | ------ | ------- | -------- | -------------- | ---------------- |
|`resnet50`|`cosinesimil`| | | | | |
|`resnet50`|`l2`| | | | | |
|`senet50`|`cosinesimil`| | | | | |
|`senet50`|`l2`| | | | | |
|`vgg16`|`cosinesimil`| | | | | |
|`vgg16`|`l2`| | | | | |

<code style="color : LightSkyBlue"></code>
-->
