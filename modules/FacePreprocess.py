import cv2
import mediapipe 
import math
import numpy as np
import pandas as pd
import time
from PIL import Image

class FacePreprocess():
    def __init__(self, model, pretrained):
        # build the facial detector - OpenCV SSD
        self.detector = cv2.dnn.readNetFromCaffe(model, pretrained)
        self.detector_colLabel = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        self.detector_confidence = 0.90

        # build the facial landmarks detector - Mediapipe 
        mp_face_mesh = mediapipe.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces = 1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # mediapipe facial landmark boundaries
        self.FACE_OVAL = [149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176]
        self.LEFT_EYE = [33, 246, 133, 155] # left, top, right, bottom
        self.RIGHT_EYE = [362, 398, 263, 249] # left, top, right, bottom

    def rotate(self, pt, radians, origin):
        x, y = pt
        offset_x, offset_y = origin
        adjusted_x, adjusted_y = (x - offset_x), (y - offset_y)

        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

        return [qx, qy]
        
    def alignEye(self, left, right):
        left_center = (int((left[0][0]+left[2][0])/2), int((left[1][1]+left[3][1])/2))
        right_center = (int((right[0][0]+right[2][0])/2), int((right[1][1]+right[3][1])/2))

        if left_center[1] >= right_center[1]:
            point_3rd = (right_center[0], left_center[1])
            direction = -1 # rotate clockwise
        else:
            point_3rd = (left_center[0], right_center[1])
            direction = 1 # rotate counterclockwise
        
        a = self.euclidDist(left_center[0], left_center[1], point_3rd[0], point_3rd[1])
        b = self.euclidDist(right_center[0], right_center[1], left_center[0], left_center[1])
        c = self.euclidDist(right_center[0], right_center[1], point_3rd[0], point_3rd[1])
        try:
            cos_a = (b*b + c*c - a*a)/(2*b*c)
        except: return 0 #不用再align
        angle = (np.arccos(cos_a) * 180) / math.pi

        angle = 90 - angle if direction == -1 else angle

        return direction*angle

    def euclidDist(self, x1, y1, x2, y2):
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    def preproc(self, image):
        ''' 
        input -> one image frame

        return -> list of normalized+aligned faces detected from the image
        '''
        start = time.process_time()  #############
        img = image.copy()
        img_h, img_w = img.shape[:2]

        # convert image to target size (detector expects (1, 3, 300, 300) input)
        target_size = (300, 300)
        image = cv2.resize(image, target_size)
        aspect_ratio_x, aspect_ratio_y = (img_w / target_size[1]), (img_h / target_size[0])
        imageBlob = cv2.dnn.blobFromImage(image = image)

        # input image into detector + filter results
        self.detector.setInput(imageBlob)
        detections = self.detector.forward() 
        detections_df = pd.DataFrame(detections[0][0], columns = self.detector_colLabel)
        detections_df = detections_df[detections_df['is_face'] == 1]
        detections_df = detections_df[detections_df['confidence'] >= self.detector_confidence]

        # rescale back to 300x300 (results are in [0:1])
        for i in ['left', 'bottom', 'right', 'top']:
            detections_df[i] = (detections_df[i] * 300).astype(int)

        # loop through each face
        faces = []
        for index, row in detections_df.iterrows():
            # face coordinates
            left, right, top, bottom = row['left'], row['right'], row['top'], row['bottom']

            # detect face
            detected_face = img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
            boundary = [int(top*aspect_ratio_y), int(bottom*aspect_ratio_y), int(left*aspect_ratio_x), int(right*aspect_ratio_x)]

            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                # find facial landmarks
                try:
                    results = self.face_mesh.process(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
                    landmarks = results.multi_face_landmarks[0]
                except:
                    try:
                        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        landmarks = results.multi_face_landmarks[0]
                    except:
                        break

                mesh_points = np.array([np.add(np.multiply((p.x, p.y), (detected_face.shape[1], detected_face.shape[0])), (left*aspect_ratio_x, top*aspect_ratio_y)).astype(int) for p in landmarks.landmark])

                # find facial alignment angle
                rotation = self.alignEye(mesh_points[self.LEFT_EYE] , mesh_points[self.RIGHT_EYE])

                # find new facial boundary
                center = [int(img.shape[1]/2), int(img.shape[0]/2)]
                top = self.rotate(mesh_points[10] , np.radians(rotation), center)[1]
                bottom = self.rotate(mesh_points[152] , np.radians(rotation), center)[1]
                left = self.rotate(mesh_points[234] , np.radians(rotation), center)[0]
                right = self.rotate(mesh_points[454] , np.radians(rotation), center)[0]

                # extracting face oval
                routes = [(i[0], i[1]) for i in mesh_points[self.FACE_OVAL]]
                mask = np.zeros((img.shape[0], img.shape[1]))
                mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
                mask = mask.astype(bool)
                out = np.zeros_like(img)
                out[mask] = img[mask]

                # rotate image 
                new_img = Image.fromarray(out)
                new_img = np.array(new_img.rotate(rotation))

                # crop image based on new boundary
                new_img = new_img[int(top):int(bottom), int(left):int(right)]

                faces.append([new_img, boundary])

        return faces
    