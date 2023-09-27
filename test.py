from models.NextFace.optimizer import Optimizer
from models.NextFace.config import Config
from models.NextFace.utils import *
from models.NextFace.image import saveImage
import math
import json
import mediapipe

input_path = './dataset/train/joy/img_1.jpg'
output_path = './output/3D_pose_generation/'

config_path = './models/NextFace/optimConfig.ini'
config = Config()
config.fillFromDicFile(config_path)

if config.device == 'cuda' and torch.cuda.is_available() == False:
    print('[WARN] no cuda enabled device found. switching to cpu... ')
    config.device = 'cpu'
if config.lamdmarksDetectorType == 'mediapipe':
    try:
        from models.NextFace.landmarksmediapipe import LandmarksDetectorMediapipe
    except:
        print('[WARN] Mediapipe for landmarks detection not availble. falling back to FAN landmarks detector. You may want to try Mediapipe because it is much accurate than FAN (pip install mediapipe)')
        config.lamdmarksDetectorType = 'fan'

sharedIdentity = None
checkpoint = ''
doStep1 = True
doStep2 = True
doStep3 = True

optimizer = Optimizer(outputDir=output_path, config=config)
optimizer.run(
    input_path,
    sharedIdentity=sharedIdentity,
    checkpoint=checkpoint,
    doStep1=doStep1,
    doStep2=doStep2,
    doStep3=doStep3
)
