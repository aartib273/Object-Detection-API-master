from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.core.files.temp import NamedTemporaryFile

import io
import os
from PIL import Image
import numpy as np
from base64 import b64decode, b64encode
from .utils import *
import traceback2 as traceback
import cv2
import time
import sys
import logging
from cfg.config import *
#logging.basicConfig(stream=sys.stdout)
#LOGGER = logging.getLogger(__name__)
#LOGGER.setLevel(ODS_LOG_LEVEL)
from yolov4.tf import YOLOv4
class YoloV4Wrapper:
    """ This is a wrapper on the yolov4 model https://pypi.org/project/yolov4/ """

    is_model_loaded = False
    yolo = None

    def load_model():
        """ Load the model from disk. This should be called before the predict API """
        start_time = time.time()
        # LOGGER.info("Start loading the YoloV4 model")
        YoloV4Wrapper.yolo = YOLOv4()
        YoloV4Wrapper.yolo.classes = YOLOV4_CLASSES
        YoloV4Wrapper.yolo.make_model()
        YoloV4Wrapper.yolo.load_weights(YOLOV4_WEIGHTS, weights_type="yolo")
        YoloV4Wrapper.is_model_loaded = True
        exec_time = time.time() - start_time
        # LOGGER.info("Done loading the YoloV4 model in {:.2f} ms".format(exec_time * 1000))

    def predict(image_path):
        """ Return the prediction for an image. """
        if not YoloV4Wrapper.is_model_loaded:
            return print("ModelNotLoaded")
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        bboxes = YoloV4Wrapper.yolo.predict(
            frame,
            iou_threshold=0.1,
            score_threshold=0.25,
        )
        exec_time = time.time() - start_time
        # LOGGER.info("YoloV4 prediction done in: {:.2f} ms".format(exec_time * 1000))

        predictions = []
        for bbox in bboxes:
            print("bbox--->",bbox)
            class_id = int(bbox[4])
            class_name = YoloV4Wrapper.yolo.classes[class_id]
            confidence_level = bbox[5]
            predictions.append({'id': 'coco_' + str(class_id), 'name': class_name, 'confidence': confidence_level})
        return predictions

    def _read_classes_names(classes_name_path):
        """
        @return {id: class name}
        """
        classes = {}
        with open(classes_name_path, "r") as fd:
            index = 0
            for class_name in fd:
                class_name = class_name.strip()
                if len(class_name) != 0:
                    classes[index] = class_name.replace(" ", "_")
                    index += 1

        return classes


# Create your views here.
#########################

@csrf_exempt
def yolo_detect_api(request):
    try:
        
        YoloV4Wrapper.load_model()
    except Exception:
        print("******************")
        traceback.print_exc()
        print("******************")
    data = {'success':False}
    url = ''
    print(request)

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image_request = request.FILES["image"]
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            result, url,yolo4_prediction,eff_prediction = yolo_detect(request.FILES["image"])

        elif request.POST.get("image64", None) is not None:
            base64_data = request.POST.get("image64", None).split(',', 1)[1]
            plain_data = b64decode(base64_data)
            plain_data = np.array(Image.open(io.BytesIO(plain_data)))
            result, url,yolo4_prediction,eff_prediction = yolo_detect(request.FILES["image"])
        if result:
            data['success'] = True

    data['objects'] = result
    data['url'] = url
    data['yolo4_prediction'] = yolo4_prediction
    data['eff_prediction'] = eff_prediction
    return JsonResponse(data)

def detect(request):
    return render(request, 'index.html')

def run_yolo4(image_path):
    """ Test that the model is able to predict a dog with a confidence level higher than 50% """
    YoloV4Wrapper.load_model()
    predictions = YoloV4Wrapper.predict(image_path)
    return predictions
    # only_dog_predictions = list(filter(lambda prediction: prediction['name'] == 'dog', predictions))
    # print(only_dog_predictions)

def run_effiecientNet(image_path):
    """ Test that the model is able to predict an Egyptian cat with a confidence level higher than 50% """
    EfficientNetB7Wrapper.load_model()
    predictions = EfficientNetB7Wrapper.predict(image_path)
    return predictions
    # only_egyptian_cat_predictions = list(filter(lambda prediction: prediction['name'] == 'Egyptian_cat', predictions))
    # print(only_egyptian_cat_predictions)
   

def yolo_detect(original_image):
    
    yolo4_prediction = "Test yolo4_prediction"
    try:
        YoloV4Wrapper.load_model()
    except Exception:
        traceback.print_exc()
    eff_prediction = "Test eff_prediction"
    objects = "objects"
    url = ""
    return objects, url, yolo4_prediction, eff_prediction
