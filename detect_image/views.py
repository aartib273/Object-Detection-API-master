from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.core.files.temp import NamedTemporaryFile

import io
import os
from PIL import Image
import cv2
import numpy as np
from base64 import b64decode, b64encode
from .utils import *
from .darknet import Darknet
from .darknet import YoloV4Wrapper
from .darknet import EfficientNetB7Wrapper

# Create your views here.
#########################

@csrf_exempt
def yolo_detect_api(request):
    data = {'success':False}
    url = ''
    print(request)

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            print(123)
            image_request = request.FILES["image"]
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            result, url.yolo4_prediction,eff_prediction = yolo_detect(image)

        elif request.POST.get("image64", None) is not None:
            print(567)
            base64_data = request.POST.get("image64", None).split(',', 1)[1]
            print(1, base64_data)
            plain_data = b64decode(base64_data)
            print(2, plain_data)
            plain_data = np.array(Image.open(io.BytesIO(plain_data)))
            print(3, plain_data)
            result, url.yolo4_prediction,eff_prediction = yolo_detect(plain_data)
            print(4, result)
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
    cfg_file = './cfg/yolov3.cfg'
    weight_file = './weights/yolov3.weights'
    namesfile = 'data/coco.names'

    m = Darknet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_class_names(namesfile)

    resized_image = cv2.resize(original_image, (m.width, m.height))

    nms_thresh = 0.6
    iou_thresh = 0.4

    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    url = plot_boxes(original_image, boxes, class_names, plot_labels = True)
    objects = print_objects(boxes, class_names)
    yolo4_prediction = self.run_yolo4(original_image)
    eff_prediction = self.run_effiecientNet(original_image)
    return objects, url, yolo4_prediction, eff_prediction
