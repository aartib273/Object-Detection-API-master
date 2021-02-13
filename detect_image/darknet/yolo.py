from yolov4.tf import YOLOv4
import cv2
import time
import sys
import logging

#logging.basicConfig(stream=sys.stdout)
#LOGGER = logging.getLogger(__name__)
#LOGGER.setLevel(ODS_LOG_LEVEL)


class YoloV4Wrapper:
    """ This is a wrapper on the yolov4 model https://pypi.org/project/yolov4/ """

    is_model_loaded = False
    yolo = None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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