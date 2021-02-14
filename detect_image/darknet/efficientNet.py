import time

import sys

import json
import numpy as np
import cv2

#logging.basicConfig(stream=sys.stdout)
#LOGGER = logging.getLogger(__name__)
#LOGGER.setLevel(ODS_LOG_LEVEL)


class EfficientNetB7Wrapper:
    """ Wrapper on the EfficientNetB7 model """

    is_model_loaded = False
    model = None

    @staticmethod
    def load_model():
        import tensorflow as tf
        from tensorflow.python.keras.applications import imagenet_utils
        """ Load the model from disk. This should be called before the predict API """
        start_time = time.time()
        #LOGGER.info("Start loading the EfficientNetB7 model")
        EfficientNetB7Wrapper.model = tf.keras.applications.EfficientNetB7(
            include_top=True, weights=None, input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation='softmax')

        EfficientNetB7Wrapper.model.load_weights(EFFICIENTNETB7_WEIGHTS)

        with open(EFFICIENTNETB7_CLASSES) as f:
            imagenet_utils.CLASS_INDEX = json.load(f)

        EfficientNetB7Wrapper.is_model_loaded = True
        exec_time = time.time() - start_time
        #LOGGER.info("Done loading the EfficientNetB7 model in {:.2f} ms".format(exec_time * 1000))

    @staticmethod
    def predict(image_path):
        import tensorflow as tf
        """ Return the prediction for an image. """
        if not EfficientNetB7Wrapper.is_model_loaded:
            raise ModelNotLoadedError

        image = cv2.imread(image_path)

        image_size = EfficientNetB7Wrapper.model.input_shape[1]
        image = cv2.resize(image, (image_size, image_size))
        image = np.expand_dims(image, 0)

        start_time = time.time()
        prediction = EfficientNetB7Wrapper.model.predict(image)
        exec_time = time.time() - start_time
        #LOGGER.info("EfficientNetB7 prediction done in: {:.2f} ms".format(exec_time * 1000))
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(prediction, top=20)

        predictions = []
        if len(decoded_predictions) > 0:
            for decoded_prediction in decoded_predictions[0]:
                class_id = str(decoded_prediction[0])
                class_name = decoded_prediction[1]
                confidence_level = float(decoded_prediction[2])
                predictions.append({'id': class_id, 'name': class_name, 'confidence': confidence_level})
        return predictions