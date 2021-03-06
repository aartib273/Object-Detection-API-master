import os

# The temporary directory to store the image files while processing
TMP_UPLOAD_FOLDER = os.getenv('TMP_UPLOAD_FOLDER', "/tmp/input/")

# Model version should be incremented only when the prediction output was changed.
# If the model version was incremented it is an indicator that the images should be reprocess
MODEL_VERSION = '0.1.0'

TRANSLATIONS_DIR = "translations"

SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "mpo"]

# Maximum image size
ONE_MB = 1024 * 1024
MAX_IMAGE_SIZE = os.getenv('MAX_IMAGE_SIZE', 20 * ONE_MB)

# EfficentNetB7 settings
EFFICIENTNETB7_WEIGHTS = 'weights/efficientnetb7.h5'
EFFICIENTNETB7_CLASSES = 'weights/imagenet_class_index.json'


# YoloV4 settings
YOLOV4_WEIGHTS = 'weights/yolov4.weights'
YOLOV4_CLASSES = 'data/coco.names'

AGGREGATE_CONFIDENCE_THRESHOLD = 0.5

# Prometheus
PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', True)
PROMETHEUS_PORT = os.getenv('PROMETHEUS_PORT', 8126)


ODS_LOG_LEVEL = os.getenv('ODS_LOG_LEVEL', "INFO")