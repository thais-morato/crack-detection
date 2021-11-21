from enum import Enum

class SetEnum(Enum):
    train = 0
    validation = 1
    test = 2

TARGET_IMAGE_SIZE = (224, 224)

AP_PATH_DATASET = "./dataset"
AP_FOLDER_ANOMALOUS = "crack"
AP_FOLDER_NORMAL = "no-crack"
AP_FOLDER_TRAIN = "train"
AP_FOLDER_VALIDATION = "validation"
AP_FOLDER_TEST = "test"

AP_FOLDER_MODELS = "./models"