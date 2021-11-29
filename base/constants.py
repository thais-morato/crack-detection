from enum import Enum

class SetEnum(Enum):
    train = 0
    validation = 1
    test = 2

TRAIN_OPTION_SCRATCH = 'scratch'
TRAIN_OPTION_RESUME = 'resume'
OPTION_CANCEL = 'cancel'

TARGET_IMAGE_SIZE = (224, 224)

AP_PATH_DATASET = "./dataset"
AP_FOLDER_ANOMALOUS = "crack"
AP_FOLDER_NORMAL = "no-crack"
AP_FOLDER_TRAIN = "train"
AP_FOLDER_VALIDATION = "validation"
AP_FOLDER_TEST = "test"

AP_FOLDER_MODELS = "./models"
AP_FOLDER_CHECKPOINTS = "./checkpoints"
AP_FOLDER_LOG = "./log"

EXTENSION_MODEL = ".h5"
EXTENSION_CHECKPOINT = "-epoch-{epoch:02d}.h5"
EXTENSION_LOG = "-log.csv"