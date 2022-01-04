IMAGE_SIZE = (227, 227)
BATCH_SIZE = 32

PATH_DATABASE = './Concrete Crack Images for Classification'
FOLDER_NORMAL = "Negative"
FOLDER_ANOMALOUS = "Positive"

# Percentage of the samples used for train, validation and test
# Should sum up to 1
PERCENTAGE_TRAIN = 0.7
PERCENTAGE_VALIDATION = 0.15
PERCENTAGE_TEST = 0.15

IS_FINE_TUNING = False
EPOCHS = 25
LEARNING_RATE = 0.001
APPLY_DATA_AUGMENTATION = False
APPLY_PREPROCESSING = True
LOAD_DATASETS_SEED = 11