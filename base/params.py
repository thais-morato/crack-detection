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

MODEL_NAME = "mymodel"
EPOCHS = 25
LEARNING_RATE = 0.001