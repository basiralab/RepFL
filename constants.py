MANUAL_SEED = 1773

NUM_LOCAL_EPOCHS = 10
NUMBER_OF_ROUNDS = 10

BATCH_SIZE = 5

ALPHA_COEF = 0.01
DC_ROUND = 1
AGG_ROUND = 10

NUMBER_FOLDS = 5
NUMBER_CLIENTS = NUMBER_FOLDS - 1
PERCENTAGE_PERTURBED_SAMPLES = 22
NUMBER_REPLICAS = 5

ALGORITHM = None
LR = 0.1
OPTIM = "sgd"

GLOBAL_RUN_NAME = 'replicas_distribution'
RUN_NAME = 'test_plots'

# Add the dataset path and saving folder path
DATASET_FOLDER_PATH = ""
SAVING_FOLDER_PATH = ""

aggregating_layers = ["conv1.bias", "conv1.nn.0.weight", "conv1.nn.0.bias", "conv1.lin.weight", "conv1.root"]

aggregating_layers_daisy_chain = ["conv1.bias", "conv1.nn.0.weight", "conv1.nn.0.bias", "conv1.lin.weight", "conv1.root",
                                "conv2.bias", "conv2.nn.0.weight", "conv2.nn.0.bias", "conv2.lin.weight", "conv2.root",
                                "conv3.bias", "conv3.nn.0.weight", "conv3.nn.0.bias", "conv3.lin.weight", "conv3.root"]