import torch
import argparse
import warnings
import random
from client.SGRepGenerator import *
from plots.plots import *
import constants
from utils.utils import *
import time
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

def run(args):
    create_directories_for_saving()

    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    run_name = args.run_name

    models = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Test Fold {fold}')
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        X_test_source, X_test_target1, X_test_target2 = source_data[test_index], target_data1[test_index], target_data2[test_index]
        
        models.append([])
        for client in range(constants.NUMBER_CLIENTS):
            model = SGRepGenerator(client_resolution[client], fold, client)
            model.set_generator_saving_path(f'{constants.SAVING_FOLDER_PATH}/{run_name}/weights/weight_run{fold}_client{client}_replica0_generator_{client_resolution[client]}.model')
            models[fold].append(model)

            if client_resolution[client] == 160:
                models[fold][client].test(X_test_source, X_test_target1)
            elif client_resolution[client] == 268:
                models[fold][client].test(X_test_source, X_test_target2)

        save_mae_to_file(models, fold, run_name)

    save_average_mae_to_file(models, current_run_name=run_name)




if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-alg', "--algorithm", default="baseline")

    # directories
    parser.add_argument('-global_run_name', "--global_run_name", default="global_run_name")
    parser.add_argument('-run_name', "--run_name", default="run_name")

    args = parser.parse_args()

    constants.ALGORITHM = args.algorithm

    constants.GLOBAL_RUN_NAME = args.global_run_name
    constants.RUN_NAME = args.run_name

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))

    print("Global run name: {}".format(args.global_run_name))
    print("Run name: {}".format(args.run_name))
    print("Paths: {}".format(args.paths))
    print("Saving folder path: {}".format(constants.SAVING_FOLDER_PATH))

    print("=" * 50)

    run(args)

    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")