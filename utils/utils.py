import os
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random
import copy
import constants

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


def read_and_preprocess_files():
    source_data = np.random.normal(0, 0.5, (279, 595))
    target_data1 = np.random.normal(0, 0.5, (279, 12720))
    target_data2 = np.random.normal(0, 0.5, (279, 35778))

    return source_data, target_data1, target_data2

def create_directories_for_saving():
    global_saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}'
    global_saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}'
    if not os.path.exists(global_saving_path):
        os.makedirs(global_saving_path)
        print("Directory created:", global_saving_path)
    else:
        print("Directory already exists:", global_saving_path)

    saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}'
    saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print("Directory created:", saving_path)
    else:
        print("Directory already exists:", saving_path)

    saving_weights_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights'
    saving_weights_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights'
    if not os.path.exists(saving_weights_path):
        os.makedirs(saving_weights_path)
        print("Directory created:", saving_weights_path)
    else:
        print("Directory already exists:", saving_weights_path)

    saving_results_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results'
    saving_results_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results'
    if not os.path.exists(saving_results_path):
        os.makedirs(saving_results_path)
        print("Directory created:", saving_results_path)
    else:
        print("Directory already exists:", saving_results_path)

def initialize_losses_arrays():
    losses = []
    for fold in range(constants.NUMBER_FOLDS):
        losses.append([])
        for client in range(constants.NUMBER_CLIENTS):
            losses_current = {'l1_test1': [], 'pcc_test1': [], 'topological_test1': [], 'global_test1': [], 
                    'l1_test2': [], 'pcc_test2': [], 'topological_test2': [], 'global_test2': [], 
                    'global': []}
            losses[fold].append(losses_current)

    return losses

def initialize_losses_arrays_simple():
    losses = []
    for fold in range(constants.NUMBER_FOLDS):
        losses.append([])
        for client in range(constants.NUMBER_CLIENTS):
            losses_current = {'g': [], 'd': [], 'a': [], 'l1_loss': [], 'pcc_loss': [], 'eigen_loss': [], 'global_test1': []}
            losses[fold].append(losses_current)

    return losses

def generate_fold_configurations(n):
    fold_configs = []
    one_config = [i for i in range(n)]
    fold_configs.append(one_config)

    for i in range(n - 1):
        one_config = one_config[1:] + [one_config[0]]
        fold_configs.append(one_config)
    
    return fold_configs

def generate_fold_indices(source_data):
    kf = KFold(n_splits=constants.NUMBER_FOLDS, shuffle=True, random_state=1773)
    
    fold_indices = []
    for train_index, test_index in kf.split(source_data):
        fold_indices.append(test_index)

    return fold_indices

def generate_fold_indices_dataset(dataset):
    kf = StratifiedKFold(n_splits=constants.NUMBER_FOLDS, shuffle=True, random_state=1773)
    input_list = []
    target_list = []
    for input, target in dataset:
        input_list.append(input)
        target_list.append(target)

    fold_indices = []
    for train_index, test_index in kf.split(input_list, target_list):
        fold_indices.append(test_index)

    return fold_indices

def save_losses_to_file(losses, fold, loss_name):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results/losses_{loss_name}_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        losses_current_client = ""
        for epoch_loss in losses[fold][client][loss_name]:
            losses_current_client += str(epoch_loss) + ","
        
        f.write(losses_current_client)
        f.write('\n')

    f.close()

def save_mae_to_file(models, fold, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        f.write(str(models[fold][client].get_mae_test()))
        f.write('\n')
        
    f.close()

def save_average_mae_to_file(models, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_average_folds.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        mae_folds = []
        for fold in range(constants.NUMBER_FOLDS):
            mae_folds.append(models[fold][client].get_mae_test())
        mae_folds = np.array(mae_folds)
        mae_average = np.mean(mae_folds)

        f.write(str(mae_average))
        f.write('\n')
        
    f.close()

def read_losses_from_file(fold, current_run_name = constants.RUN_NAME, loss_name=None):
    if loss_name != None:
        f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/losses_{loss_name}_fold_{fold}.txt', "r")
    else:
        f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/losses_fold_{fold}.txt', "r")

    losses_fold = []
    for client in range(constants.NUMBER_CLIENTS):
        losses_current_client_as_string = f.readline()
        losses_current_client = losses_current_client_as_string.split(',')
        losses_current_client = losses_current_client[:-1]
        # losses_current_client = np.array(losses_current_client)
        # losses_current_client_float = losses_current_client.astype(np.float)
        losses_current_client_float = np.asarray(losses_current_client, dtype=float)
        
        losses_fold.append(losses_current_client_float)
    
    f.close()

    return losses_fold

def read_mae_from_file(fold, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_fold_{fold}.txt', "r")

    mae_fold = []
    for client in range(constants.NUMBER_CLIENTS):
        mae_current_client_as_string = f.readline()

        if len(mae_current_client_as_string) > 0:
            mae_current_client_float = np.asarray(mae_current_client_as_string, dtype=float)
            mae_fold.append(mae_current_client_float)
        
    
    f.close()

    return mae_fold

def write_details_to_file(args, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/run_details.txt', "a")

    # Iterate through all parsed arguments
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        f.write(f"Argument '{arg_name}': {arg_value}\n")
        
    f.close()

def write_time_to_file(time_cost, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/run_details.txt', "a")

    f.write(f"Time cost: {time_cost}\n")
        
    f.close()