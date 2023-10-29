import sys
import os

import torch
import argparse
import random
import time
import numpy as np
from utils.preprocess import *
from client.SGRepGenerator import *
from server.FederatedServer import *
from plots.plots import *
import constants
from utils.utils import *

from collections import Counter

# Add the parent directory to the sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# warnings.filterwarnings("ignore")
# torch.cuda.empty_cache()

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

    if constants.ALGORITHM == "baseline":
        run_baseline(args)
    elif constants.ALGORITHM == "fedavg":
        run_fedavg(args)
    elif constants.ALGORITHM == "feddyn":
        run_feddyn(args)
    elif constants.ALGORITHM == "feddc":
        run_feddc(args)
    elif constants.ALGORITHM == "repfl":
        run_repfl(args)


def plot_losses(losses, method):
    for client in range(constants.NUMBER_CLIENTS):
        global_loss = [losses[fold][client]['global_test1'] for fold in range(constants.NUMBER_FOLDS)]
        global_loss = np.array(global_loss)
        average_global_loss = np.mean(global_loss, axis = 0)

        plot_one_loss(average_global_loss, client, method=method, current_run_name=constants.RUN_NAME)

def plot_losses_one_fold(losses, method, fold=0):
    for client in range(constants.NUMBER_CLIENTS):
        global_loss = [losses[fold][client]['global_test1']]
        global_loss = np.array(global_loss)
        average_global_loss = np.mean(global_loss, axis = 0)

        plot_one_loss(average_global_loss, client, method=method, current_run_name=constants.RUN_NAME)


def test_method(models_fed):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    for fold in range(constants.NUMBER_FOLDS):
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        X_test_source, X_test_target1, X_test_target2 = source_data[test_index], target_data1[test_index], target_data2[test_index]
        
        for client in range(constants.NUMBER_CLIENTS):
            if client_resolution[client] == 160:
                models_fed[fold][client].test(X_test_source, X_test_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].test(X_test_source, X_test_target2)

        save_mae_to_file(models_fed, fold, constants.RUN_NAME)


# # ************   Baseline   ************

def run_baseline(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses = initialize_losses_arrays()
    models = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Baseline Fold {fold}')
        current_fold_configuration = fold_configuration[fold]

        models.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="Baseline"))
            if client_resolution[client] == 160:
                models[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Baseline Fold {fold} Client {client}')
                global_loss_train = models[fold][client].run_one_round()

                losses[fold][client]['global_test1'].extend(global_loss_train)

        for client in range(constants.NUMBER_CLIENTS):
            models[fold][client].save_model()

        save_losses_to_file(losses, fold, 'global_test1')
    
    test_method(models, args)
    plot_losses(models, method="baseline")


# # ************   FedAvg   ************

def run_fedavg(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Federated Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="FedAvg"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            global_model = federated_server.federated_average_intermodality(models_fed[fold])
            if round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedavg")


# # ************   FedDyn   ************

def run_feddyn(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'FedDyn Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="FedDyn"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            federated_server.update_global_model_feddyn(models_fed[fold])

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="feddyn")


# # ************   FedDC   ************

def run_feddc(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'FedDC Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]
        
        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]
            train_datasets.append([X_train_source, X_train_target1, X_train_target2])

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, "FedDC"))

        for epoch in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                X_train_source, X_train_target1, X_train_target2 = train_datasets[client]
                if client_resolution[client] == 160:
                    l1_loss_train, global_loss_train = models_fed[fold][client].train(X_train_source, X_train_target1)
                elif client_resolution[client] == 268:
                    l1_loss_train, global_loss_train = models_fed[fold][client].train(X_train_source, X_train_target2)

                losses_fed[fold][client]['l1_test1'].extend(l1_loss_train)
                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            if epoch % constants.DC_ROUND == constants.DC_ROUND - 1:
                new_models = federated_server.federated_daisy_chain(models_fed[fold])
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters_daisy_chain(new_models[client])

            if epoch % constants.AGG_ROUND == constants.AGG_ROUND - 1:
                print(f'FedDC Fold {fold} Client {client} Epoch {epoch}')
                global_model = federated_server.federated_average_intermodality(models_fed[fold])

                if epoch % constants.AGG_ROUND < int(constants.NUMBER_OF_ROUNDS / 10) - 1:
                    for client in range(constants.NUMBER_CLIENTS):
                        models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="feddc")


# # ************   RepFL   ************

def run_repfl(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Federated Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]
        
        models_fed.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="RepFL"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1, no_perturbed_samples = 0, offset_perturbed_samples = 0)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2, no_perturbed_samples = 0, offset_perturbed_samples = 0)

        for current_round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                l1_loss_train, global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['l1_test1'].extend(l1_loss_train)
                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            global_model = federated_server.federated_average_intermodality(models_fed[fold])

            if current_round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(global_model)

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="repfl")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-alg', "--algorithm", default="baseline")
    parser.add_argument('-seed', "--seed", type=int, default=1773)
    parser.add_argument('-le', "--local_epochs", type=int, default=10)
    parser.add_argument('-r', "--rounds", type=int, default=10)
    parser.add_argument('-batch', "--batch_size", type=int, default=5)
    parser.add_argument('-folds', "--folds", type=int, default=5)

    # directories
    parser.add_argument('-global_run_name', "--global_run_name", default="global_run_name")
    parser.add_argument('-run_name', "--run_name", default="run_name")

    # RepFL
    parser.add_argument('-perturbed', "--no_perturbed_samples", type=int, default=30)
    parser.add_argument('-replicas', "--no_replicas", type=int, default=5)

    args = parser.parse_args()

    constants.ALGORITHM = args.algorithm
    constants.MANUAL_SEED = args.seed
    constants.NUM_LOCAL_EPOCHS = args.local_epochs
    constants.NUMBER_OF_ROUNDS = args.rounds
    constants.BATCH_SIZE = args.batch_size
    constants.NUMBER_FOLDS = args.folds
    constants.GLOBAL_RUN_NAME = args.global_run_name
    constants.RUN_NAME = args.run_name
    constants.PERCENTAGE_PERTURBED_SAMPLES = args.no_perturbed_samples
    constants.NUMBER_REPLICAS = args.no_replicas

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Random seed: {}".format(args.seed))
    print("Local epochs: {}".format(args.local_epochs))
    print("Rounds: {}".format(args.rounds))
    print("Batch: {}".format(args.batch_size))
    print("Folds: {}".format(args.folds))
    print("Global run name: {}".format(args.global_run_name))
    print("Run name: {}".format(args.run_name))
    print("Saving folder path: {}".format(constants.SAVING_FOLDER_PATH))
    print("Perturbed: {}".format(args.no_perturbed_samples))
    print("Replicas: {}".format(args.no_replicas))

    print("=" * 50)

    create_directories_for_saving()
    write_details_to_file(args, constants.RUN_NAME)

    run(args)

    time_cost = round(time.time()-total_start, 2)
    print(f"\nTotal time cost: {time_cost}s.")
    write_time_to_file(time_cost, constants.RUN_NAME)