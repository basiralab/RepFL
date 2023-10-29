import torch
import copy
from torch_geometric.data import DataLoader
import copy
import random
from models.generators import Generator1, Generator2
from utils.preprocess import*
import constants
from utils.utils import *
from dataset.dataset_brain_connectomes import MultiResolutionBrainConnectomeDataset

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

class SGRepGenerator():
    def __init__(self, resolution, run_index, client_index, is_replica=False, replica_index=0, fed_alg="FedAvg", training=True):
        self.generator_save_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights/weight_run{run_index}_client{client_index}_replica{replica_index}_generator.model'
        self.resolution = resolution
        self.is_replica = is_replica

        self.replicas_list = []
        if fed_alg == "RepFL" and training:
            if is_replica == False:
                for new_replica_index in range(constants.NUMBER_REPLICAS):
                    self.replicas_list.append(SGRepGenerator(resolution=resolution, run_index=run_index, client_index=client_index, is_replica=True, 
                                                            replica_index=new_replica_index, fed_alg=fed_alg))

        self.client_index = client_index
        self.replica_index = replica_index

        self.fed_alg = fed_alg

        self.source_test_sample = None
        self.target_test_sample = None
        self.predicted_test_sample = None

        self.mae_test = None

        if self.resolution == 160:
            self.generator = Generator1()
        elif self.resolution == 268:
            self.generator = Generator2()

        self.generator.to(device)

        self.generator_optimizer = torch.optim.SGD(self.generator.parameters(), lr=constants.LR)

        old_grad = copy.deepcopy(self.generator)
        for param in old_grad.parameters():
            param.data = torch.zeros_like(param.data)
        self.old_grad = get_mdl_params([old_grad])[0]

    def train(self):
        self.generator.train()

        global_loss_train = []
        l1_loss = torch.nn.L1Loss()
        l1_loss.to(device)

        for epochs in range(constants.NUM_LOCAL_EPOCHS):
            global_loss_train_current_epoch = []

            with torch.autograd.set_detect_anomaly(True):
                for batch_idx, batch_samples in enumerate(self.data_loader):
                    source, target_x = batch_samples['source'], batch_samples['target_x']

                    self.generator_optimizer.zero_grad()

                    source = source.to(device)
                    predicted_x = self.generator(source)
                    torch.cuda.empty_cache()

                    target_x = target_x.to(device)
                    global_loss = l1_loss(target_x, predicted_x)
                    torch.cuda.empty_cache()
                    
                    global_loss.backward()
                    self.generator_optimizer.step()
                    torch.cuda.empty_cache()

                    global_loss_train_current_epoch.append(global_loss.detach().cpu().numpy())
                    torch.cuda.empty_cache()

                global_loss_train.append(np.mean(global_loss_train_current_epoch))

        torch.cuda.empty_cache()

        return global_loss_train

    def test(self, X_test_source, X_test_target):
        X_casted_test_source = convert_list_of_vectors_to_graphs(X_test_source, resolution=35)
        X_casted_test_target = convert_list_of_vectors_to_graphs(X_test_source, resolution=self.resolution)

        dataset = MultiResolutionBrainConnectomeDataset(X_casted_test_source, X_casted_test_target, self.resolution)
        data_loader = DataLoader(dataset, batch_size=1)

        self.generator.load_state_dict(torch.load(self.generator_save_path), strict=False)
        self.generator.eval()

        mae_test = []

        for batch_idx, batch_samples in enumerate(data_loader):
            source, target_x, source_x = batch_samples['source'], batch_samples['target_x'], batch_samples['source_x']

            source = source.to(device)
            predicted_x = self.generator(source)

            torch.cuda.empty_cache()

            source_x = source_x.detach().cpu().clone().numpy()
            target_x = target_x.detach().cpu().clone().numpy()
            predicted_x = predicted_x.detach().cpu().clone().numpy()

            torch.cuda.empty_cache()

            residual_error = np.abs(target_x - predicted_x)
            residual_error_mean = np.mean(residual_error)
            mae_test.append(residual_error_mean)

        mean_mae_test = np.mean(mae_test)

        self.source_test_sample = source_x
        self.target_test_sample = target_x
        self.predicted_test_sample = predicted_x

        self.mae_test = mean_mae_test

        return mean_mae_test
    
    def run_one_round(self):
        global_loss_train = self.train()
        self.no_epochs_trained += constants.NUM_LOCAL_EPOCHS

        print(f'Epochs trained: {self.no_epochs_trained}')

        # Run one round replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].run_one_round()

        # Aggregate the replicas
        if len(self.replicas_list) > 0:
            self.aggregate_replicas()

        return global_loss_train
    
    def aggregate_replicas(self):
        model_state = self.generator.state_dict()

        # Anchors and replicas are assigned equal weights
        weight = 1. / (1 + constants.NUMBER_REPLICAS)
        for name, param in self.generator.named_parameters():
            if name in constants.aggregating_layers:
                replicas_updates = [weight * copy.deepcopy(replica.get_generator().state_dict()[name].data) for replica in self.replicas_list]
                replicas_updates.append(weight * copy.deepcopy(self.generator.state_dict()[name].data))
                averaged_update = torch.stack(replicas_updates).mean(dim=0)
                model_state[name] = averaged_update

        # Update the replica aggregated model with the aggregated parameters
        self.generator.load_state_dict(model_state)

    def save_model(self):
        torch.save(self.generator.state_dict(), self.generator_save_path)

    def set_generator_saving_path(self, new_path):
        self.generator_save_path = new_path
    
    def get_source_target_prediction_sample(self):
        return self.source_test_sample, self.predicted_test_sample, self.target_test_sample
    
    def get_mae_test(self):
        return self.mae_test
    
    def get_generator(self):
        return self.generator
    
    def get_replicas_list(self):
        return self.replicas_list
    
    def initialize_model_parameters(self, global_model_state_dict):
        self.generator.load_state_dict(global_model_state_dict)

        # Initialize model parameters for replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].initialize_model_parameters(global_model_state_dict)

    def update_parameters(self, global_model):
        generator_new_state_dict = copy.deepcopy(self.generator.state_dict())
        for param_name in self.generator.state_dict():
            if param_name in constants.aggregating_layers:
                generator_new_state_dict[param_name].data = copy.deepcopy(global_model.state_dict()[param_name])
                
        self.generator.load_state_dict(generator_new_state_dict)

        for params in self.generator.parameters():
            params.requires_grad = True

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].update_parameters(global_model)

    def update_parameters_daisy_chain(self, new_model):
        generator_new_state_dict = copy.deepcopy(self.generator.state_dict())
        for param_name in self.generator.state_dict():
            if param_name in constants.aggregating_layers_daisy_chain and param_name in new_model.state_dict().keys():
                generator_new_state_dict[param_name].data = copy.deepcopy(new_model.state_dict()[param_name])
                
        self.generator.load_state_dict(generator_new_state_dict)

        for params in self.generator.parameters():
            params.requires_grad = True
    
    def set_training_data(self, X_train_source, X_train_target, no_perturbed_samples, offset_perturbed_samples = 0):
        X_train_source = copy.deepcopy(X_train_source)
        X_train_target = copy.deepcopy(X_train_target)

        if self.is_replica:
            # Remove a number of samples from the original dataset starting with offset
            if offset_perturbed_samples >= len(X_train_source):
                offset_perturbed_samples = offset_perturbed_samples % len(X_train_source)

            replica_indices_1 = [i for i in range(0, offset_perturbed_samples)]
            replica_indices_2 = [i for i in range(offset_perturbed_samples + no_perturbed_samples, len(X_train_source))]
            replica_indices = replica_indices_1 + replica_indices_2

            X_train_source = X_train_source[replica_indices]
            X_train_target = X_train_target[replica_indices]
        
        X_casted_train_source = convert_list_of_vectors_to_graphs(X_train_source, resolution=35)
        X_casted_train_target = convert_list_of_vectors_to_graphs(X_train_source, resolution=self.resolution)

        dataset = MultiResolutionBrainConnectomeDataset(X_casted_train_source, X_casted_train_target, self.resolution)
        self.data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE)

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                no_perturbed_samples_next_depth = int((constants.PERCENTAGE_PERTURBED_SAMPLES / 100.0) * len(X_train_source))
                self.replicas_list[replica].set_training_data(X_train_source, X_train_target, no_perturbed_samples = no_perturbed_samples_next_depth, 
                                                              offset_perturbed_samples = replica * no_perturbed_samples_next_depth)