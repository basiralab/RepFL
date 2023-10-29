import torch
import copy
import random
from models.generators import Generator1, Generator2
import constants
from utils.utils import *

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

class FederatedServer():
    def __init__(self, run_index):
        self.global_model = Generator2().to(device)
        self.server_state = copy.deepcopy(self.global_model)

        self.generator_save_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights/global_weight_run{run_index}.model'

    def federated_average_intermodality(self, model_list):
        global_model_state = self.global_model.state_dict()
        for name, param in self.global_model.named_parameters():
            if name in constants.aggregating_layers:
                client_updates = [copy.deepcopy(model.get_generator().state_dict()[name].data) for model in model_list]
                averaged_update = torch.stack(client_updates).mean(dim=0)
                # print(averaged_update)
                global_model_state[name] = averaged_update
                # break

        # Update the global model with the aggregated parameters
        self.global_model.load_state_dict(global_model_state)

        return copy.deepcopy(self.global_model)

    def federated_daisy_chain(self, model_list):
        new_models = []

        new_models_permutation = [i for i in range(constants.NUMBER_CLIENTS)]
        random.shuffle(new_models_permutation)

        for new_model_index in new_models_permutation:
            new_models.append(copy.deepcopy(model_list[new_model_index].get_generator()))

        return new_models
    
    def aggregate_parameters_feddyn(self, model_list):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for param_name in model_list[0].get_generator().state_dict():
            if param_name in constants.aggregating_layers:
                self.global_model.state_dict()[param_name].data = torch.stack([model.get_generator().state_dict()[param_name].data.clone() for model in model_list]).sum(0) / len(model_list)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1/constants.ALPHA_COEF) * state_param

    def update_server_state(self, model_list):
        model_delta = copy.deepcopy(self.global_model)
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in model_list:
            for param_name in model_list[0].get_generator().state_dict():
                if param_name in constants.aggregating_layers:
                    model_delta.state_dict()[param_name].data += (client_model.get_generator().state_dict()[param_name] - self.global_model.state_dict()[param_name]) / constants.NUMBER_CLIENTS

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= constants.ALPHA_COEF * delta_param

    def get_global_model(self):
        return copy.deepcopy(self.global_model)
