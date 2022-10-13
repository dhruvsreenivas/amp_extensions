import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import os

import time
#From https://github.com/jdchang1/milo/blob/main/milo/milo/dynamics_model/mlp_dynamics.py

import sys
class DynamicsEnsemble():
    def __init__(self,
                 state_dim,
                 action_dim,
                 train_dataset,
                 validate_dataset,
                 num_models=4,
                 batch_size=256,
                 hidden_sizes=[512,512],
                 use_resnet=False,
                 dense_connect=True,
                 activation='relu',
                 transform=True,
                 optim_args={'optim': 'sgd', 'lr': 1e-4, 'momentum': 0.9},
                 device=torch.device('cpu'),
                 base_seed=100,
                 num_workers=1):
        '''
        Constructor for Dynamics Ensemble

        Parameter state_dim: (int) Input size for models
        Parameter action_dim: (int) Output size for models
        Parameter train_dataset: Dataset used to train dynamics models. This dataset should be an instance of torch.utils.data.Dataset
        Parameter validate_dataset: Validation dataset. If None, no validation done. Should be an instance of torch.utils.data.Dataset
        Parameter num_models: (int) Number of models to train in the ensemble. Only difference between models is the seed.
        Parameter batch size: (int) Batch size used in training
        Parameter hidden sizes: (list of int) Specifies number and size of hidden layers
        Parameter dense_connect: (bool) If true, models used will be DenseNet-style MLPs.
        Parameter activation: (String) Activation function used in network. Currently, if not 'relu', activation will be 'tanh'.
        Parameter transform: (bool) If true, normalize input and output.
        Parameter optim_args: (dict) Arguments used for optimizer
        Parameter device: Location to load models and data. This is also where training will be done.
        Parameter base_seed: (int) Seed used for seeding numpy, torch, and ensembles
        Parameter num_workers: (int) Number of workers for dataloader
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                           pin_memory=True)

        self.transformations = train_dataset.get_transformations(device) if transform else None

        if validate_dataset:
            self.validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
        self.num_models = num_models
        self.transform = transform
        self.device = device
        self.base_seed = base_seed
        self.models = [DynamicsModel(state_dim,
                                     action_dim,
                                     hidden_sizes=hidden_sizes,
                                     use_resnet=use_resnet,
                                     dense_connect=dense_connect,
                                     activation=activation,
                                     transform=transform,
                                     optim_args=optim_args,
                                     device=device,
                                     seed=base_seed+k) for k in range(num_models)]
        self.threshold = 0.0

    def train(self, epochs, validate=False, logger=None, log_epoch=False, grad_clip=0, save_path=None, save_checkpoints=False, writer=None):
        '''
        Train dynamics ensemble
        '''
        assert (not validate) or (validate and self.validate_dataloader)
        model_train_info = []
        for i, model in enumerate(self.models):
            if logger is not None:
                logger.info(f'>>>>Training Model {i+1}/{self.num_models}')
            if save_path is not None:
                model_checkpoint_path=os.path.join(save_path, f'model{i}')
                if not os.path.exists(model_checkpoint_path):
                    os.mkdir(model_checkpoint_path)
            info = model.train(epochs,
                               self.train_dataloader,
                               transformations=self.transformations,
                                validate=validate,
                               validate_dataloader=self.validate_dataloader if validate else None,
                               logger=logger,
                               log_epoch=log_epoch,
                               grad_clip=grad_clip,
                               save_path=model_checkpoint_path if save_path else None ,
                               save_checkpoints=save_checkpoints,
                               writer=writer)

            model_train_info.append(info)
        return model_train_info

    def save_ensemble(self, save_path):
        '''
        Saves the ensemble state as a list of dictionaries. Each dictionary corresponds to a model and
        contains the state dictionary for the model and optimizer.
        '''
        state_dicts = [model.get_state_dicts() for model in self.models]
        torch.save(state_dicts, save_path)

    def load_ensemble(self, state_dict_path):
        '''
        Loads ensemble and the weights for both the model and optimizer.
        '''
        state_dicts = torch.load(state_dict_path, map_location=self.device)
        assert len(state_dicts)==len(self.models)
        print("loading ensemble")
        for model, state_dict in zip(self.models, state_dicts):
            model.load(state_dict['model'], state_dict['optim'])
        print("Done loading ensemble")
        if self.transform:
                    for model in self.models:
                        model.state_mean, model.state_scale, model.action_mean, model.action_scale, \
                            model.diff_mean, model.diff_scale = self.transformations
        #Load in the state, action

    def compute_discrepancy(self, state, action):
        """
        Computes the maximum discrepancy for a given state and action
        """
        with torch.no_grad():
            preds = torch.cat([model.forward(state, action).unsqueeze(0) for model in self.models], dim=0)
        disc = torch.cat([torch.norm(preds[i] - preds[j], p=2, dim=1).unsqueeze(0) \
                          for i in range(preds.shape[0]) for j in range(i + 1, preds.shape[0])],
                         dim=0)  # (n_pairs*batch)
        return disc.max(0).values.to(torch.device('cpu'))

    def compute_threshold(self):
        '''
        Computes the maximum discrepancy for the current ensemble for an entire offline dataset
        '''
        results = []
        for state, action, _ in self.train_dataloader:
            results.append(self.compute_discrepancy(state, action))
        self.threshold = torch.cat(results, dim=0).max().item()

    def get_action_discrepancy(self, state, action):
        """
        Computes the discrepancy of a given (s,a) pair
        """
        # Add Batch Dimension
        if len(state.shape) == 1: state.unsqueeze(0)
        if len(action.shape) == 1: action.unsqueeze(0)


        state = state.float().to(self.device)
        action = action.float().to(self.device)
        return self.compute_discrepancy(state, action)

class DynamicsModel():
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes=[512,512],
                 use_resnet=False,
                 dense_connect=True,
                 activation='relu',
                 transform=True,
                 optim_args={'optim': 'sgd', 'lr': 1e-4, 'momentum': 0.9},
                 device=torch.device('cpu'),
                 seed=100):
        '''
        Constructor for a dynamics model. For details about the parameters, see specification for DynamicsEnsemble.

        The dynamics model predicts the DIFFERENCE in state between the next state and current state, not the next state.
        '''
        # Set Seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transform = transform

        self.device = device
        if use_resnet:
            self.model = ResidualMLP(state_dim + action_dim, state_dim, hidden_sizes, activation).to(self.device)
        else:
            self.model = BasicMLP(state_dim+action_dim, state_dim, hidden_sizes, dense_connect, activation).to(self.device)

        self.loss_fn = nn.MSELoss().to(self.device)
        if optim_args['optim'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optim_args['lr'], momentum=optim_args['momentum'], nesterov=True)
        elif optim_args['optim'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_args['lr'], eps=optim_args['eps'])
        else:
            assert False, 'Use valid optimizer'

    def get_grad_norms(self):
        '''
        Returns the norm of the gradient of the model parameters.
        '''
        params = [p for p in self.model.parameters() if p.grad is not None]
        if len(params) == 0:
            return 0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
        return total_norm.detach().cpu().item()

    def forward(self, state, action, unnormalize_out=True):
        #Convert state and action to tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        #Move to device and normalize
        state, action = state.to(self.device), action.to(self.device)
        if self.transform:
            state = (state - self.state_mean) / (self.state_scale)
            action = (action - self.action_mean) / (self.action_scale)

        #Predict the difference in state
        state_diff = self.model.forward(torch.cat([state, action], dim=1))
        if self.transform and unnormalize_out: #Need both since self.transform means NN predicts normalized outputs so we unnormalize only when this is true.
            state_diff = (state_diff * (self.diff_scale)) + self.diff_mean
        return state_diff


    def train_step(self, grad_clip, state, action, next_state):
        '''
        Perform one train step with the batch of state, action, and next_state.
        '''
        self.optimizer.zero_grad()
        pred = self.forward(state, action, unnormalize_out=False) #We'll always be training with normalized inputs/outputs
        target = (next_state - state).to(self.device)
        if self.transform:
            target = (target - self.diff_mean) / (self.diff_scale)
        loss = self.loss_fn(pred, target)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        return loss.item()

    def validate_step(self, state, action, next_state):
        '''
        Perform one validation step.
        '''
        pred = self.forward(state, action, unnormalize_out=False)
        target = (next_state - state).to(self.device)
        if self.transform:
            target = (target - self.diff_mean) / (self.diff_scale)
        loss = self.loss_fn(pred, target)
        return loss.item()

    def train_epoch(self, epoch, train_dataloader, epochs, logger=None, log_epoch=False, grad_clip=0, save_path=None):
        '''
        Train the model for one epoch of the dataset.
        '''
        self.model.train()  # Set train mode
        epoch_train_loss = []
        for state, action, next_state in train_dataloader:
            train_loss = self.train_step(grad_clip, state, action, next_state)
            epoch_train_loss.append(train_loss)

        epoch_average_loss = np.average(epoch_train_loss)
        if save_path is not None:
            if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs // 2:
                torch.save({'model': self.model.state_dict(),
                            'optim': self.optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'loss': epoch_average_loss}, os.path.join(save_path, f'{epoch + 1}_checkpoint.pt'))
        if logger is not None:
            if log_epoch: logger.info('Epoch: {}, Train Loss: {}'.format(epoch, epoch_average_loss))

        else:
            print('Epoch {} Train Loss: {}'.format(epoch, epoch_average_loss))
        return epoch_average_loss

    def validate_epoch(self, epoch, validate_dataloader, logger=None, log_epoch=False):
        '''
        Validate the model on the entire validation dataset.
        '''
        assert validate_dataloader is not None, "Need to set validation dataset"
        self.model.eval()
        epoch_validate_loss = []
        with torch.no_grad():
            for state, action, next_state in validate_dataloader:
                validate_loss = self.validate_step(state, action, next_state)
                epoch_validate_loss.append(validate_loss)
        epoch_average_loss = np.average(epoch_validate_loss)
        if logger is not None:
            if log_epoch: logger.info('Epoch {} Validation Loss: {}'.format(epoch, epoch_average_loss))
        else:
            print('Epoch {} Validation Loss: {}'.format(epoch, epoch_average_loss))
        return epoch_average_loss

    def train(self, epochs, train_dataloader, transformations = None, validate=False, validate_dataloader = None, logger=None, log_epoch=False, grad_clip=0, save_path=None, save_checkpoints=False, writer=None):
        '''
        Train the model
        '''

        #Set transformations
        if self.transform:
            assert transformations is not None
            self.state_mean, self.state_scale, self.action_mean, self.action_scale, self.diff_mean, self.diff_scale = transformations

        train_min_loss, train_min_epoch = float('inf'), float('inf')
        best_train_model_state_dict, best_train_optim_state_dict = None, None
        train_losses = []

        if validate:
            validate_min_loss, validate_min_epoch = float('inf'), float('inf')
            best_validate_model_state_dict, best_validate_optim_state_dict = None, None
            validate_losses = []

        #Training loop
        for epoch in tqdm(range(epochs)):
            train_epoch_average_loss = self.train_epoch(epoch, train_dataloader, epochs, logger, log_epoch, grad_clip, save_path if save_checkpoints else None )
            train_losses.append(train_epoch_average_loss)
            if writer is not None:
                writer.add_scalar("Loss/train", train_epoch_average_loss, epoch)
            if train_epoch_average_loss < train_min_loss:
                #Keep track of model with lowest training loss
                train_min_epoch = epoch
                train_min_loss = train_epoch_average_loss
                best_train_model_state_dict = self.model.state_dict()
                best_train_optim_state_dict = self.optimizer.state_dict()

            #Validation step
            if validate:

                validate_epoch_average_loss = self.validate_epoch(epoch, validate_dataloader, logger, log_epoch)
                validate_losses.append(validate_epoch_average_loss)
                if writer is not None:
                    writer.add_scalar("Loss/validate", validate_epoch_average_loss, epoch)

                if validate_epoch_average_loss < validate_min_loss:
                    #Keep track of model with lowest validation lloss
                    validate_min_epoch = epoch
                    validate_min_loss = validate_epoch_average_loss
                    best_validate_model_state_dict = self.model.state_dict()
                    best_validate_optim_state_dict = self.optimizer.state_dict()

        #Save final model and best train/validation models
        if save_path is not None:
            torch.save({'model': self.model.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'epoch': epochs,
                        'loss': train_epoch_average_loss}, os.path.join(save_path, 'final_model.pt'))
            torch.save({'model': best_train_model_state_dict,
                        'optim': best_train_optim_state_dict,
                        'epoch': train_min_epoch + 1,
                        'loss': train_min_loss}, os.path.join(save_path, 'best_train_checkpoint.pt'))
            if validate:
                torch.save({'model': best_validate_model_state_dict,
                            'optim': best_validate_optim_state_dict,
                            'epoch': validate_min_epoch+1,
                            'loss': validate_min_loss}, os.path.join(save_path, 'best_validate_checkpoint.pt'))

        #After training, load in model with best training loss
        self.model.load_state_dict(best_train_model_state_dict)
        self.optimizer.load_state_dict(best_train_optim_state_dict)


        if logger is not None:
            logger.info('Train: Dynamics Start | Best Loss | Epoch: {} | {} | {}'.format(train_losses[0], train_min_loss, train_min_epoch))
            if validate:
                logger.info('Validate: Dynamics Start | Best Loss | Epoch: {} | {} | {}'.format(validate_losses[0], validate_min_loss, validate_min_epoch))

        return train_min_loss, train_losses[0]

    def load(self, model_state_dict, optimizer_state_dict = None):
        '''
        Load in the weights for the model and optimizer
        '''
        self.model.load_state_dict(model_state_dict)
        if optimizer_state_dict:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def get_state_dicts(self):
        '''
        Save weights for model and optimizer.
        '''
        return {'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}

class BasicMLP(nn.Module):
    """
    MLP Dynamics model implementation for Amp specifically. Had to make special class instead of just using
    nn.Sequential due to using densenet implementation.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=[64, 64],
                 dense_connect=False,
                 activation='relu'):
        super(BasicMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense_connect = dense_connect
        self.nonlinearity = torch.relu if activation == 'relu' else torch.tanh

        self.layer_sizes = [input_dim,] + hidden_sizes + [output_dim,]
        layers = []
        for i in range(len(self.layer_sizes)-1):
            layer_input_size = self.layer_sizes[i]
            if self.dense_connect:
                for j in range(0, i):
                    layer_input_size += self.layer_sizes[j]
            layers.append(nn.Linear(layer_input_size, self.layer_sizes[i+1]))
        self.fc_layers = nn.ModuleList(layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        input = x

        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](input)
            out = self.nonlinearity(out)
            input = torch.cat([input, out], dim = 1) if self.dense_connect else out

        out = self.fc_layers[-1](input)
        return out

class ResidualMLP(nn.Module):
    """
    MLP Dynamics model implementation for Amp specifically. Applying residual connections (addition)
    as opposed to dense connections (concatenation) to the dynamics model.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=[128, 128],
                 activation='relu'):
            
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

        layers = []
        dim = hidden_sizes[0] if hidden_sizes else self.output_dim
        layers.append(nn.Linear(self.input_dim, dim))
        if hidden_sizes:
            for size in hidden_sizes[1:]:
                assert dim == size, "not residual"
                layers.append(self.activation)
                layers.append(nn.Linear(dim, size))
                dim = size
            layers.append(self.activation)
        
        layers.append(nn.Linear(dim, self.output_dim))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float() # TODO add device handling, seems to only be for CPU

        # initial preprocessing
        x = self.layers[0](x)
        x = self.activation(x)

        # processing through residual layers
        for layer in self.layers[1:-1]:
            y = layer(x)
            y = self.activation(y)
            x = x + y
        
        out = self.layers[-1](x)
        return out
    
            
