import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

class AmpDataset(Dataset):

    def __init__(self, states, actions, next_states, device=torch.device('cpu')):
        '''
        Pytorch Dataset class for the offline dataset collected for the AMP environment. Note we return (s,a,s') triples
        :param state: (torch Tensor) tensor with shape (number of samples, state dimension) with state data
        :param action: (torch Tensor) tensor with shape (number of samples, action dimension) with action data
        :param next_state: (torch Tensor) tensor with shape (number of samples, state dimension) with next state data
        :param device: (torch Device) device for pytorch.
        '''

        self.device = device
        self.states = states
        self.actions = actions
        self.next_states = next_states


    def get_transformations(self, device=None):
        '''
        Returns the mean and scales of the states, actions, and delta between the next state and current state.
        '''
        diff = self.next_states - self.states

        # Compute Means
        state_mean = self.states.mean(dim=0).float().requires_grad_(False)
        action_mean = self.actions.mean(dim=0).float().requires_grad_(False)
        diff_mean = diff.mean(dim=0).float().requires_grad_(False)

        # Compute Scales
        state_scale = torch.abs(
            self.states - state_mean).mean(dim=0).float().requires_grad_(False) + 1e-8
        action_scale = torch.abs(
            self.actions - action_mean).mean(dim=0).float().requires_grad_(False) + 1e-8
        diff_scale = torch.abs(
            diff - diff_mean).mean(dim=0).float().requires_grad_(False) + 1e-8
        dev = self.device if device is None else device
        return state_mean.to(dev), state_scale.to(dev), action_mean.to(dev), \
            action_scale.to(dev), diff_mean.to(dev), diff_scale.to(dev)

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return self.states[idx].float(), self.actions[idx].float(), self.next_states[idx].float()
    
class AgentReplayBuffer(Dataset):
    '''Used primarily in trying to make online GAIL work.'''
    def __init__(self, states: Optional[torch.Tensor]=None, next_states: Optional[torch.Tensor]=None, device=torch.device('cpu'), max_size=100000):
        self.device = device
        
        self.states = states
        if self.states is not None:
            self.states = states.to(device)
        
        self.next_states = next_states
        if self.next_states is not None:
            self.next_states = next_states.to(device)
            
        self.max_size = max_size
        
    def __len__(self):
        if self.states is None:
            return 0
        return self.states.size(0)
    
    def __getitem__(self, idx):
        if self.states is None:
            raise Exception('Not possible to select from empty dataset.')
        state = self.states[idx]
        next_state = self.next_states[idx]
        return state, next_state
    
    def add(self, states, next_states):
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        
        if self.states is None:
            self.states = states
        else:
            self.states = torch.cat([self.states, states], dim=0)
        
        if self.next_states is None:
            self.next_states = next_states
        else:
            self.next_states = torch.cat([self.next_states, next_states], dim=0)
            
        if self.states.size(0) > self.max_size:
            self.states = self.states[-self.max_size:]
        
        if self.next_states.size(0) > self.max_size:
            self.next_states = self.next_states[-self.max_size:]
        
    def sample(self, batch_size):
        if self.states is None:
            raise Exception('Not possible to sample from empty dataset.')
        idxs = np.random.randint(0, self.states.size(0), batch_size)
        states = self.states[idxs]
        next_states = self.next_states[idxs]
        return torch.cat([states, next_states], dim=-1) # so then we can actually do cost stuff immediately
    