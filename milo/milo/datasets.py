import torch
from torch.utils.data import Dataset
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