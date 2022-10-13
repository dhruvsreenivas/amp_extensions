import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

def disc_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class Discriminator(nn.Module):
    '''Discriminator from AMP repo.'''
    def __init__(self, input_dim, hidden_sizes=[1024, 512], output_dim=1, activation='relu'):
        super().__init__()
        
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        
        if not hidden_sizes:
            self.net = nn.Linear(input_dim, output_dim) # (s, s') transitions, already concatenated
        else:
            dim = hidden_sizes[0]
            layers = [nn.Linear(input_dim, dim)]
            
            later_sizes = hidden_sizes[1:] + [output_dim]
            for size in later_sizes:
                layers.append(self.activation)
                layers.append(nn.Linear(dim, size))
                dim = size
            
            self.net = nn.Sequential(*layers)
        
        self.apply(disc_weight_init)
        
    def forward(self, obs):
        return self.net(obs)
    
class GAILCost:
    def __init__(self,
                 expert_data: torch.Tensor,
                 feature_dim: int = 1,
                 hidden_dims: List[int] =[1024, 512],
                 input_type: str = 'ss',
                 scaling_coef: float = 0.5,
                 lambda_b: float = 0.5,
                 disc_loss_type='least_squares',
                 disc_opt: str = 'sgd',
                 disc_opt_args: dict = {'lr': 0.00001,
                                        'momentum': 0.9}):

        self.expert_data = expert_data
        self.n_expert_samples = expert_data.size(0)
        self.input_dim = expert_data.size(1) # 2 * state_dim
        self.feature_dim = feature_dim
        self.input_type = input_type
        self.scaling_coef = scaling_coef
        self.lambda_b = lambda_b
        self.disc_loss_type = disc_loss_type
        
        # discriminator setup
        self.disc = Discriminator(self.input_dim,
                                  hidden_dims,
                                  self.feature_dim,
                                  activation='relu')
        
        # discriminator optimization setup
        if disc_opt == 'sgd':
            self.disc_opt = optim.SGD(self.disc.parameters(),
                                      lr=disc_opt_args['lr'],
                                      momentum=disc_opt_args['momentum'])
        else:
            self.disc_opt = optim.Adam(self.disc.parameters(),
                                       lr=disc_opt_args['lr'])
            
    def sample_from_expert_data(self, batch_size: int):
        idxs = np.random.randint(0, self.n_expert_samples, batch_size)
        expert_ss = self.expert_data[idxs]
        return expert_ss
    
    def least_squares_gail_losses(self, expert_ss: torch.Tensor, mb_ss: torch.Tensor):
        '''Least squares GAIL loss from AMP paper (5.2).'''
        # outputs
        expert_outs = self.disc(expert_ss)
        mb_outs = self.disc(mb_ss)
        
        # construct loss
        expert_loss = ((expert_outs - 1.0) ** 2).mean()
        mb_loss = ((mb_outs + 1.0) ** 2).mean()
        total_loss = self.scaling_coef * expert_loss + (1.0 - self.scaling_coef) * mb_loss
        
        return expert_loss, mb_loss, total_loss
    
    def ll_gail_losses(self, expert_ss: torch.Tensor, mb_ss: torch.Tensor):
        '''Standard GAIL loss from AMP paper (5.1).'''
        # outputs
        expert_outs = self.disc(expert_ss)
        mb_outs = self.disc(mb_ss)
        
        # construct loss
        expert_loss = -torch.log(expert_outs).mean()
        mb_loss = -torch.log(1.0 - mb_outs).mean()
        total_loss = self.scaling_coef * expert_loss + (1.0 - self.scaling_coef) * mb_loss
        
        return expert_loss, mb_loss, total_loss
    
    def update_disc(self, mb_ss: torch.Tensor):
        # here we do +1, -1 GAIL loss used in AMP
        bs = mb_ss.size(0)
        expert_ss = self.sample_from_expert_data(bs)
        
        # get losses to log
        if self.disc_loss_type == 'least_squares':
            expert_loss, mb_loss, total_loss = self.least_squares_gail_losses(expert_ss, mb_ss)
        else:
            expert_loss, mb_loss, total_loss = self.ll_gail_losses(expert_ss, mb_ss)
        
        # backpropagate
        self.disc_opt.zero_grad()
        total_loss.backward()
        self.disc_opt.step()
        
        metrics = {
            'expert_disc_loss': expert_loss.detach().item(),
            'model_based_disc_loss': mb_loss.detach().item(),
            'total_disc_loss': total_loss.detach().item()
        }
        
        return metrics
    
    @torch.no_grad()
    def get_ls_costs(self, disc_outs: torch.Tensor):
        '''AMP least-squares cost (negative reward) associated with discriminator output.'''
        rewards = 1.0 - 0.25 * (1.0 - disc_outs) ** 2
        rewards = torch.maximum(rewards, 0.0)
        return -rewards
    
    @torch.no_grad()
    def get_ll_costs(self, disc_outs: torch.Tensor):
        '''AMP log-likelihood cost (negative reward) associated with discriminator output.'''
        # torch.log(disc_outs) should do the trick here
        costs = torch.log(disc_outs)
        costs = torch.maximum(costs, 0.0)
        return costs
    
    @torch.no_grad()
    def get_costs(self, disc_outs: torch.Tensor):
        if self.disc_loss_type == 'least_squares':
            return self.get_ls_costs(disc_outs)
        else:
            return self.get_ll_costs(disc_outs)
    
    def get_bonus_costs(self, states: torch.Tensor, actions: torch.Tensor, ensemble, next_states=None):
        '''cost with pessimism, similar to linear cost in this repo.'''
        if self.input_type == 'sa':
            input = torch.cat([states, actions], dim=1)
        elif self.input_type == 'ss':
            assert next_states is not None
            input = torch.cat([states, next_states], dim=1)
        elif self.input_type == 'sas':
            input = torch.cat([states, actions, next_states], dim=1)
        elif self.input_type == 's':
            input = states
        else:
            raise NotImplementedError("Input type not implemented")
        
        input_cost = self.get_costs(input)
        ipm = (1 - self.lambda_b) * input_cost
        
        # ensemble bonus cost
        discrepancy = ensemble.get_action_discrepancy(states, actions)
        bonus = self.lambda_b * discrepancy.view(-1, 1)
        
        cost = ipm - bonus

        # Logging info
        info = {'bonus': bonus, 'ipm': ipm, 'v_targ': input_cost, 'cost': cost}
        return cost, info
    

        