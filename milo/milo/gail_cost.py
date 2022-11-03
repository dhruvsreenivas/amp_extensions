from tokenize import ContStr
from milo.datasets import AgentReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from typing import List, Optional

def disc_weight_init(m):
    if isinstance(m, nn.Linear):
        if m.out_features == 1:
            nn.init.uniform_(m.weight.data, a=-1.0, b=1.0)
        else:
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
                 agent_rb: Optional[AgentReplayBuffer] = None,
                 feature_dim: int = 1,
                 hidden_dims: List[int] = [1024, 512],
                 input_type: str = 'ss',
                 scaling_coef: float = 0.5,
                 reg_coef: float = 0.05,
                 lambda_b: float = 0.5,
                 seed=100,
                 grad_lambda=10.0,
                 disc_loss_type='least_squares',
                 disc_opt: str = 'sgd',
                 disc_opt_args: dict = {'lr': 0.00001,
                                        'momentum': 0.9}):

        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.expert_data = expert_data
        if agent_rb is not None:
            self.online_rb = agent_rb
        
        self.n_expert_samples = expert_data.size(0)
        self.input_dim = expert_data.size(1) # 2 * state_dim
        self.feature_dim = feature_dim
        self.input_type = input_type
        self.scaling_coef = scaling_coef
        self.reg_coef = reg_coef
        self.lambda_b = lambda_b
        self.grad_lambda = grad_lambda
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
    
    def sample_from_online_buffer(self, batch_size: int):
        assert self.online_rb is not None, 'Cannot sample from nonexistent buffer.'
        agent_ss = self.online_rb.sample(batch_size)
        return agent_ss
    
    def least_squares_gail_losses(self, expert_ss: torch.Tensor, mb_ss: torch.Tensor):
        '''Least squares GAIL loss from AMP paper (5.2).'''
        # outputs
        expert_outs = self.disc(expert_ss)
        mb_outs = self.disc(mb_ss)
        
        # construct loss
        expert_loss = ((expert_outs - 1.0) ** 2).sum(dim=-1).mean() # can do just pure mean here too
        mb_loss = ((mb_outs + 1.0) ** 2).sum(dim=-1).mean() # can do just pure mean here too
        total_loss = self.scaling_coef * expert_loss + (1.0 - self.scaling_coef) * mb_loss
        
        # accuracy measurement (no update required here)
        with torch.no_grad():
            expert_acc = torch.greater(expert_outs, 0.).to(dtype=torch.float32).mean()
            mb_acc = torch.less(mb_outs, 0.).to(dtype=torch.float32).mean()
            accuracies = {
                'expert_acc': expert_acc,
                'mb_acc': mb_acc
            }
        
        return expert_loss, mb_loss, total_loss, accuracies
    
    def log_likelihood_gail_losses(self, expert_ss: torch.Tensor, mb_ss: torch.Tensor):
        '''Log-likelihood based GAIL loss from AMP paper (5.1).'''
        # outputs
        expert_outs = self.disc(expert_ss)
        expert_outs = torch.sigmoid(expert_outs)
        
        mb_outs = self.disc(mb_ss)
        mb_outs = torch.sigmoid(mb_outs)
        
        # construct loss
        expert_loss = -torch.log(expert_outs).mean()
        mb_loss = -torch.log(1.0 - mb_outs).mean()
        total_loss = self.scaling_coef * expert_loss + (1.0 - self.scaling_coef) * mb_loss
        
        return expert_loss, mb_loss, total_loss
    
    def regularizer_loss(self):
        '''Regularizer loss from https://github.com/xbpeng/DeepMimic/blob/70e7c6b22b775bb9342d4e15e6ef0bd91a55c6c0/learning/amp_agent.py#L196 '''
        no_bias_params = [p.view(-1) for n, p in self.disc.named_parameters() if 'bias' not in n]
        losses = [0.5 * (p ** 2).sum() for p in no_bias_params]
        return torch.as_tensor(losses).sum()
    
    def expert_grad_pen(self, expert_ss: torch.Tensor):
        '''AMP expert gradient penalty.
        Implementation aided by Ilya Kostrikov's repo:
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/algo/gail.py#L29
        '''
        # TODO mixing may help!
        expert_ss.requires_grad = True
        expert_outs = self.disc(expert_ss)
        ones = torch.ones(expert_outs.size())
        expert_grads = autograd.grad(
            outputs=expert_outs,
            inputs=expert_ss,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        grad_pen = 0.5 * torch.norm(expert_grads, p=2, dim=-1).mean()
        return grad_pen
    
    def update_disc(self, mb_ss: Optional[torch.Tensor]):
        # here we do +1, -1 GAIL loss used in AMP
        if mb_ss is None:
            assert self.online_rb is not None, 'need to sample from online buffer to get agent data!'
            mb_ss = self.sample_from_online_buffer(256)
        
        bs = mb_ss.size(0)
        expert_ss = self.sample_from_expert_data(bs)
        
        # get losses to log
        if self.disc_loss_type == 'least_squares':
            expert_loss, mb_loss, total_loss, ls_accuracies = self.least_squares_gail_losses(expert_ss, mb_ss)
        else:
            expert_loss, mb_loss, total_loss = self.log_likelihood_gail_losses(expert_ss, mb_ss)
        
        reg_loss = self.regularizer_loss() * self.reg_coef
        grad_pen = self.expert_grad_pen(expert_ss) * self.grad_lambda
        aux_loss = reg_loss + grad_pen
        total_loss += aux_loss
        
        # backpropagate
        self.disc_opt.zero_grad()
        total_loss.backward()
        self.disc_opt.step()
        
        metrics = {
            'expert_disc_loss': expert_loss.detach().item(),
            'model_based_disc_loss': mb_loss.detach().item(),
            'gradient_penalty': grad_pen.detach().item(),
            'regularizer_loss': reg_loss.detach().item(),
            'total_disc_loss': total_loss.detach().item()
        }
        if self.disc_loss_type == 'least_squares':
            metrics.update(ls_accuracies)
        
        return metrics
    
    @torch.no_grad()
    def compute_chi2_distance(self):
        '''Computes chi^2 distance over entire expert data and everything from the model.'''
        assert self.online_rb is not None, 'Cannot compute divergence if there is no agent data.'
        agent_data = torch.cat([self.online_rb.states, self.online_rb.next_states], dim=-1)
        if agent_data.size(0) < self.n_expert_samples:
            expert_idxs = np.random.randint(0, self.n_expert_samples, size=agent_data.size(0)) # in case there is amount mismatch
            expert_data = self.expert_data[expert_idxs]
        else:
            expert_data = self.expert_data
            
        if agent_data.size(0) >= self.n_expert_samples:
            agent_idxs = np.random.randint(0, agent_data.size(0), size=self.n_expert_samples)
            agent_data = agent_data[agent_idxs]
        
        # now we can compute the divergence
        agent_outs = self.disc(agent_data)
        expert_outs = self.disc(expert_data)
        expert_loss = ((expert_outs - 1.0) ** 2).sum(dim=-1).mean() # can do just pure mean here too
        mb_loss = ((agent_outs + 1.0) ** 2).sum(dim=-1).mean() # can do just pure mean here too
        
        total_loss = self.scaling_coef * expert_loss + (1.0 - self.scaling_coef) * mb_loss
        return total_loss.cpu().item()
    
    @torch.no_grad()
    def get_ls_costs(self, ss: torch.Tensor):
        '''AMP least-squares cost (negative reward) associated with discriminator output.'''
        disc_outs = self.disc(ss)
        rewards = 1.0 - 0.25 * (1.0 - disc_outs) ** 2
        rewards[rewards < 0.0] = 0.0
        return -rewards
    
    @torch.no_grad()
    def get_ll_costs(self, ss: torch.Tensor):
        '''AMP log-likelihood cost (negative reward) associated with discriminator output.'''
        # torch.log(disc_outs) should do the trick here
        disc_outs = self.disc(ss)
        costs = F.logsigmoid(disc_outs)
        return costs

    @torch.no_grad()
    def get_costs(self, ss: torch.Tensor):
        if self.disc_loss_type == 'least_squares':
            return self.get_ls_costs(ss)
        else:
            return self.get_ll_costs(ss)
    
    @torch.no_grad()
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