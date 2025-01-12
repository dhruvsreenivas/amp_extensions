import torch
import torch.nn as nn

import numpy as np

class RBFLinearCost:
    """
    MMD cost implementation with rff feature representations

    NOTE: Currently hardcoded to cpu

    :param expert_data: (torch Tensor) expert data used for feature matching
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda_b: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """
    def __init__(self,
                 expert_data,
                 feature_dim=1024,
                 input_type='ss',
                 cost_range=[-1.,0.],
                 bw_quantile=0.1,
                 bw_samples=100000,
                 lambda_b=1.0,
                 lr=0.0,
                 seed=100):
        # Set Random Seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.expert_data = expert_data
        input_dim = expert_data.size(1)
        self.input_type = input_type
        self.feature_dim = feature_dim
        self.cost_range = cost_range
        if cost_range is not None:
            self.c_min, self.c_max = cost_range
        self.lambda_b = lambda_b
        self.lr = lr

        # Fit Bandwidth
        self.quantile = bw_quantile
        self.bw_samples = bw_samples
        self.bw = self.fit_bandwidth(expert_data)

        # Define Phi and Cost weights
        self.rff = nn.Linear(input_dim, feature_dim)
        self.rff.bias.data = (torch.rand_like(self.rff.bias.data)-0.5)*2.0*np.pi
        self.rff.weight.data = torch.rand_like(self.rff.weight.data)/(self.bw+1e-8)

        # W Update Init
        self.w = None

        # Compute Expert Phi Mean
        self.expert_rep = self.get_rep(expert_data)
        self.phi_e = self.expert_rep.mean(dim=0)

    def get_rep(self, x):
        """
        Returns an RFF representation given an input
        """
        with torch.no_grad():
            out = self.rff(x.cpu())
            out = torch.cos(out)*np.sqrt(2/self.feature_dim)
        return out

    def fit_bandwidth(self, data):
        """
        Uses the median trick to fit the bandwidth for the RFF kernel
        """
        num_data = data.shape[0]
        idxs_0 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        norm = torch.norm(data[idxs_0, :]-data[idxs_1, :], dim=1)
        bw = torch.quantile(norm, q=self.quantile).item()
        return bw

    def fit_cost(self, data_pi):
        """
        Updates the weights of the cost with the closed form solution
        """
        phi = self.get_rep(data_pi).mean(0)
        feat_diff = phi - self.phi_e

        # Closed form solution
        self.w = feat_diff

        return torch.dot(self.w, feat_diff).item()

    def get_costs(self, x):
        """
        Returrns the IPM (MMD) cost for a given input
        """
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.mm(data, self.w.unsqueeze(1)), self.c_min, self.c_max)
        return torch.mm(data, self.w.unsqueeze(1))

    def get_expert_cost(self):
        """
        Returns the mean expert cost given our current discriminator weights and representations
        """
        return (1-self.lambda_b)*torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.c_min, self.c_max).mean()

    def get_bonus_costs(self, states, actions, ensemble, next_states=None):
        """
        Computes the cost with pessimism
        """
        if self.input_type == 'sa':
            rff_input = torch.cat([states, actions], dim=1)
        elif self.input_type == 'ss':
            assert(next_states is not None)
            rff_input = torch.cat([states, next_states], dim=1)
        elif self.input_type == 'sas':
            rff_input = torch.cat([states, actions, next_states], dim=1)
        elif self.input_type == 's':
            rff_input = states
        else:
            raise NotImplementedError("Input type not implemented")

        # Get Linear Cost
        rff_cost = self.get_costs(rff_input)

        if self.cost_range is not None:
            # Get Bonus from Ensemble
            discrepancy = ensemble.get_action_discrepancy(states, actions)/ensemble.threshold
            discrepancy = discrepancy.view(-1, 1)
            discrepancy[discrepancy>1.0] = 1.0
            # Bonus is LOW if (s,a) is unknown
            bonus = discrepancy * self.c_min
        else:
            bonus = ensemble.get_action_discrepancy(states, actions).view(-1,1)

        # Weight cost components
        ipm = (1-self.lambda_b)*rff_cost

        # Conservative/Pessimism Penalty term
        weighted_bonus = self.lambda_b*bonus.cpu() # Note cpu hardcoding

        # Cost
        cost = ipm - weighted_bonus

        # Logging info
        info = {'bonus': weighted_bonus, 'ipm': ipm, 'v_targ': rff_cost, 'cost': cost}

        return cost, info

class MLPCost:
    """
    MMD cost implementation with MLP feature representations
    NOTE: Currently hardcoded to cpu -- TODO make it cuda compatible

    :param expert_data: (torch Tensor) expert data used for feature matching (size N x data_dim)
    :param hidden_dims: (List[int]) hidden dims of MLP
    :param activation: (str) activation function for MLP
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda_b: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """

    def __init__(self,
                 expert_data,
                 hidden_dims=[2048, 2048],
                 activation='relu',
                 feature_dim=1024,
                 input_type='ss',
                 cost_range=[-1., 0.],
                 bw_quantile=0.1,
                 bw_samples=100000,
                 lambda_b=1.0,
                 lr=0.0,
                 seed=100):
        
        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.expert_data = expert_data
        input_dim = expert_data.size(1)
        self.input_type = input_type
        self.feature_dim = feature_dim
        self.cost_range = cost_range
        self.lambda_b = lambda_b
        self.lr = lr

        # make net
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        dim = feature_dim if not hidden_dims else hidden_dims[0]
        layers = [nn.Linear(input_dim, dim)]
        if hidden_dims[1:]:
            for size in hidden_dims[1:]:
                layers.append(self.activation)
                layers.append(nn.Linear(dim, size))
                dim = size
            layers.append(self.activation)
            layers.append(nn.Linear(dim, feature_dim))
            # TODO think about bounding outputs with tanh or sigmoid at the end
        layers.append(nn.Tanh()) # just in case
        
        self.net = nn.Sequential(*layers)

        # Fit Bandwidth
        self.quantile = bw_quantile
        self.bw_samples = bw_samples
        self.bw = self.fit_bandwidth(expert_data)

        # representations
        self.expert_rep = self.get_rep(expert_data)
        self.phi_e = self.expert_rep.mean(0)
        self.w = None

    def fit_bandwidth(self, expert_data):
        num_examples = expert_data.size(0)
        idxs_0 = torch.randint(low=0, high=num_examples, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_examples, size=(self.bw_samples,))
        norm = torch.norm(expert_data[idxs_0, :]-expert_data[idxs_1, :], dim=1)
        bw = torch.quantile(norm, q=self.quantile).item()
        return bw

    def get_rep(self, x):
        with torch.no_grad():
            out = self.net(x) # phi(x)
            # TODO: do we bound here by ± 2 / ndim by doing cos(...) * 2 / ndim?
            out = torch.cos(out) * np.sqrt(2 / self.feature_dim) # yes for now
        return out

    def fit_cost(self, policy_data):
        policy_rep = self.get_rep(policy_data).mean(0)
        feat_diff = policy_rep - self.phi_e

        self.w = feat_diff
        return torch.dot(self.w, feat_diff).item() # w^T [phi(pi) - phi(pi_e)]

    def get_costs(self, x):
        """
        Returns the IPM (MMD) cost for a given input
        """
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.mm(data, self.w.unsqueeze(1)), self.cost_range[0], self.cost_range[1])
        return torch.mm(data, self.w.unsqueeze(1))

    def get_expert_cost(self):
        """
        Returns the mean expert cost given our current discriminator weights and representations
        """
        return (1-self.lambda_b)*torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.cost_range[0], self.cost_range[1]).mean()

    def get_bonus_costs(self, states, actions, ensemble, next_states=None):
        """
        Computes the cost with pessimism
        """
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

        # cost
        input_cost = self.get_costs(input)

        if self.cost_range is not None:
            # Get ensemble bonus
            discrepancy = ensemble.get_action_discrepancy(states, actions)/ensemble.threshold
            discrepancy = discrepancy.view(-1, 1)
            discrepancy[discrepancy > 1.0] = 1.0
            # Bonus is LOW if (s, a) is unknown
            bonus = discrepancy * self.cost_range[0]
        else:
            bonus = ensemble.get_action_discrepancy(states, actions).view(-1, 1)

        # Weight cost components
        ipm = (1-self.lambda_b) * input_cost

        # Conservative/Pessimism Penalty term
        weighted_bonus = self.lambda_b * bonus.cpu() # Note cpu hardcoding

        # Cost
        cost = ipm - weighted_bonus

        # Logging info
        info = {'bonus': weighted_bonus, 'ipm': ipm, 'v_targ': input_cost, 'cost': cost}
        return cost, info