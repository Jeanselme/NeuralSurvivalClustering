from torch.autograd import grad
import numpy as np
import torch.nn as nn
import torch

class PositiveLinear(nn.Module):
  """
    Neural network with positive weights
  """

  def __init__(self, in_features, out_features, bias = False):
    super(PositiveLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.log_weight)
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
      bound = np.sqrt(1 / np.sqrt(fan_in))
      nn.init.uniform_(self.bias, -bound, bound)
    self.log_weight.data.abs_().sqrt_()

  def forward(self, input):
    if self.bias is not None:
      return nn.functional.linear(input, self.log_weight ** 2, self.bias)
    else:
      return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, activation, dropout = 0):
  """
   Create a simple multi layer neural network of positive layers
   With final SoftPlus
  """
  modules = []
  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'Tanh':
    act = nn.Tanh()
  else:
    raise ValueError("Unknown {} activation".format(activation))
  
  prevdim = inputdim
  for hidden in layers:
    modules.append(PositiveLinear(prevdim, hidden, bias=True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden

  # Need all values positive 
  modules[-1] = nn.Softplus()

  return nn.Sequential(*modules)

def create_representation(inputdim, layers, activation, dropout = 0., last = None):
  if activation == 'ReLU6':
      act = nn.ReLU6()
  elif activation == 'ReLU':
      act = nn.ReLU()
  elif activation == 'Tanh':
      act = nn.Tanh()

  modules = []
  prevdim = inputdim

  for hidden in layers:
      modules.append(nn.Linear(prevdim, hidden, bias=True))
      if dropout > 0:
          modules.append(nn.Dropout(p = dropout))
      modules.append(act)
      prevdim = hidden

  if last is not None:
      modules[-1] = last

  return nn.Sequential(*modules)

class NeuralSurvivalClusterTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU6', layers_surv = [100], representation = 50, act_surv = 'Tanh',
               risks = 1, k = 3, dropout = 0., optimizer = "Adam"):
    super(NeuralSurvivalClusterTorch, self).__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.k = k          # Number mixture
    self.representation = representation # Latent input for clusters (centroid representation)
    self.dropout = dropout
    self.optimizer = optimizer

    self.profile = create_representation(inputdim, layers + [self.k], act, self.dropout, last = nn.LogSoftmax(dim = 1)) # Proba for cluster P(cluster | x)
    self.latent = nn.ParameterList([nn.Parameter(torch.randn((1, self.representation))) for _ in range(self.k)])
    self.outcome = nn.ModuleList([create_representation_positive(1 + self.representation, layers_surv + [risks], act_surv, self.dropout) for _ in range(self.k)]) # Model P(survival | risk, cluster)
    self.competing = nn.ParameterList([nn.Parameter(torch.randn(risks)) for _ in range(self.k)]) # Proba for the given risk P(risk | cluster) - Fixed for a cluster
    self.softlog = nn.LogSoftmax(dim = 0)

  def forward(self, x, horizon):
    # Compute proba cluster len(x) * 1 * k
    log_alphas = torch.zeros((len(x), 1, 1), requires_grad = True).float().to(x.device) if self.k == 1 else \
                 self.profile(x).unsqueeze(1)

    # For each cluster
    log_sr, log_beta = [], []
    tau_outcome = [horizon.clone().detach().requires_grad_(True).unsqueeze(1) for _ in range(self.k)] # Requires independent clusters
    self.latent._size, self.competing._size = self.k, self.k
    for outcome, latent, balance, tau in zip(self.outcome, self.latent, self.competing, tau_outcome):
      # Compute survival distribution for each distribution 
      latent = latent.repeat(len(x), 1) # Shape: len(x) * representation
      logOutcome = tau * outcome(torch.cat((latent, tau), 1)) # Outcome at time t for all risks
      log_sr.append(- torch.cat(torch.split(logOutcome, len(x), 0), 1).unsqueeze(-1)) # len(x), risks
      log_beta.append(self.softlog(balance).unsqueeze(0).repeat(len(x), 1).unsqueeze(-1)) # Balance between risks in this cluster (fixed for the cluster)

    log_sr = torch.cat(log_sr, -1)  # Dim: Point * Risks * Cluster
    log_beta = torch.cat(log_beta, -1) # Dim: Point * Risks * Cluster
    return log_alphas, log_beta, log_sr, tau_outcome
  
  def gradient(self, outcomes, horizon, e):
    # Avoid computation of all gradients by focusing on the one used later
    return torch.cat(grad([- outcomes[:, risk][e == (risk + 1)].sum() for risk in range(self.risks)], horizon, create_graph = True), 1).clamp_(1e-10)