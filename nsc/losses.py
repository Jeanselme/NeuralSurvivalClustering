import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e):
  # Go through network
  log_alphas, log_beta, log_sr, taus = model.forward(x, t)

  log_hr = model.gradient(log_sr, taus, e).log()
  log_sr = log_alphas + log_beta + log_sr

  error = - torch.logsumexp(log_sr[e == 0], dim = [1, 2]).sum() # Sum over the different mixture and risks, and then across patient
  for k in range(model.risks):
      error -= torch.logsumexp(log_sr[e == (k + 1)][:, k] + log_hr[e == (k + 1)], dim = 1).sum() # Sum over the different mixture 

  return error / len(x)