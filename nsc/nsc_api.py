from dsm.dsm_api import DSMBase
from nsc.nsc_torch import NeuralSurvivalClusterTorch
import nsc.losses as losses
from nsc.utilities import train_nsc

import torch
import numpy as np
from tqdm import tqdm

class NeuralSurvivalCluster(DSMBase):
  """
    Model API to call for using the method
    Preprocess data to shape it to the right format and handle CUDA
  """

  def __init__(self, cuda = torch.cuda.is_available(), **params):
    self.params = params
    self.fitted = False
    self.cuda = cuda

  def _gen_torch_model(self, inputdim, optimizer, risks):
    model = NeuralSurvivalClusterTorch(inputdim,
                                     **self.params,
                                     risks = risks,
                                     optimizer = optimizer).double()
    if self.cuda:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    """
      This method is used to train an instance of the NSC model.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.
      vsize: float
          Amount of data to set aside as the validation set.
      val_data: tuple
          A tuple of the validation dataset. If passed vsize is ignored.
      optimizer: str
          The choice of the gradient based optimization method. One of
          'Adam', 'RMSProp' or 'SGD'.
      random_state: float
          random seed that determines how the validation set is chosen.
    """
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data
    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(x_train.size(1), optimizer, risks = maxrisk)
    model = train_nsc(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda == 2,
                         **args)

    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e):
    """
      This method computes the negative log likelihood of the given data.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.

      Returns
        float: NLL
    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data

    if self.cuda == 2:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = losses.total_loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = 1):
    """
      This method computes the survival prediction of the given data at times t.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: float or list
          A list of times at which evaluate the model.

      Returns
        np.array (len(x), len(t)) Survival prediction for each points
    """
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.DoubleTensor([t_] * len(x)).to(x.device)
        log_alphas, log_betas, log_sr, _  = self.torch_model(x, t_)
        outcomes = 1 - ((log_alphas + log_betas).exp() * (1 - log_sr.exp())).sum(-1) 
        scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_alphas(self, x):
    """
      This method computes the weights on the population's distributions for the given input.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      risk: int
          Risk to consider

      Returns:
        np.array (length x, number components): Weights for each component
    """
    x = self._preprocess_test_data(x)
    if self.fitted:
      log_alphas, _, _, _ = self.torch_model(x, torch.zeros(len(x), dtype = torch.double).to(x.device))
      return log_alphas[:, 0, :].exp().detach().cpu().numpy()
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_alphas`.")

  def survival_cluster(self, t, risk = 1):
    """
      This method computes the population's survival distributions at times t.

      Parameters
      ----------
      t: int or list
          Times at which evaluate the population distributions
      risk: int
          Risk to consider

      Returns:
        np.array (number components, length of t): Survival distributions
    """
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      t = torch.DoubleTensor(t)
      x = torch.zeros(len(t), self.torch_model.input_dim, dtype = torch.double)

      # Push on the right device
      if self.cuda > 0:
        x, t = x.cuda(), t.cuda()

      _, log_betas, log_sr, _ = self.torch_model(x, t)
      return 1 - (log_betas.exp() * (1 - log_sr.exp()))[:, int(risk) - 1, :].detach().cpu().numpy()
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `survival_cluster`.")

  def feature_importance(self, x, t, e, risk = None, n = 100):
    """
      This method computes the features' importance by a  random permutation of the input variables.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.
      n: int
          Number of permutations used for the computation

      Returns:
        (dict, dict): Dictionary of the mean impact on likelihood and normal confidence interval

    """
    if not(risk is None):
      e = e == risk # Cause specific computation for risk
    global_nll = self.compute_nll(x, t, e)
    permutation = np.arange(len(x))
    performances = {j: [] for j in range(x.shape[1])}
    for _ in tqdm(range(n)):
      np.random.shuffle(permutation)
      for j in performances:
        x_permuted = x.copy()
        x_permuted[:, j] = x_permuted[:, j][permutation]
        performances[j].append(self.compute_nll(x_permuted, t, e))
    return {j: np.mean((np.array(performances[j]) - global_nll)/global_nll) for j in performances}, \
           {j: 1.96 * np.std((np.array(performances[j]) - global_nll)/global_nll) / np.sqrt(n) for j in performances}
          