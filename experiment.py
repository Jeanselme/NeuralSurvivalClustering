"""
    This file contains the experimental framework with cross validation
    For new methods, add a child method to Experiment
"""
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import copy
import os
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

def from_surv_to_t(pred, times):
    """
        Interpolate pred for predictions at time

        Pred: Horizon * Patients
    """
    from scipy.interpolate import interp1d
    res = []
    for i in pred.columns:
        res.append(interp1d(pred.index, pred[i].values, fill_value = (1, pred[i].values[-1]), bounds_error = False)(times))
    return np.vstack(res)

class ToyExperiment():

    def train(self, *args):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, hyper_grid = None, n_iter = 100, fold = None, k = 5, 
                random_seed = 0, times = 100, path = 'results', save = True):
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.times = times
        self.k = k
        
        # Allows to reload a previous model
        self.all_fold = fold
        self.iter, self.fold = 0, 0
        self.best_hyper = {}
        self.best_model = {}
        self.best_nll = None

        self.path = path
        self.tosave = save

    @classmethod
    def create(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
                random_seed = 0, times = 100, path = 'results', force = False, save = True):
        if not(force):
            path = path if fold is None else path + '_{}'.format(fold)
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    return cls.load(path+ '.pickle')
                except Exception as e:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(hyper_grid, n_iter, fold, k, random_seed, times, path, save)

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        try:
            se = pickle.load(file)
            return se
        except Exception as e:
            # Load on CPU
            se = CPU_Unpickler(file).load()
            for model in se.best_model:
                if type(se.best_model[model]) is dict:
                    for m in se.best_model[model]:
                        se.best_model[model][m].cuda = False
                else:
                    se.best_model[model].cuda = False
            return se
        
    @classmethod
    def merge(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
            random_seed = 0, times = 100, path = 'results', save = True):
        if os.path.isfile(path + '.csv'):
            print(path)
            return cls.load(path + '.pickle')
        else:
            merged = cls(hyper_grid, n_iter, fold, k, random_seed, times, path, save)
            for i in range(k):
                path_i = path + '_{}.pickle'.format(i)
                if os.path.isfile(path_i):
                    merged.best_model[i] = cls.load(path_i).best_model[i]
                else:
                    print('Fold {} has not been computed yet'.format(i))
            merged.fold = k # Nothing to run
            return merged

    @staticmethod
    def save(obj):
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')

    def save_results(self, x):
        clusters, predictions = [], []
        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            predictions.append(pd.concat([self._predict_(model, x[index], r, index) for r in self.risks], axis = 1))
            clusters.append(pd.DataFrame(self._predict_cluster_(self.best_model[i], x[index]), index = index))
        predictions = pd.concat(predictions, axis = 0).loc[self.fold_assignment.dropna().index]
        clusters = pd.concat(clusters, axis = 0).loc[self.fold_assignment.dropna().index]
        clusters.columns = pd.MultiIndex.from_product([['Assignment'], clusters.columns])

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis = 1).to_csv(self.path + '.csv')

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['Fold']])
            pd.concat([predictions, fold_assignment, clusters], axis = 1).to_csv(self.path + '.csv')

        return predictions, clusters

    def train(self, x, t, e):
        """
            Cross validation model

            Args:
                x (Dataframe n * d): Observed covariates
                t (Dataframe n): Time of censoring or event
                e (Dataframe n): Event indicator

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)

        self.risks = np.unique(e[e > 0])
        self.fold_assignment = pd.Series(np.nan, index = range(len(x)))
        if self.k == 1:
            kf = ShuffleSplit(n_splits = self.k, random_state = self.random_seed, test_size = 0.2)
        else:
            kf = StratifiedKFold(n_splits = self.k, random_state = self.random_seed, shuffle = True)

        # First initialization
        if self.best_nll is None:
            self.best_nll = np.inf
        for i, (train_index, test_index) in enumerate(kf.split(x, e)): # To ensure to split on both censoring and treatment
            self.fold_assignment[test_index] = i
            if i < self.fold: continue # When reload: start last point
            if not(self.all_fold is None) and (self.all_fold != i): continue
            print('Fold {}'.format(i))
            
            train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = self.random_seed, stratify = e[train_index])
            dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = self.random_seed, stratify = e[dev_index])
            
            x_train, x_dev, x_val = x[train_index], x[dev_index], x[val_index]
            t_train, t_dev, t_val = t[train_index], t[dev_index], t[val_index]
            e_train, e_dev, e_val = e[train_index], e[dev_index], e[val_index]

            # Train on subset one domain
            ## Grid search best params
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter: continue # When reload: start last point
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                model = self._fit_(x_train, t_train, e_train, x_val, t_val, e_val, hyper.copy())
                nll = self._nll_(model, x_dev, t_dev, e_dev, hyper.copy())
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll
                self.iter = j + 1
                self.save(self)
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            self.save(self)
            return self.save_results(x)

    def _fit_(self, *params):
        raise NotImplementedError()
    
    def _nll_(self, model, x, t, e, *train):
        return model.compute_nll(x, t, e)

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), risk = r), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

    def _predict_cluster_(self, model, x):
        return np.zeros(len(x))

    def likelihood(self, x, t, e):
        """
            Compute the nll over the different folds
            Data must match original index
        """
        x = self.scaler.transform(x)
        nll_fold = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            nll_fold[i] = self._nll_(model, x[index], t[index], e[index], self.best_hyper[i])

        return nll_fold

    def importance(self, x, t, e):
        return None
    
    def survival_cluster(self, x):
        """
            Compute the multiple clusters by hard assingment of all patients
            Return 0 if no clusters
        """
        x = self.scaler.transform(x)
        clusters = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            
            # Assign all patients
            assignment = self._predict_cluster_(model, x[index]).argmax(1)
            survival = self._predict_(model, x[index], 1, x[index].index)

            # Compute for all treatment cluster and cox cluster the average survival
            clusters[i] = []
            for cluster in range(model.k):
                selection = assignment == cluster
                # Ensure the cluster is not empty
                if selection.sum() > 0:
                    clusters[i].append(survival[selection].mean(0).reshape((-1, 1)))
                else:
                    clusters[i].append(np.zeros(len(self.times), 1))

            clusters[i] = np.concatenate(clusters[i], 1)

        return clusters
    
# Survival models (no clustering)
class DeepHitExperiment(Experiment):
    """
        This class require a slightly more involved saving scheme to avoid a lambda error with pickle
        The models are removed at each save and reloaded before saving results 
    """

    @classmethod
    def load(cls, path):
        from pycox.models import DeepHitSingle, DeepHit
        file = open(path, 'rb')
        try:
            exp = pickle.load(file)
            for i in exp.best_model:
                if isinstance(exp.best_model[i], tuple):
                    net, cuts = exp.best_model[i]
                    exp.best_model[i] = DeepHit(net, duration_index = cuts) if len(exp.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
            return exp
        except:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if isinstance(se.best_model[i], tuple):
                    net, cuts = se.best_model[i]
                    se.best_model[i] = DeepHit(net, duration_index = cuts) if len(se.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
                    se.best_model[i].cuda = False
            return se


    @classmethod
    def save(cls, obj):
        from pycox.models import DeepHitSingle, DeepHit
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                obj_save = copy.copy(obj)
                obj_save.best_model = {}
                for i in obj.best_model:
                    # Split model and save components (error pickle otherwise)
                    if isinstance(obj.best_model[i], DeepHit) or isinstance(obj.best_model[i], DeepHitSingle):
                        obj_save.best_model[i] = (obj.best_model[i].net, obj.best_model[i].duration_index)
                pickle.dump(obj_save, output)
            except Exception as e:
                print('Unable to save object')

    def save_results(self, x):
        from pycox.models import DeepHitSingle, DeepHit

        # Reload models in memory
        for i in self.best_model:
            if isinstance(self.best_model[i], tuple):
                # Reload model
                net, cuts = self.best_model[i]
                self.best_model[i] = DeepHit(net, duration_index = cuts) if len(self.risks) > 1 \
                                else DeepHitSingle(net, duration_index = cuts)
        return super().save_results(x)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter): 
        from pycox.models import DeepHitSingle, DeepHit
        import torchtuples as tt

        n = hyperparameter.pop('n', 15)
        nodes = hyperparameter.pop('nodes', [100])
        shared = hyperparameter.pop('shared', [100])
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        self.eval_times = np.linspace(0, t.max(), n)
        callbacks = [tt.callbacks.EarlyStopping()]
        num_risks = len(np.unique(e))- 1
        if  num_risks > 1:
            assert len(np.unique(e[t != 0])) > 1, 'DeepHit does not YET handle competing risks'
            from deephit.utils import CauseSpecificNet, tt, LabTransform
            self.labtrans = LabTransform(self.eval_times.tolist())
            net = CauseSpecificNet(x.shape[1], shared, nodes, num_risks, self.labtrans.out_features, False)
            model = DeepHit(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        else:
            self.labtrans = DeepHitSingle.label_transform(self.eval_times.tolist())
            net = tt.practical.MLPVanilla(x.shape[1], shared + nodes, self.labtrans.out_features, False)
            model = DeepHitSingle(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        model.optimizer.set_lr(lr)
        model.fit(x.astype('float32'), self.labtrans.transform(t, e), batch_size = batch, epochs = epochs, 
                    callbacks = callbacks, val_data = (x_val.astype('float32'), self.labtrans.transform(t_val, e_val)))
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.score_in_batches(x.astype('float32'), self.labtrans.transform(t, e))['loss']

    def _predict_(self, model, x, r, index):
        if len(self.risks) == 1:
            survival = model.predict_surv_df(x.astype('float32')).values
        else:
            survival = 1 - model.predict_cif(x.astype('float32'))[r - 1]

        # Interpolate at the point of evaluation
        survival = pd.DataFrame(survival, columns = index, index = model.duration_index)
        predictions = pd.DataFrame(np.nan, columns = index, index = self.times)
        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]
        return survival.loc[self.times].set_index(pd.MultiIndex.from_product([[r], self.times])).T

class DeepSurvExperiment(Experiment):

    @classmethod
    def load(cls, path):
        from pycox.models import CoxPH
        file = open(path, 'rb')
        try:
            exp = pickle.load(file)
            for i in exp.best_model:
                if isinstance(exp.best_model[i], tuple):
                    net, hazard, cum_hazard = exp.best_model[i]
                    exp.best_model[i] = CoxPH(net)
                    exp.best_model[i].baseline_hazards_ = hazard
                    exp.best_model[i].baseline_cumulative_hazards_ = cum_hazard
            return exp
        except:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if isinstance(se.best_model[i], tuple):
                    net, hazard, cum_hazard = se.best_model[i]
                    se.best_model[i] = CoxPH(net)
                    se.best_model[i].baseline_hazards_ = hazard
                    se.best_model[i].baseline_cumulative_hazards_ = cum_hazard
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def save(cls, obj):
        from pycox.models import CoxPH
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                obj_save = copy.copy(obj)
                obj_save.best_model = {}
                for i in obj.best_model:
                    # Split model and save components (error pickle otherwise)
                    if isinstance(obj.best_model[i], CoxPH):
                        obj_save.best_model[i] = (obj.best_model[i].net, obj.best_model[i].baseline_hazards_, obj.best_model[i].baseline_cumulative_hazards_)
                pickle.dump(obj_save, output)
            except Exception as e:
                print('Unable to save object')

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from pycox.models import CoxPH
        import torchtuples as tt

        assert len(np.unique(e[t != 0])) > 1, 'DeepSurv does not handle competing risks'

        nodes = hyperparameter.pop('nodes', 100)
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        callbacks = [tt.callbacks.EarlyStopping()]
        net = tt.practical.MLPVanilla(x.shape[1], nodes, 1, False).double()
        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(lr)
        model.fit(x, (t, e), batch_size = batch, epochs = epochs, callbacks = callbacks, val_data = (x_val, (t_val, e_val)))
        _ = model.compute_baseline_hazards()

        return model

    def _nll_(self, model, x, t, e, *train):
        return - model.partial_log_likelihood(x, (t, e)).mean()

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(from_surv_to_t(model.predict_surv_df(x), self.times), index = index, columns = pd.MultiIndex.from_product([[r], self.times]))

class SuMoExperiment(Experiment):
    def __process__(self, t, save = False):
        if save:
            self.max_t = t.max()
        return t / self.max_t

    def train(self, x, t, e):
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        t_norm = self.__process__(t, True)
        return super().train(x, t_norm, e)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from sumo import SuMo

        assert len(np.unique(e[t != 0])) > 1, 'SuMo does not handle competing risks'

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = SuMo(**hyperparameter, cuda = torch.cuda.is_available())
        model.fit(x, t, e, n_iter = epochs, bs = batch,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.__process__(self.times).tolist()), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

    def likelihood(self, x, t, e):
        t_norm = self.__process__(t)
        return super().likelihood(x, t_norm, e)

class DSMExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from dsm import DeepSurvivalMachines

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = DeepSurvivalMachines(**hyperparameter, cuda = torch.cuda.is_available())
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model
    
    def _predict_cluster_(self, model, x):
        return model.predict_alphas(x)
    
# Survival models with clustering components
class CoxExperiment(Experiment):

    def __process__(self, x, t, e):
        res = pd.DataFrame(x)
        res['event'] = e
        res['duration'] = t
        return res

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from lifelines import CoxPHFitter

        assert len(np.unique(e[t != 0])) > 1, 'Cox does not handle competing risks'

        data = self.__process__(np.concatenate([x, x_val]), 
                                np.concatenate([t, t_val]), 
                                np.concatenate([e, e_val]))
        
        k = hyperparameter.pop('k', 3)

        model = CoxPHFitter(**hyperparameter)
        model.k = k # Save a copy of the number of clusters
        model.fit(data, duration_col = 'duration', event_col = 'event')
        return model

    def _nll_(self, model, x, t, e, *train):
        return - model.score(self.__process__(x, t, e))

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival_function(pd.DataFrame(x), self.times).T.values, index = index, columns = pd.MultiIndex.from_product([[r], self.times]))

    def _predict_cluster_(self, model, x):
        """
            Fit a weighted K-Means given the Cox model weights
        """
        # Normalise the data
        x = self.scaler.transform(x)

        # Rescale to adjust weights
        weights = model.params_
        weighted_x = x * weights.values.reshape(1, -1)

        # Use a K-Means given the Cox model weights
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = model.k, random_state = self.random_seed)

        # Change format from cluster assignment to matrix
        from sklearn.preprocessing import LabelBinarizer
        return LabelBinarizer().fit_transform(kmeans.fit_predict(weighted_x))     

class NSCExperiment(Experiment):

    def __process__(self, t, save = False):
        if save:
            self.max_t = t.max()
        return t / self.max_t

    def train(self, x, t, e):
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        t_norm = self.__process__(t, True)
        return super().train(x, t_norm, e)
    
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from nsc import NeuralSurvivalCluster

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = NeuralSurvivalCluster(**hyperparameter)
        model.fit(x, t, e, n_iter = epochs, bs = batch,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.__process__(self.times).tolist(), r if model.torch_model.risks >= r else 1), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

    def _predict_cluster_(self, model, x):
        return model.predict_alphas(x)

    def survival_cluster(self, x):
        clusters = {}

        for i in self.best_model:
            model = self.best_model[i]
            clusters[i] = model.survival_cluster(self.__process__(self.times).tolist())

        return clusters
    
    def importance(self, x, t, e, **params):
        importance = {}

        for i in self.best_model:
            model = self.best_model[i]
            importance[i] = model.feature_importance(x, self.__process__(t), e, **params)

        return importance

    def likelihood(self, x, t, e):
        t_norm = self.__process__(t)
        return super().likelihood(x, t_norm, e)
    
class DCMExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from dsm.contrib import DeepCoxMixtures

        assert len(np.unique(e[t != 0])) > 1, 'DCM does not handle competing risks'

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = DeepCoxMixtures(**hyperparameter)
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _predict_cluster_(self, model, x):
        """
            Compute assignment of all points
        """
        return model.predict_alphas(x)
    
    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist()), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)


class SurvivalTreeExperiment(Experiment):
    
    def __process__(self, t, e):
        return np.array([(e[i], t[i]) for i in range(len(e))], dtype = [('e', bool), ('t', float)])
    
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        from sksurv.tree import SurvivalTree

        # No validation set
        x, t, e = np.concatenate([x, x_val]), np.concatenate([t, t_val]), np.concatenate([e, e_val])

        model = SurvivalTree(**hyperparameter, random_state = self.random_seed)
        model.fit(x, self.__process__(t, e))
        
        return model

    def _nll_(self, model, x, t, e, *train):
        return - model.score(x, self.__process__(t, e))

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(from_surv_to_t(pd.DataFrame(model.predict_survival_function(x, return_array = True), columns = model.unique_times_).T, self.times), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

    def _predict_cluster_(self, model, x):
        from sklearn.preprocessing import LabelBinarizer
        return LabelBinarizer().fit_transform(model.apply(x.astype(np.float32)))

# TODO: Add your method here
class NewExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter):  
        # TODO: Import your library
        # TODO: Ensure default hyperparameter if not specified
        # TODO: Return trained model

        # NB: Always ensure that a new model is initated
        pass

    def _nll_(self, model, x, t, e, *train):
        # TODO: Compute the nll of your method given input
        pass

    def _predict_(self, model, x, times, r):
        # TODO: Predict model outcome on the data
        pass

    def _predict_cluster_(self, model, x):
        # TODO: If your model discovers clusters, return the cluster assignment
        pass
