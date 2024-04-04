import numpy as np
from optimizers.dp_ight import fit as dp_ight_fit
from optimizers.dpsl_kt import fit as dpsl_kt_fit
from optimizers.ht_frank_wolfe import fit as ht_frank_wolfe_fit
from optimizers.ht_priv_lasso import fit as ht_priv_lasso_fit
from optimizers.ht_sparse_lasso import fit as ht_sparse_lasso_fit
from optimizers.ht_sparse_opt import fit as ht_sparse_opt_fit
from optimizers.ldp_ni import fit as ldp_ni_fit
from optimizers.frank_wolfe import fit as frank_wolfe_fit
from optimizers.noisy_mirror import fit as noisy_mirror_fit
from optimizers.poly_frank_wolfe import fit as poly_frank_wolfe_fit
from optimizers.proj_erm import fit as proj_erm_fit
from optimizers.two_stage import fit as two_stage_fit
from optimizers.vr_frank_wolfe import fit as vr_frank_wolfe_fit
from optimizers.sgd import fit as sgd_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.extmath import safe_sparse_dot
from optimizers.utils import mse_grad, mse_support, mse_obj_pert

class LinearRegression():
    def __init__(self, optimizer):
        self.w = None
        self.optimizer = {
            'dpight': dp_ight_fit, 
            'dpslkt': dpsl_kt_fit, 
            'fw': frank_wolfe_fit,
            'htfw': ht_frank_wolfe_fit,
            'htpl': ht_priv_lasso_fit,
            'htsl': ht_sparse_lasso_fit,
            'htso': ht_sparse_opt_fit,
            'ldp-ni': ldp_ni_fit, 
            'ts': two_stage_fit, 
            'nm': noisy_mirror_fit, 
            'projerm': proj_erm_fit, 
            'polyfw': poly_frank_wolfe_fit, 
            'vrfw': vr_frank_wolfe_fit, 
            'sgd': sgd_fit
        }[optimizer]

    def fit(self, X, y, **kwargs):
        if self.optimizer in {dp_ight_fit, dpsl_kt_fit, frank_wolfe_fit, 
                              noisy_mirror_fit, ht_sparse_lasso_fit, 
                              ht_sparse_opt_fit, ht_priv_lasso_fit, 
                              ht_frank_wolfe_fit, poly_frank_wolfe_fit, 
                              vr_frank_wolfe_fit}:
            self.w = self.optimizer(X, y, grad=mse_grad, **kwargs)
        if self.optimizer is ldp_ni_fit:
            self.w = self.optimizer(X, y, **kwargs)
        if self.optimizer in {proj_erm_fit, sgd_fit}:
            self.w = self.optimizer(X, y, type='linear', **kwargs)
        if self.optimizer is two_stage_fit:
            self.w = self.optimizer(X, y, supp=mse_support, perturbed_optim=mse_obj_pert, **kwargs)
        self.w = np.where(np.abs(self.w) <= 1e-11, 0, self.w)

    def get_params(self):
        return self.w
    
    def predict(self, X):
        assert isinstance(self.w, np.ndarray)
        return safe_sparse_dot(X, self.w, dense_output=True)
    
    def loss(self, X, y):
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)
        
    def score(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)