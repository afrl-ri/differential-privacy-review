import numpy as np
from optimizers.admm import fit as admm_fit
from optimizers.admm_half import fit as admm_half_fit
from optimizers.dp_ight import fit as dp_ight_fit
from optimizers.dpsl_kt import fit as dpsl_kt_fit
from optimizers.ldp_ni import fit as ldp_ni_fit
from optimizers.frank_wolfe import fit as frank_wolfe_fit
from optimizers.two_stage import fit as two_stage_fit
from optimizers.ht_frank_wolfe import fit as ht_frank_wolfe_fit
from optimizers.ht_sparse_opt import fit as ht_sparse_opt_fit
from optimizers.noisy_mirror import fit as nm_fit
from optimizers.proj_erm import fit as proj_erm_fit
from optimizers.poly_frank_wolfe import fit as poly_fw_fit
from optimizers.vr_frank_wolfe import fit as vr_fw_fit
from optimizers.sgd import fit as sgd_fit
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.utils.extmath import safe_sparse_dot
from optimizers.utils import bce_grad, bce_support, bce_obj_pert

class LogisticRegression():
    def __init__(self, optimizer):
        self.w = None
        self.optimizer = {
            'admm': admm_fit,
            'admmhalf': admm_half_fit,
            'dpight': dp_ight_fit, 
            'dpslkt': dpsl_kt_fit, 
            'fw': frank_wolfe_fit,
            'ts': two_stage_fit, 
            'htfw': ht_frank_wolfe_fit, 
            'htso': ht_sparse_opt_fit, 
            'nm': nm_fit, 
            'projerm': proj_erm_fit, 
            'polyfw': poly_fw_fit, 
            'vrfw': vr_fw_fit, 
            'sgd': sgd_fit
        }[optimizer]

    def fit(self, X, y, **kwargs): 
        if self.optimizer in {admm_fit, admm_half_fit}:
            self.w = self.optimizer(X, y, **kwargs)
        if self.optimizer in {dp_ight_fit, dpsl_kt_fit, frank_wolfe_fit, 
                              ht_frank_wolfe_fit, ht_sparse_opt_fit, 
                              nm_fit, poly_fw_fit, vr_fw_fit}:
            self.w = self.optimizer(X, y, grad=bce_grad, **kwargs)
        if self.optimizer is two_stage_fit:
            self.w = self.optimizer(X, y, supp=bce_support, perturbed_optim=bce_obj_pert, **kwargs)
        if self.optimizer in {proj_erm_fit, sgd_fit}:
            self.w = self.optimizer(X, y, type='logistic', **kwargs)
        self.w = np.where(np.abs(self.w) <= 1e-11, 0, self.w)

    def get_params(self):
        return self.w

    def predict_proba(self, X):
        assert isinstance(self.w, np.ndarray)
        return expit(safe_sparse_dot(X, self.w, dense_output=True))
    
    def loss(self, X, y):
        probas = self.predict_proba(X)
        return log_loss(y, probas)



