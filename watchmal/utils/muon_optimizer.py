'''
Muon optimizer wrapper using the official torch.optim.Muon.
'''
import torch


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """
    Combines torch.optim.Muon and torch.optim.AdamW in one optimizer object.

    Matrix parameters (ndim >= 2) use Muon; everything else uses AdamW. (Muon cannot be used for scalar parameters)
    """

    def __init__(self, params, lr, weight_decay, aux_lr, aux_betas, aux_weight_decay):
        params = list(params)
        if params and isinstance(params[0], dict):
            params = [p for group in params for p in list(group['params'])]

        muon_params = [p for p in params if p.ndim >= 2 and p.requires_grad]
        adam_params = [p for p in params if p.ndim <  2 and p.requires_grad]

        self._muon  = torch.optim.Muon(muon_params,  lr=lr,     weight_decay=weight_decay)
        self._adamw = torch.optim.AdamW(adam_params, lr=aux_lr, betas=aux_betas,
                                        weight_decay=aux_weight_decay)

        _dummy = [torch.empty(1, requires_grad=True)]
        super().__init__(_dummy, dict(lr=lr))
        self.param_groups.clear()
        self.state.clear()
        self.param_groups = self._muon.param_groups + self._adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)


    def state_dict(self):
        return {'muon': self._muon.state_dict(), 'adamw': self._adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self._muon.load_state_dict(state_dict['muon'])
        self._adamw.load_state_dict(state_dict['adamw'])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups
