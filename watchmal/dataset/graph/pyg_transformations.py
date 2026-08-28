
import numpy as np
import torch
import omegaconf
from omegaconf import OmegaConf
from torch_geometric.data import Data
from watchmal.dataset.graph.data_utils import match_type
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


"""
This file should contain all the callables (i.e python functions or class with a __call__ attribut) 
used for creating graph datasets.

"""

# À FAIRE : 
# 30/01 :  - Ajouter une erreur si les tailles de feat/target_norm et du nombre de features dans data.x/y ne correspondent pas
#          - Ajouter de la doc sur les appels [0] et [1]
# 14/02 :  - Mettre à jour la doc de cette fonction
class Normalize(torch.nn.Module):
    """Normalize a torch_geometric Data object with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(
            self, 
            feat_norm,
            target_norm=None, 
            apply_log=[False],
            target_apply_log=[False],
            eps=1e-8, 
            inplace=False        
    ):
        
        super().__init__()
        
        self.feat_norm   = feat_norm
        self.target_norm = target_norm
        self.eps         = eps
        self.inplace     = inplace
        self.apply_log   = apply_log
        self.target_apply_log = target_apply_log

        # For hydra compatibility
        if isinstance(self.feat_norm, omegaconf.listconfig.ListConfig):
            self.feat_norm = OmegaConf.to_container(self.feat_norm)
            if self.target_norm is not None:
                self.target_norm = OmegaConf.to_container(self.target_norm)

        # Need to convert list to torch tensor to perform addition & subtraction
        self.feat_norm = torch.tensor(self.feat_norm)
        if self.target_norm is not None:
            self.target_norm = torch.tensor(self.target_norm)


    def forward(self, data):
        """
        self.feat_norm and self.target_norm must contain Tensor object
        """    
   
        if self.feat_norm is not None:

            if data.x.dim() == 1:
                data.x = torch.unsqueeze_copy(data.x, 1)
                # print(f"Data size after unsqueeze {data.x.dim()} ; {data.x.size()}")

            for ft_index in range(data.x.size(dim=1)):
                if self.apply_log[ft_index]:
                    ##log1p to handle zero charge
                    data.x[:, ft_index] = (data.x[:, ft_index].log1p() - self.feat_norm[1, ft_index].log1p()) / (self.feat_norm[0, ft_index].log1p() - self.feat_norm[1, ft_index].log1p() + self.eps) 
                else :
                    data.x[:, ft_index] = (data.x[:, ft_index] - self.feat_norm[1, ft_index]) / (self.feat_norm[0, ft_index] - self.feat_norm[1, ft_index] + self.eps)

        if self.target_norm is not None:
            data.y = torch.squeeze(data.y) # Remove any 1d dimension of the tensor for compatibility with the way we compute normalization            
            
            # Erwan - To do : add support for multi dim target with log norm
            if self.target_apply_log[0]:
                data.y = (data.y.log() - self.target_norm[1].log()) / (self.target_norm[0].log() - self.target_norm[1].log() + self.eps)
            else :
                data.y = (data.y - self.target_norm[1]) / (self.target_norm[0] - self.target_norm[1] + self.eps)

        return data


class MapLabels(torch.nn.Module):
    """
    Arguments:
        label_set (list of integers) : which label to assign the PID to. 
            For example label_set=[11, 13, 111] will convert 11 -> 0, 13 -> 1 and 111 -> 2. 
    """
    def __init__(self, label_set: list):
        super().__init__()
        self.label_set = list(label_set)

    def forward(self, data):
        y = data.y
        if y.dim() == 0 or y.numel() == 1:
            new_target = self.label_set.index(y.item())
            data.y = torch.tensor([new_target])
        else:
            new_targets = [self.label_set.index(v.item()) for v in y]
            data.y = torch.tensor(new_targets)

        return data


class AddFeaturesInData(torch.nn.Module):
    def __init__(
        self,
        feature_names: list[str],
        min_vals: list[float],
        max_vals: list[float],
        charge_index: int | None = None,
        eps: float = 1e-10,
    ):
        super().__init__()
        assert (
            'event_total_charge' not in feature_names or charge_index is not None
        ), "If you normalize event_total_charge you must pass a charge_index"

        self.feature_names = feature_names
        self.charge_index = charge_index

        # register buffers so that .to(device) moves them automatically

        min_vals = OmegaConf.to_container(min_vals, resolve=True)
        max_vals = OmegaConf.to_container(max_vals, resolve=True)
        #print(f"Min values: {min_vals}")
        #print(f"Max values: {max_vals}")

        self.register_buffer('min_vals', torch.tensor(min_vals, dtype=torch.float))
        self.register_buffer('max_vals', torch.tensor(max_vals, dtype=torch.float))
        
        # precompute denominator = max – min + eps
        self.register_buffer('denom_vals',
                             self.max_vals - self.min_vals + eps)

    def forward(self, data):

        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        
        # grab x-device once
        device = data.x.device


        for i, name in enumerate(self.feature_names):

            lo   = self.min_vals[i]   # already on `device`
            rng  = self.denom_vals[i] # ditto
            #print(f"Adding feature {name} with lo: {lo}, rng: {rng})")

            if name == 'n_hits':
                # We suppose number of nodes is data.x.shape[0]
                count = torch.tensor(data.x.size(0), device=device, dtype=torch.float)
                value = (count - lo) / rng

            elif name == 'event_total_charge':
                charge = data.x[:, self.charge_index].sum()
                value  = (charge - lo) / rng

            elif name == 'prefit':
                prefit = data.prefit if hasattr(data, 'prefit') else None
                prefit_dim = prefit.size(dim=0) if prefit is not None else 0
                if prefit is not None:
                    value = (prefit - self.min_vals[i:prefit_dim+1]) / self.denom_vals[i:prefit_dim+1]
                else:
                    log.warning("Data does not have 'prefit' attribute. Skipping feature addition.")
                    value = None
                
            else:
                raise ValueError(f"Unknown feature {name!r}")

            setattr(data, name, value)

        return data


class ConvertAndToDict(torch.nn.Module):
    """
    Arguments:
        feature_to_type (string)      : type to convert each feature to
        target_to_type (string)       : type to convert the target to 

    Note :
        - "idx" is only used when looking at the output folder (for evaluation only in watchmal)
        - The class Negative Log Likelyhood of torch requires int as labels
    """

    def __init__(self, feature_to_type: str, target_to_type: str):
        super().__init__()

        # We consider homogeneous graphs for now (the 'Data' obj). 
        # So all features are of the same type
        # Hence the should be converted to the same type also (no need to make a list for feat..type and target..type)
        self.feature_to_type = match_type(feature_to_type) 
        self.target_to_type  = match_type(target_to_type)

    def forward(self, data):

        # log.info(f"\n\nType de data : {type(data.y)}")
        # log.info(f"Convert - Contenant : {data.y}\n\n")
        
        if isinstance(data, dict): 
            return data

        data.x = data.x.to(self.feature_to_type)
        data.y = data.y.to(self.target_to_type)

        data_dict = {
            'data': data,
            'target': data.y,
            'indice': data.idx # might a better for for this. We duplicate the idx information (data + dict)
        }

        return data_dict 
    


class Threshold(torch.nn.Module):

    def __init__(self, key: str, max_thresholds: list, min_thresholds: list):

        super().__init__()

        self.key = key

        # And inf when no thresholds
        max_thresholds = [value if value is not None else torch.inf for value in max_thresholds]
        min_thresholds = [value if value is not None else -torch.inf for value in min_thresholds]

        # Convert to torch tensor necessary for torch.maximum and torch.minimum
        self.max_thresholds = torch.tensor(max_thresholds)
        self.min_thresholds = torch.tensor(min_thresholds)

        assert len(self.max_thresholds) == len(self.min_thresholds), f"max_threshold and min_thresold should be of same lentgh. Received {len(self.max_thresholds)} and {len(self.min_thresholds)}."

    def forward(self, data):

        #log.info(f"\n\nType de data : {type(data)}")
        #log.info(f"Threshold - Contenant : {data.x[:5]}\n\n")

        att = getattr(data, self.key) # return the attribute directly. No deepcopy
        assert att.shape[1] == len(self.max_thresholds), f"The threshold list does not match the input shape. Threshold : {len(self.thresholds)} vs {att.shape[0]} for data."

        # Need to use att.clamp_(..) to do in-place modification, 
        # torch.clamp(att, ..) points to a new tensor
        att.clamp_(min=self.min_thresholds, max=self.max_thresholds)

        return data