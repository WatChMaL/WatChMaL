
from omegaconf import DictConfig
from omegaconf import OmegaConf

# torch import
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# hydra imports 
from hydra.utils import instantiate

# watchmal import
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


def build_model(model_config, device, use_ddp=False):
    """
    Build the model and wrap it with SynBatchNorm and  data_config if using torch DDP
    """
    model = instantiate(model_config)
    nb_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if ( device == 'cpu') or ( str(device) in ['0', 'cuda:0'] ):
        log.info(f"Number of parameters in the model : {nb_parameters}\n")

    model.to(device)

    if use_ddp:
        # Convert model batch norms to synchbatchnorm (if the model contains BatchNorm layers)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap the model with DistributedDataParallel mode
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    return model, nb_parameters


def build_segmentation_model(encoder, head, wrapper, aux_head=None):
    """Build a segmentation model from Hydra configs. aux_head is optional (default
    None); if its c_in is null it is filled from the encoder."""
    enc = instantiate(encoder)
    hd = instantiate(head)
    aux = None
    if aux_head is not None:
        aux_kwargs = {}
        if "c_in" not in aux_head or aux_head.get("c_in", None) is None:
            aux_kwargs["c_in"] = enc.c_bottleneck
        aux = instantiate(aux_head, **aux_kwargs)
    return instantiate(wrapper, encoder=enc, head=hd, aux_head=aux)


def merge_config(hydra_config, wandb_config):
    """
    Update Hydra configuration with values from W&B configuration if the keys match.
    
    Args:
    - hydra_config (omegaconf.DictConfig): The Hydra configuration object.
    - wandb_config (dict): The dictionary containing W&B configuration.
    
    Returns:
    - hydra_config (omegaconf.DictConfig): The updated Hydra configuration object.
    """
    modified_keys, not_found_keys =[], []
    for key, value in wandb_config.items():
		
        # list_of_keys = ['data', 'dataset', 'root_file_path'] e.g.
        list_of_keys = key.split("-") 

        # key_name = ['root_file_path'] e.g.
        key_name = list_of_keys[-1] 

        # define the intial location 
        location = hydra_config 

        # Update the location based on the directory structure
        try:
            if len(list_of_keys) == 1:
                i = list_of_keys[0]
                location = location[i]
            else:
                for i in list_of_keys[0:-1]:
                    location = location[i]
            
        except Exception as e:
            log.debug(f"{list_of_keys} not found ({e})")
            not_found_keys.append(key)

        else:
            location[key_name] = value
            modified_keys.append(key)
    
        
	# End of the loop over wandb_config
    log.info(f"Modified keys: {modified_keys}")
    log.info(f"Keys not found: {not_found_keys}")

    return hydra_config