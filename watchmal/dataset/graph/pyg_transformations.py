
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
class HilbertOrderTransform(torch.nn.Module):
    """
    Assigns each hit a Hilbert-curve rank by looking up its pmt_id against a
    precomputed table of Hilbert orderings (multiple angular-offset variants).
    A new variant is randomly sampled per event (per forward call) to avoid
    fixed token-to-PMT identity across training. Stores the rank as an extra
    feature/attribute for downstream windowing in CheRP's forward pass.
    """
    def __init__(self, hilbert_lookup_npz_path: str):
        super().__init__()
        lookup = np.load(hilbert_lookup_npz_path)
        rank_lookup = lookup["rank_lookup"]   # [num_variants, max_tube_id+1], int32

        self.register_buffer('rank_lookup', torch.from_numpy(rank_lookup).long())
        self.num_variants = rank_lookup.shape[0]

    def forward(self, data: Data) -> Data:
        if data.pmt_id.numel() == 0:
            data.hilbert_rank = torch.zeros((0,), dtype=torch.long, device=data.x.device)
            return data

        variant = torch.randint(0, self.num_variants, (1,)).item()

        # pmt_id is presumably 1-indexed (matching pmt_pos[pmt_noise - 1] usage above)
        # rank_lookup is indexed directly by tube_id value, so use pmt_id as-is unless
        # it needs the same -1 offset used elsewhere for pmt_pos lookups
        data.hilbert_rank = self.rank_lookup[variant][data.pmt_id]

        return data
class RandomOrder(torch.nn.Module):
    """
    Reorders nodes randomly (a fresh permutation each forward call), so
    CheRP's windowing — which derives node position from data.x's row order —
    groups together a random subset of hits each pass instead of hits
    adjacent in raw storage order.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        if data.x.size(0) == 0:
            return data

        perm = torch.randperm(data.x.size(0), device=data.x.device)

        data.x = data.x[perm]
        if hasattr(data, 'pmt_id') and data.pmt_id is not None:
            data.pmt_id = data.pmt_id[perm]
        if hasattr(data, 'is_noise') and data.is_noise is not None:
            data.is_noise = data.is_noise[perm]

        if data.edge_index.numel() > 0:
            inv_perm = torch.zeros_like(perm)
            inv_perm[perm] = torch.arange(len(perm), device=perm.device)
            data.edge_index = inv_perm[data.edge_index]

        return data
class TimeOrder(torch.nn.Module):
    """
    Transform to reorder nodes by time (first feature in x).
    """
    def __init__(self, time_index=0):
        super().__init__()
        self.time = time_index

    def forward(self, data: Data):
        # get time feature (first column)
        times = data.x[:, self.time]
        perm = times.argsort()  # ascending order

        # reorder node features
        data.x = data.x[perm]

        # reorder edge_index if present
        if data.edge_index.numel() > 0:
            inv_perm = torch.zeros_like(perm)
            inv_perm[perm] = torch.arange(len(perm), device=perm.device)
            data.edge_index = inv_perm[data.edge_index]

        return data
        
class TimingSmear(torch.nn.Module):
    """
    Adds Gaussian jitter to the time
    """
    def __init__(self, smear_ns: float = 0.5, precision_ns: float = 0.1, time_col: int = 1):
        super().__init__()
        self.smear_ns = smear_ns
        self.precision_ns = precision_ns
        self.time_col = time_col

    def forward(self, data: Data) -> Data:
        if data.x.size(0) == 0:
            return data

        # 1. Add the jitter
        data.x[:, self.time_col] += torch.randn(data.x.size(0), device=data.x.device) * self.smear_ns
        
        # 2. Snap to clock grid
        data.x[:, self.time_col] = torch.round(data.x[:, self.time_col] / self.precision_ns) * self.precision_ns
        
        return data
    
class AddPMTPositions(torch.nn.Module):
    """
    Appends PMT x,y,z to data.x using data.pmt_id as a lookup key.
    Expects data.x = [time, charge] (or any prefix); appends [x, y, z] columns.
    pmt_id values are 1-indexed (1–19746); geometry is 0-indexed (0–19745).
    """
    def __init__(self, geometry_npz_path: str):
        super().__init__()
        geo = np.load(geometry_npz_path)
        self.register_buffer('pmt_pos', torch.from_numpy(geo["position"]).float())

    def forward(self, data: Data) -> Data:
        positions = self.pmt_pos[data.pmt_id - 1]   # pmt_id is 1-indexed
        data.x = torch.cat([data.x, positions], dim=-1)
        return data


class DarkNoiseAugmentation(torch.nn.Module):
    """
    Ultra-fast, loop-free dark noise approximation with dynamic noise scaling.
    Samples a new dark rate from a uniform distribution per event on the fly,
    and uses the exact WCSim Box & Line 20-inch HQE continuous charge parameterization formula.
    
    Features layout is strictly enforced as: [Charge, Time, X, Y, Z]
    """
    def __init__(
        self,
        geometry_npz_path: str,
        dark_rate_range_khz: tuple = (5.0, 15.0), 
        t_low: float = 550.0,
        t_high: float = 1900.0,
        deadtime_ns: float = 560.0,               
        timing_precision_ns: float = 0.1,    
        pe_precision_pe: float = 0.3,        
        charge_col: int = 0,                         # [Charge=0, Time=1, X=2, Y=3, Z=4]
        time_col: int = 1,
        pos_cols=slice(2, 5),
        conv_rate: float = 1.1125
    ):
        super().__init__()
        self.dark_rate_low, self.dark_rate_high = dark_rate_range_khz
        self.t_low = t_low
        self.t_high = t_high
        self.deadtime_ns = deadtime_ns
        self.timing_precision_ns = timing_precision_ns
        self.pe_precision_pe = pe_precision_pe
        self.conv_rate = conv_rate
        
        self.charge_col = charge_col
        self.time_col = time_col
        self.pos_cols = pos_cols
        
        # Load detector PMT geometry positions
        geo = np.load(geometry_npz_path)
        self.register_buffer('pmt_pos', torch.from_numpy(geo["position"]).float())
        
        # Exact WCSim Box & Line 20-inch HQE CDF Table
        qpe_bnl = torch.tensor([
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000001, 0.000001, 0.000002, 0.000004,
            0.000008, 0.000014, 0.000025, 0.000044, 0.000486,
            0.007195, 0.019406, 0.031920, 0.044503, 0.057189,
            0.070020, 0.083060, 0.096388, 0.110108, 0.124351,
            0.139276, 0.155072, 0.171956, 0.190167, 0.209961,
            0.231594, 0.255310, 0.281319, 0.309777, 0.340762,
            0.374259, 0.410142, 0.448167, 0.487976, 0.529101,
            0.570993, 0.613041, 0.654608, 0.695067, 0.733833,
            0.770390, 0.804317, 0.835304, 0.863151, 0.887777,
            0.909203, 0.927543, 0.942987, 0.955778, 0.966198,
            0.974543, 0.981116, 0.986205, 0.990078, 0.992974,
            0.995104, 0.996642, 0.997734, 0.998495, 0.999017,
            0.999369, 0.999601, 0.999752, 0.999848, 0.999909,
            0.999946, 0.999969, 0.999982, 0.999990, 0.999994,
            0.999997, 0.999998, 0.999999, 1.000000,
        ], dtype=torch.float32)
        
        # Pad remainder out to match the 500 element WCSim array bounds
        self.register_buffer('spe_cdf', torch.cat([qpe_bnl, torch.ones(500 - len(qpe_bnl))]))

    def _sample_spe_charge(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Vectorized equivalent to WCSimWCPMT::rn1pe() using your exact parameterization formula."""
        u = torch.rand(num_samples, device=device)
        u2 = torch.rand(num_samples, device=device)
        
        # Binary search lookup on GPU array to get index matching the probability u
        indices = torch.searchsorted(self.spe_cdf, u).clamp(0, 499)
        
        # Your exact WCSim mathematical parameterization formula
        charge = (indices.float() - 50.0 + u2) / 22.83
        return torch.clamp(charge, min=0.01) # Safety floor against negative edge conditions

    def forward(self, data: Data) -> Data:
        # Pass data straight through untouched if running validation/evaluation loops
        if not torch.is_grad_enabled() or data.x.size(0) == 0:
            return data

        device = data.x.device
        n_pmts = self.pmt_pos.size(0)
        orig_n = data.x.size(0)

        # 1. Dynamic uniform dark rate sampling — stay in torch to avoid numpy interop
        sampled_rate = torch.empty(1, device=device).uniform_(self.dark_rate_low, self.dark_rate_high).item()
        current_dark_rate_khz = sampled_rate * self.conv_rate

        # 2. Calculate fast deadtime proxy drop fraction
        drop_rate = current_dark_rate_khz * self.deadtime_ns * 1e-6

        # 3. Fast statistical throughput proxy for deadtime drops
        if drop_rate > 0:
            keep_mask = torch.rand(orig_n, device=device) > drop_rate
            x_base = data.x[keep_mask]
            pmt_base = data.pmt_id[keep_mask]
        else:
            x_base = data.x
            pmt_base = data.pmt_id

        # 4. Generate Poisson dark noise hit counts for total window
        window_ns = self.t_high - self.t_low
        lam = n_pmts * current_dark_rate_khz * window_ns * 1e-6
        total_noise = int(torch.poisson(torch.tensor(lam, dtype=torch.float32, device=device)).item())

        if total_noise > 0:
            t_noise = torch.rand(total_noise, device=device) * window_ns + self.t_low
            pmt_noise = torch.randint(1, n_pmts + 1, (total_noise,), device=device)

            # --- SEED INTEGRATED SPE SPECTRUM FORMULA ---
            q_noise = self._sample_spe_charge(total_noise, device=device)

            # All 5 columns written explicitly — no need to zero-init
            x_noise = torch.empty(total_noise, 5, device=device)
            x_noise[:, self.charge_col] = q_noise
            x_noise[:, self.time_col] = t_noise
            x_noise[:, self.pos_cols] = self.pmt_pos[pmt_noise - 1]

            n_base = x_base.size(0)
            is_noise_flag = torch.zeros(n_base + total_noise, dtype=torch.bool, device=device)
            is_noise_flag[n_base:] = True

            final_x = torch.cat([x_base, x_noise], dim=0)
            final_pmt = torch.cat([pmt_base, pmt_noise], dim=0)
        else:
            final_x = x_base
            final_pmt = pmt_base
            is_noise_flag = torch.zeros(x_base.size(0), dtype=torch.bool, device=device)

        if final_pmt.numel() == 0:
            return Data(x=torch.zeros((0, 5), device=device), pmt_id=torch.zeros((0,), dtype=torch.long), y=data.y)

        # 5. Electronic/Hardware Quantization Digitize Rounding
        final_x[:, self.time_col] = torch.round(final_x[:, self.time_col] / self.timing_precision_ns) * self.timing_precision_ns
        final_x[:, self.charge_col] = torch.round(final_x[:, self.charge_col] / self.pe_precision_pe) * self.pe_precision_pe

        data.x = final_x
        data.pmt_id = final_pmt
        data.is_noise = is_noise_flag

        return data
class AddDarkNoise(torch.nn.Module):
    """
    Blazing fast, fully vectorized version of the DAQ-aware dark noise transform.
    Uses exact WCSim BoxandLine20inchHQE CDF table for SPE charge sampling.
    Features layout is strictly enforced as: [Charge, Time, X, Y, Z]
    """
    def __init__(
        self,
        geometry_npz_path: str,
        dark_rate_khz: float = 8.41,
        conv_rate: float = 1.1125,
        t_low: float = 0.0,
        t_high: float = 4000.0,
        deadtime_ns: float = 560.0,
        integration_window_ns: float = 200.0,
        timing_precision_ns: float = 0.1,
        pe_precision_pe: float = 0.3,
        charge_col: int = 0,
        time_col: int = 1,
        pos_cols=slice(2, 5),
    ):
        super().__init__()
        self.dark_rate_khz = dark_rate_khz
        self.conv_rate = conv_rate
        self.t_low = t_low
        self.t_high = t_high
        self.deadtime_ns = deadtime_ns
        self.integration_window_ns = integration_window_ns
        self.timing_precision_ns = timing_precision_ns
        self.pe_precision_pe = pe_precision_pe
        self.charge_col = charge_col
        self.time_col = time_col
        self.pos_cols = pos_cols
        
        # Load geometry configurations
        geo = np.load(geometry_npz_path)
        self.register_buffer('pmt_pos', torch.from_numpy(geo["positions"]).float())

        qpe0 = np.array([
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000001, 0.000001, 0.000002, 0.000004,
                    0.000008, 0.000014, 0.000025, 0.000044, 0.000486,
                    
                    0.007195, 0.019406, 0.031920, 0.044503, 0.057189,
                    0.070020, 0.083060, 0.096388, 0.110108, 0.124351,
                    0.139276, 0.155072, 0.171956, 0.190167, 0.209961,
                    0.231594, 0.255310, 0.281319, 0.309777, 0.340762,
                    0.374259, 0.410142, 0.448167, 0.487976, 0.529101,
                    0.570993, 0.613041, 0.654608, 0.695067, 0.733833,
                    0.770390, 0.804317, 0.835304, 0.863151, 0.887777,
                    0.909203, 0.927543, 0.942987, 0.955778, 0.966198,
                    0.974543, 0.981116, 0.986205, 0.990078, 0.992974,
                    0.995104, 0.996642, 0.997734, 0.998495, 0.999017,
                    0.999369, 0.999601, 0.999752, 0.999848, 0.999909,
                    0.999946, 0.999969, 0.999982, 0.999990, 0.999994,
                    0.999997, 0.999998, 0.999999, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    ], dtype=np.float32)

        self.register_buffer(
            "qpe_cdf",
            torch.tensor(qpe0, dtype=torch.float32)
        )

    def _rn1pe_batch(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample SPE charge exactly like WCSim PMT20inch::rn1pe().
        """
        qpe_cdf = self.qpe_cdf.to(device)

        u = torch.rand(n, device=device)
        u2 = torch.rand(n, device=device)

        idx = torch.searchsorted(qpe_cdf, u)

        charge = (idx.float() - 50.0 + u2) / 22.83

        return charge

    def forward(self, data: Data) -> Data:
        device = data.x.device
        n_pmts = self.pmt_pos.size(0)
        orig_n = data.x.size(0)
        
        # 1. Sample Raw Dark Noise counts across event window bounds
        window_ns = self.t_high - self.t_low
        lam = n_pmts * self.dark_rate_khz * self.conv_rate * window_ns * 1e-6
        total_noise = int(torch.poisson(torch.tensor(lam, device=device)).item())
        
        if total_noise == 0 and orig_n == 0:
            return Data(x=torch.zeros((0, 5), device=device), 
                        pmt_id=torch.zeros((0,), dtype=torch.long, device=device), y=data.y)

        # Draw raw parameters using your specific macro distribution math
        if total_noise > 0:
            t_noise = torch.rand(total_noise, device=device) * window_ns + self.t_low
            pmt_idx_noise = torch.randint(1, n_pmts + 1, (total_noise,), device=device)
            q_noise = self._rn1pe_batch(total_noise, device=device)
        else:
            t_noise = torch.empty(0, device=device)
            pmt_idx_noise = torch.empty(0, dtype=torch.long, device=device)
            q_noise = torch.empty(0, device=device)

        # 2. Merge data and noise streams directly (Maintains original native file pmt_id tracking)
        if orig_n > 0:
            all_pmts = torch.cat([data.pmt_id.to(device), pmt_idx_noise], dim=0)
            all_times = torch.cat([data.x[:, self.time_col], t_noise], dim=0)
            all_charges = torch.cat([data.x[:, self.charge_col], q_noise], dim=0)
            is_noise_flag = torch.cat([
                torch.zeros(orig_n, dtype=torch.bool, device=device),
                torch.ones(total_noise, dtype=torch.bool, device=device)
            ], dim=0)
        else:
            all_pmts = pmt_idx_noise
            all_times = t_noise
            all_charges = q_noise
            is_noise_flag = torch.ones(total_noise, dtype=torch.bool, device=device)

        # 3. Chronological sorting
        sort_idx = torch.argsort(all_times)
        all_pmts = all_pmts[sort_idx]
        all_times = all_times[sort_idx]
        all_charges = all_charges[sort_idx]
        is_noise_flag = is_noise_flag[sort_idx]

        # 4. Vectorized Digitizer State Processing
        pmt_sort_idx = torch.argsort(all_pmts, stable=True)
        all_pmts = all_pmts[pmt_sort_idx]
        all_times = all_times[pmt_sort_idx]
        all_charges = all_charges[pmt_sort_idx]
        is_noise_flag = is_noise_flag[pmt_sort_idx]

        pmt_changed = torch.cat([torch.tensor([True], device=device), all_pmts[1:] != all_pmts[:-1]])
        
        keep_mask = torch.zeros_like(all_pmts, dtype=torch.bool)
        accum_target_idx = torch.zeros_like(all_pmts, dtype=torch.long)

        current_idx = torch.where(pmt_changed)[0]
        
        while current_idx.numel() > 0:
            keep_mask[current_idx] = True
            base_times = all_times[current_idx]
            base_pmts = all_pmts[current_idx]
            
            gate_end = base_times + self.integration_window_ns
            dead_end = base_times + self.deadtime_ns
            
            next_idx = current_idx + 1
            
            while True:
                valid_next = (next_idx < all_pmts.size(0)) & (all_pmts[next_idx] == base_pmts)
                if not valid_next.any():
                    break
                    
                active_next = next_idx[valid_next]
                active_base_mapping = current_idx[valid_next]
                next_times = all_times[active_next]
                
                in_gate = next_times < gate_end[valid_next]
                if in_gate.any():
                    accum_target_idx[active_next[in_gate]] = active_base_mapping[in_gate]

                next_idx = next_idx + 1

            search_next = current_idx + 1
            found_next = []
            
            while search_next.numel() > 0:
                valid_search = (search_next < all_pmts.size(0)) & (all_pmts[search_next] == all_pmts[current_idx[:search_next.numel()]])
                if not valid_search.any():
                    break
                
                past_deadtime = all_times[search_next[valid_search]] >= dead_end[valid_search]
                if past_deadtime.any():
                    found_next.append(search_next[valid_search][past_deadtime])
                    break
                search_next = search_next + 1
                
            if found_next:
                current_idx = torch.cat(found_next)
            else:
                break

        final_pmts = all_pmts[keep_mask]
        final_times = all_times[keep_mask]
        final_is_noise = is_noise_flag[keep_mask]
        
        final_charges = torch.zeros_like(final_times)
        final_charges.index_add_(0, accum_target_idx[~keep_mask], all_charges[~keep_mask])
        final_charges += all_charges[keep_mask]

        if final_pmts.numel() == 0:
            return Data(x=torch.zeros((0, 5), device=device), 
                        pmt_id=torch.zeros((0,), dtype=torch.long, device=device), y=data.y)

        # 5. Enforce Hardware Step Resolution (Quantization)
        out_times = torch.round(final_times / self.timing_precision_ns) * self.timing_precision_ns
        out_charges = torch.round(final_charges / self.pe_precision_pe) * self.pe_precision_pe

        # 6. Rebuild Matrix directly into specified [Charge, Time, X, Y, Z] layout
        total_final_hits = final_pmts.size(0)
        noise_x = torch.zeros(total_final_hits, 5, device=device)
        
        noise_x[:, self.charge_col] = out_charges            
        noise_x[:, self.time_col]   = out_times              
        noise_x[:, self.pos_cols]   = self.pmt_pos[final_pmts] 

        data.x = noise_x
        data.pmt_id = final_pmts  
        data.is_noise = final_is_noise

        return data
class ThreeMomentum(torch.nn.Module):  
    '''
    Transform from H5GraphDataset full regression [energy,vertex_time,vertex_position,direction] to [3 momentum, vertex_time,vertex_position] 
    optionally with relative loss (3 momentum / ||momentum||)
    '''
    def __init__(self):
        super().__init__()
    def forward(self, data):
        y = torch.squeeze(data.y)
        energy = y[0]
        vertex_time = y[1]
        vertex_position = y[2:5]
        direction = y[5:8]

        momentum = energy * direction

        new_y = torch.cat([momentum, vertex_time.unsqueeze(0), vertex_position,direction])
        data.y = new_y

        return data

class RandomCylindricalTransform(torch.nn.Module):
    """
    Cylinder-preserving augmentation:

    1. Random rotation around z.
    2. Random upside-down flip: z → -z, (dx,dy,dz) → rotated then dz → -dz.

    Applied to:
      - data.x[:, -3:]      -> PMT positions
      - data.y[-6:-3]       -> vertex (vx,vy,vz)
      - data.y[-3:]         -> direction (dx,dy,dz)
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        device = data.x.device

        # ------------------------------------------------------------
        # 1. Random rotation around z
        # ------------------------------------------------------------
        theta = torch.rand(1, device=device) * 2 * torch.pi
        c = torch.cos(theta)
        s = torch.sin(theta)

        Rz = torch.tensor([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)

        # ------------------------------------------------------------
        # 2. Random upside-down flip z → −z
        # ------------------------------------------------------------
        if torch.rand(1, device=device) < 0.5:
            Flip = torch.tensor([
                [ 1.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0],
                [ 0.0, 0.0,-1.0]
            ], device=device)
        else:
            Flip = torch.eye(3, device=device)

        R = Flip @ Rz

        # ------------------------------------------------------------
        # Apply to PMT positions
        # ------------------------------------------------------------
        if data.x is not None and data.x.size(1) >= 3:
            pos = data.x[:, -3:]
            data.x[:, -3:] = pos @ R.T

        # ------------------------------------------------------------
        # Apply to targets
        # ------------------------------------------------------------
        if data.y is not None and data.y.numel() >= 6:
            y = data.y.clone()

            v = y[-6:-3]
            d = y[-3:]

            v_rot = (v.unsqueeze(0) @ R.T).squeeze(0)
            d_rot = (d.unsqueeze(0) @ R.T).squeeze(0)

            d_rot = d_rot / (torch.norm(d_rot) + 1e-8)

            y[-6:-3] = v_rot
            y[-3:]   = d_rot
            data.y   = y

        return data


class CenterTimeByEvent(torch.nn.Module):
    """
    Centers the time feature on its own per-event mean, making it invariant to any
    constant additive shift applied uniformly to all hit times in the event (e.g.
    trigger jitter). Preserves within-event relative timing structure exactly, since
    (t_i + delta) - mean(t + delta) = t_i - mean(t) for any delta.

    Args:
        time_col (int): index in data.x to center
    """

    def __init__(self, time_col=0):
        super().__init__()
        self.time_col = time_col

    def forward(self, data):
        if data.x.size(0) == 0:
            return data
        t = data.x[:, self.time_col]
        data.x[:, self.time_col] = t - t.mean()
        return data


class FixVertexTime(torch.nn.Module):
    """
    Makes the vertex time component make sense with respect to the timings in a given event.
    Custom transform to correct a target value in data.y.
    Given an index `time_index`, it computes:
        corrected_time = 950 - data.y[time_index]
    and stores it back or in a new field `data.corrected_time`.
    
    Args:
        time_index (int): index in the target vector to use for correction
        inplace (bool): if True, overwrite data.y; if False, store in data.corrected_time
    """

    def __init__(self, time_index=0, inplace=True):
        super().__init__()
        self.time_index = time_index
        self.inplace = inplace

    def forward(self, data):
        # ensure y is at least 1D
        y = torch.squeeze(data.y)

        # get the value at the target index
        value_to_subtract = y[self.time_index]

        # compute corrected time
        corrected_time = 950 - value_to_subtract

        if self.inplace:
            data.y[self.time_index] = corrected_time
        else:
            data.corrected_time = corrected_time

        return data
class SliceY(torch.nn.Module):
    """
    Transform to slice `data.y` in place. Useful if you have a dataset with all targets (energy, vertex, direction etc.) but want a subset.
    
    Args:
        slice_indices (slice or list/torch.Tensor): indices to select from data.y
    """
    def __init__(self, slice_indices):
        super().__init__()
        self.slice_indices = slice_indices

    def forward(self, data):
        y = torch.squeeze(data.y)

        # slice y
        if isinstance(self.slice_indices, slice):
            y_sliced = y[self.slice_indices]
        else:
            indices = torch.tensor(self.slice_indices, device=y.device)
            y_sliced = y[indices]

        # overwrite y in place
        data.y = y_sliced
        return data
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