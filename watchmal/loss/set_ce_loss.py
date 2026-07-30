# -*- coding: utf-8 -*-
"""
Set-matching cross-entropy + Dice loss for per-voxel ring segmentation.

The model outputs Pi (V x K) — a probability distribution over K ring slots
for each of V voxels. The ground truth y (voxel_parent_frac, V x K) gives the
fractional charge contribution of each ring to each voxel (soft target).

Two terms are combined:
  CE   : soft cross-entropy H(y, Pi) = -sum_v sum_k y[v,k] * log(Pi[v,k])
  Dice : (1 - Dice(Pi[:,k], y[:,k])) summed over foreground rings k=1..K-1.

  total = CE + lambda_dice * Dice

Optional matching (matching='hungarian' or 'min'):
  Resolves ring-label permutation ambiguity by finding the assignment of GT
  columns to predicted columns that minimises Dice cost before computing CE.
  
Ref: DETR paper : Carion et al, "End-to-End Object Detection with Transformers" 

Additions :
Optional symmetric=True:
  Symmetrises CE as 0.5 * [H(y, Pi) + H(Pi, y)].
Optionnal poisson=True:
    Weights the CE loss by the observed digicharge in each voxel

"""
from __future__ import annotations
from typing import Dict, Any, List
import logging
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)
_LOG_LAMBDA_MAX = 30.0

def _sanitised_cost(cost: torch.Tensor, where: str) -> np.ndarray:
    if not torch.isfinite(cost).all():
        n_bad = int((~torch.isfinite(cost)).sum())
        log.warning(
            "%s: %d/%d non-finite entries in the matching cost matrix "
            "(cost finite-range=[%s, %s]); repairing and continuing.",
            where, n_bad, cost.numel(),
            f"{cost[torch.isfinite(cost)].min():.4g}" if torch.isfinite(cost).any() else "n/a",
            f"{cost[torch.isfinite(cost)].max():.4g}" if torch.isfinite(cost).any() else "n/a",
        )
        cost = torch.nan_to_num(cost, nan=0.0, posinf=1e12, neginf=-1e12)
    return cost.detach().cpu().numpy()


def dice_coeff(prob: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    num = 2.0 * (prob * target_bin).sum()
    den = prob.sum() + target_bin.sum() + eps
    return num / den


def smallK_match(cost: torch.Tensor) -> List[int]:
    N, K = cost.shape
    if K == 0 or N == 0:
        return []
    assign: List[int] = []
    for kk in range(K):
        col = cost[:, kk]
        qidx = int(torch.argmin(col).item())
        assign.append(qidx)
    return assign


def _stable_probs(pi: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    pi = pi.clamp(0.0, 1.0)
    row_sum = pi.sum(dim=1, keepdim=True).clamp_min(eps)
    return pi / row_sum



def loss_set_ce_dice(lambda_dice: float = 2.0,
                     eps: float = 1e-10,
                     poisson: bool = False,
                     matching: str = "none",
                     symmetric: bool = False):

    def _loss(batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        Pi_list: List[torch.Tensor] = out["Pi_list"]
        device = Pi_list[0].device if len(Pi_list) > 0 else "cpu"

        coords_all = batch.get("coords", None)
        feats_all = batch.get("feats", None)
        if poisson:
            coords_all = coords_all.to(device)
            feats_all = feats_all.to(device)

        ce_sum = torch.tensor(0.0, device=device)
        dice_sum = torch.tensor(0.0, device=device)
        total_weight = 0.0
        total_vox = 0
        B = len(out["Pi_list"])

        for b in range(B):
            Pi_b = out["Pi_list"][b]
            if Pi_b.numel() == 0:
                continue

            Pi_b = _stable_probs(Pi_b, eps=eps)
            logPi_b = (Pi_b + eps).log()

            meta_b = batch["meta"][b]

            y_b = meta_b["voxel_parent_frac"].to(Pi_b.device).float()
            if y_b.dim() == 1:
                y_b = y_b.unsqueeze(1)

            V_b, C = Pi_b.shape

            C_t = y_b.shape[1]
            if C_t < C:
                pad = torch.zeros((V_b, C - C_t), device=Pi_b.device, dtype=y_b.dtype)
                y_b = torch.cat([y_b, pad], dim=1)
            elif C_t > C:
                y_b = y_b[:, :C]

            y_b = _stable_probs(y_b, eps=eps)

            if matching != "none" and C > 1:
                # Foreground-only soft-Dice cost matrix, vectorized:
                #   cost[ip, ig] = 1 - 2*<Pi[:,ip+1], y[:,ig+1]> / (Pi[:,ip+1].sum() + y[:,ig+1].sum() + eps)
                Pi_fg = Pi_b[:, 1:]                                  # [V, C-1]
                y_fg = y_b[:, 1:]                                    # [V, C-1]
                inter = Pi_fg.t() @ y_fg                             # [C-1, C-1]
                pi_sum = Pi_fg.sum(dim=0)                            # [C-1]
                y_sum = y_fg.sum(dim=0)                              # [C-1]
                cost = 1.0 - (2.0 * inter) / (pi_sum[:, None] + y_sum[None, :] + eps)

                if matching == "min":
                    assign = smallK_match(cost)
                    perm_inv = [-1] * (C - 1)
                    for gt_idx, pred_idx in enumerate(assign):
                        if 0 <= pred_idx < (C - 1):
                            perm_inv[pred_idx] = gt_idx
                    for pred_idx in range(C - 1):
                        if perm_inv[pred_idx] < 0:
                            perm_inv[pred_idx] = pred_idx
                    perm = torch.as_tensor(perm_inv, device=y_b.device, dtype=torch.long)
                    y_b = torch.cat([y_b[:, :1], y_b[:, 1:][:, perm]], dim=1)

                elif matching == "hungarian":
                    row_ind, col_ind = linear_sum_assignment(_sanitised_cost(cost, "loss_set_ce_dice"))
                    perm = torch.as_tensor(col_ind, device=y_b.device, dtype=torch.long)
                    y_b = torch.cat([y_b[:, :1], y_b[:, 1:][:, perm]], dim=1)

                else:
                    raise ValueError("matching must be one of: 'none', 'min', 'hungarian'")

            ce_vec = -(y_b * logPi_b).sum(dim=1)
            if symmetric and C > 1:
                ce_vec_sym = -(Pi_b * (y_b + eps).log()).sum(dim=1)
                ce_vec = 0.5 * (ce_vec + ce_vec_sym)
            

            if poisson:
                mask_b = (coords_all[:, 0] == b)

                w_v = feats_all[mask_b, 0].float().clamp(0.0, 10.0)
                valid = w_v > 0.0

                if valid.any():
                    ce_sum = ce_sum + (w_v[valid] * ce_vec[valid]).sum()
                    total_weight += float(w_v[valid].sum().item())
                else:
                    ce_sum = ce_sum + ce_vec.mean()
                    total_weight += 1.0
            else:
                ce_sum = ce_sum + ce_vec.sum()
                total_vox += V_b

            if lambda_dice > 0.0 and C > 1:
                dsum = 0.0
                for c in range(1, C):
                    pc = Pi_b[:, c]
                    tc = y_b[:, c]
                    if float(tc.sum().item()) < eps:
                        continue
                    dsum += (1.0 - dice_coeff(pc, tc, eps=eps))
                dice_sum = dice_sum + dsum

        if poisson:
            denom = max(total_weight, 1.0)
            ce = ce_sum / float(denom)
        else:
            vox_ = max(total_vox, 1)
            ce = ce_sum / float(vox_)

        if lambda_dice > 0.0 and B > 0:
            dice = dice_sum / float(B)
        else:
            dice = torch.tensor(0.0, device=device)

        total = ce + lambda_dice * dice
        return {"loss": total, "loss_ce": ce, "loss_dice": dice}

    return _loss

def loss_set_poisson(lambda_count: float = 1.0,
                     lambda_dice: float = 2.0,
                     eps: float = 1e-8,
                     matching: str = "hungarian",
                     full: bool = True,
                     dice_kind: str = "occupancy"):

    import torch.nn.functional as F

    def _loss(batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        S_list: List[torch.Tensor] = out["LogLambda_list"]
        device = S_list[0].device if len(S_list) > 0 else "cpu"

        nll_sum = torch.zeros((), device=device)
        dice_sum = torch.zeros((), device=device)
        total = 0.0
        B = len(S_list)

        for b in range(B):
            S_b = S_list[b]
            if S_b.numel() == 0:
                continue
            V, C = S_b.shape

            y_b = batch["meta"][b]["voxel_parent_npe"].to(device).float()
            if y_b.dim() == 1:
                y_b = y_b.unsqueeze(1)
            Ct = y_b.shape[1]
            if Ct < C:
                y_b = torch.cat([y_b, y_b.new_zeros(V, C - Ct)], dim=1)
            elif Ct > C:
                y_b = y_b[:, :C]

            if matching != "none" and C > 1:
                lam = S_b.clamp(max=_LOG_LAMBDA_MAX).exp()
                cost = torch.empty((C - 1, C - 1), device=device, dtype=torch.float32)
                for ip in range(1, C):
                    li = lam[:, ip]
                    si = S_b[:, ip]
                    for ig in range(1, C):
                        cost[ip - 1, ig - 1] = (li - y_b[:, ig] * si).sum()
                if matching == "hungarian":
                    _, col = linear_sum_assignment(_sanitised_cost(cost, "loss_set_poisson"))
                    perm = torch.as_tensor(col, device=device, dtype=torch.long)
                elif matching == "min":
                    assign = smallK_match(cost)
                    perm_inv = [-1] * (C - 1)
                    for gt_idx, pred_idx in enumerate(assign):
                        if 0 <= pred_idx < (C - 1):
                            perm_inv[pred_idx] = gt_idx
                    for pred_idx in range(C - 1):
                        if perm_inv[pred_idx] < 0:
                            perm_inv[pred_idx] = pred_idx
                    perm = torch.as_tensor(perm_inv, device=device, dtype=torch.long)
                else:
                    raise ValueError("matching must be one of: 'none', 'min', 'hungarian'")
                y_b = torch.cat([y_b[:, :1], y_b[:, 1:][:, perm]], dim=1)

            nll = F.poisson_nll_loss(S_b, y_b, log_input=True, full=full, reduction="sum")
            nll_sum = nll_sum + nll
            total += V * C

            if lambda_dice > 0.0 and C > 1:
                if dice_kind == "count":
                    lam = S_b.exp()                         # (V, C) predicted rates
                    for c in range(1, C):
                        lc = lam[:, c]; yc = y_b[:, c]
                        num = 2.0 * (lc * yc).sum()
                        den = (lc * lc).sum() + (yc * yc).sum() + eps
                        dice_sum = dice_sum + (1.0 - num / den)
                else:  
                    p_occ = 1.0 - torch.exp(-S_b.exp())
                    y_occ = (y_b > 0).float()
                    for c in range(1, C):
                        num = 2.0 * (p_occ[:, c] * y_occ[:, c]).sum()
                        den = p_occ[:, c].sum() + y_occ[:, c].sum() + eps
                        dice_sum = dice_sum + (1.0 - num / den)

        nll = nll_sum / max(total, 1.0)
        dice = dice_sum / max(B, 1)
        loss = lambda_count * nll + lambda_dice * dice
        return {"loss": loss, "loss_poisson": nll, "loss_dice": dice}

    return _loss


def loss_seg_plus_vertex(lambda_dice: float = 1.0,
                         eps: float = 1e-10,
                         poisson: bool = True,
                         matching: str = "hungarian",
                         symmetric: bool = True,
                         lambda_reg: float = 1.0,
                         w_vtx: float = 1.0,
                         w_dir: float = 1.0,
                         w_ene: float = 1.0,
                         w_pid: float = 1.0,
                         vertex_norm: float = 4000.0,
                         energy_scale_mev: float = 1000.0,
                         pid_classes=(11, 13),
                         seg_kind: str = "ce",
                         lambda_count: float = 1.0,
                         full: bool = True,
                         dice_kind: str = "occupancy"):

    if seg_kind == "poisson":
        seg_loss = loss_set_poisson(lambda_count=lambda_count, lambda_dice=lambda_dice,
                                    eps=eps, matching=matching, full=full, dice_kind=dice_kind)
    else:
        seg_loss = loss_set_ce_dice(lambda_dice=lambda_dice, eps=eps, poisson=poisson,
                                    matching=matching, symmetric=symmetric)
    pid_map = {int(p): i for i, p in enumerate(pid_classes)}

    def _t(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    def _loss(batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        seg = seg_loss(batch=batch, out=out)
        device = seg["loss"].device
        preds = out.get("vtx_ene_pred_list", None)
        pid_logits = out.get("pid_logits_list", None)

        se_vtx = torch.zeros((), device=device)
        se_dir = torch.zeros((), device=device)
        se_ene = torch.zeros((), device=device)
        ce_pid = torch.zeros((), device=device)
        n_valid = 0.0
        n_pid = 0.0

        if preds is not None:
            for b, pred_b in enumerate(preds):
                if pred_b.numel() == 0:
                    continue
                meta_b = batch["meta"][b]
                vertex = meta_b.get("vertex", None)
                pdir = meta_b.get("parent_dir", None)
                if vertex is None or pdir is None:
                    continue
                pene = meta_b.get("parent_energy_mev", None)
                ppdg = meta_b.get("parent_pdg", None)
                pvalid = meta_b.get("parent_valid", None)
                N = pred_b.shape[0]
                tgt = pred_b.new_zeros((N, pred_b.shape[1]))
                mask = pred_b.new_zeros((N,))
                pid_tgt = torch.full((N,), -1, dtype=torch.long, device=device)
                v = _t(vertex, device).view(-1)[:3] / float(vertex_norm)
                pdir_t = _t(pdir, device).view(-1, 3)
                for p in range(min(N, pdir_t.shape[0])):
                    if pvalid is not None and float(pvalid[p]) == 0.0:
                        continue
                    tgt[p, 0:3] = v
                    tgt[p, 3:6] = pdir_t[p]
                    tgt[p, 6] = (float(pene[p]) / float(energy_scale_mev)) if pene is not None else 0.0
                    mask[p] = 1.0
                    if ppdg is not None:
                        pid_tgt[p] = pid_map.get(abs(int(ppdg[p])), -1)
                if float(mask.sum().item()) == 0.0:
                    continue
                diff = pred_b - tgt
                m = mask.unsqueeze(1)
                se_vtx = se_vtx + ((diff[:, 0:3] ** 2) * m).sum()
                se_dir = se_dir + ((diff[:, 3:6] ** 2) * m).sum()
                se_ene = se_ene + ((diff[:, 6:7] ** 2) * m).sum()
                n_valid += float(mask.sum().item())

                if pid_logits is not None:
                    ce_pid = ce_pid + torch.nn.functional.cross_entropy(
                        pid_logits[b], pid_tgt, ignore_index=-1, reduction="sum")
                    n_pid += float((pid_tgt >= 0).sum().item())

        denom = max(n_valid, 1.0)
        loss_vtx = se_vtx / (denom * 3.0)
        loss_dir = se_dir / (denom * 3.0)
        loss_ene = se_ene / denom
        loss_pid = ce_pid / max(n_pid, 1.0)
        loss_reg = w_vtx * loss_vtx + w_dir * loss_dir + w_ene * loss_ene + w_pid * loss_pid
        total = seg["loss"] + lambda_reg * loss_reg

        seg_main = seg.get("loss_ce", seg.get("loss_poisson"))
        return {"loss": total, "loss_ce": seg_main, "loss_dice": seg["loss_dice"],
                "loss_reg": loss_reg, "loss_vtx": loss_vtx, "loss_dir": loss_dir,
                "loss_ene": loss_ene, "loss_pid": loss_pid}

    return _loss
