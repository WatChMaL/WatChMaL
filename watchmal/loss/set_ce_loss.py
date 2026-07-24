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
import torch
from scipy.optimize import linear_sum_assignment


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
                    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
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