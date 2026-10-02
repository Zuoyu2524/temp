from __future__ import annotations

"""Multi-GPU query-parallel multi-scale UOT kNN retrieval.

This script runs different queries concurrently on different GPUs.

- Each worker process is pinned to one GPU.
- Each worker pulls one query at a time and runs the level-wise torch solver.
- The main process receives progress events and prints human-readable progress.
"""

import csv
import multiprocessing as mp
import os
import queue
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import h5py
import torch


EPS = 1e-32


@dataclass
class CandidateState:
    sample_index: int
    sample_id: int
    lower_bound_dual: float = 0.0
    lower_bound_primal: float = float("inf")
    upper_bound_dual: float = 0.0
    upper_bound_primal: float = float("inf")
    exact_dual: float = 0.0
    exact_primal: float = float("inf")
    exact_plan: Optional[np.ndarray] = None

    @property
    def lower_bound(self) -> float:
        return max(self.lower_bound_dual, self.exact_dual)
    
    @property
    def upper_bound(self) -> float:
        return min(self.upper_bound_primal, self.exact_primal)

    @property
    def delta_opt(self) -> float:
        lower_gap = max(self.lower_bound_primal - self.lower_bound_dual, 0.0)
        upper_gap = max(self.upper_bound_primal - self.upper_bound_dual, 0.0)
        return lower_gap + upper_gap

    @property
    def delta_block(self) -> float:
        return max(self.upper_bound_dual - self.lower_bound_primal, 0.0)


@dataclass
class QuantizedDataset:
    sample_ids: np.ndarray
    data_coords: np.ndarray
    measure: np.ndarray
    num_clusters_list: List[int]
    cluster_results: Dict[int, Dict[str, np.ndarray]]


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype_name must be 'float32' or 'float64'.")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    return device


def resolve_query_index(
    sample_ids: np.ndarray,
    query_index: Optional[int],
    query_id: Optional[int],
) -> int:
    if query_index is not None and query_id is not None:
        raise ValueError("Use either query_index or query_id, not both.")
    if query_id is not None:
        matches = np.where(sample_ids == int(query_id))[0]
        if len(matches) == 0:
            raise ValueError(f"query_id {query_id} not found in dataset.")
        return int(matches[0])
    if query_index is None:
        return 0
    if query_index < 0 or query_index >= len(sample_ids):
        raise ValueError("query_index is out of range.")
    return int(query_index)


def resolve_query_indices(
    sample_ids: np.ndarray,
    query_indices: Optional[Sequence[int]],
    query_ids: Optional[Sequence[int]],
) -> List[int]:
    if query_indices is not None and query_ids is not None:
        raise ValueError("Use query_indices or query_ids, not both.")
    if query_ids is not None:
        resolved = [resolve_query_index(sample_ids, None, int(query_id)) for query_id in query_ids]
    elif query_indices is not None:
        resolved = [resolve_query_index(sample_ids, int(query_index), None) for query_index in query_indices]
    else:
        resolved = [0]
    return sorted(set(resolved))


def resolve_levels(
    available_levels: Sequence[int],
    requested_levels: Optional[Sequence[int]],
) -> List[int]:
    if requested_levels is None:
        return list(available_levels)
    available_set = set(int(v) for v in available_levels)
    missing = [int(v) for v in requested_levels if int(v) not in available_set]
    if missing:
        raise ValueError(f"Requested block numbers are missing from H5: {missing}")
    return sorted(int(v) for v in requested_levels)


def read_quantized_point_cloud_batch(input_path: str) -> QuantizedDataset:
    with h5py.File(input_path, "r") as h5f:
        sample_ids = h5f["sample_ids"][:].astype(np.int64)
        data_coords = h5f["data_coords"][:].astype(np.float32)
        measure = h5f["measure"][:].astype(np.float32)
        num_clusters_list = sorted(
            int(v) for v in np.asarray(h5f.attrs["num_clusters_list"], dtype=np.int32)
        )
        cluster_results: Dict[int, Dict[str, np.ndarray]] = {}
        for block_number in num_clusters_list:
            cluster_results[block_number] = {
                "clusters": h5f[f"clusters_{block_number}"][:].astype(np.int32),
                "cluster_weights": h5f[f"cluster_weights_{block_number}"][:].astype(np.float32),
                "cluster_centers": h5f[f"cluster_centers_{block_number}"][:].astype(np.float32),
                "cluster_radii": h5f[f"cluster_radii_{block_number}"][:].astype(np.float32),
                "weighted_means": h5f[f"weighted_means_{block_number}"][:].astype(np.float32),
                "cluster_variances": h5f[f"cluster_variances_{block_number}"][:].astype(np.float32),
            }

    return QuantizedDataset(
        sample_ids=sample_ids,
        data_coords=data_coords,
        measure=measure,
        num_clusters_list=num_clusters_list,
        cluster_results=cluster_results,
    )


def build_candidate_states(
    dataset: QuantizedDataset,
    query_index: int,
    candidate_limit: Optional[List[int]],
) -> List[CandidateState]:
    states: List[CandidateState] = []
    candidate_indices = set(candidate_limit) if candidate_limit is not None else None

    for sample_index, sample_id in enumerate(dataset.sample_ids):
        if sample_index == query_index:
            continue
        if candidate_indices is not None and sample_index not in candidate_indices:
            continue
        states.append(CandidateState(sample_index=int(sample_index), sample_id=int(sample_id)))
    return states


def get_level(dataset: QuantizedDataset, block_number: int) -> Dict[str, np.ndarray]:
    return dataset.cluster_results[int(block_number)]


def reset_coarse_level_state(candidates: Sequence[CandidateState]) -> None:
    for candidate in candidates:
        candidate.lower_bound_dual = 0.0
        candidate.lower_bound_primal = float("inf")
        candidate.upper_bound_dual = 0.0
        candidate.upper_bound_primal = float("inf")


def compute_candidate_batch_size(
    problem_size: int,
    reference_problem_size: int,
    base_batch_size: int,
    min_batch_size: int,
) -> int:
    if base_batch_size <= 0:
        raise ValueError("base_batch_size must be positive.")
    if min_batch_size <= 0:
        raise ValueError("min_batch_size must be positive.")
    reference = max(int(reference_problem_size), 1)
    current = max(int(problem_size), 1)
    scaled = int(base_batch_size * (reference / current) ** 2)
    return max(min_batch_size, scaled)


def torch_from_numpy(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def batched_pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm = torch.sum(y * y, dim=-1).unsqueeze(1)
    dist2 = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return torch.clamp(dist2, min=0.0)


def batched_pairwise_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(batched_pairwise_sq_dists(x, y))


def kl_div_batched(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    a_safe = torch.clamp(a, min=eps)
    b_safe = torch.clamp(b, min=eps)
    mask = a > 0
    return torch.sum(
        torch.where(
            mask,
            a * torch.log(a_safe / b_safe) - a + b,
            b,
        ),
        dim=1,
    )


def mm_unbalanced_kl_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    lambda_a: float,
    lambda_b: float,
    num_iter: int,
    init_plan: Optional[torch.Tensor] = None,
    stop_thr: float = 1e-8,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lambda_a <= 0 or lambda_b <= 0:
        raise ValueError("lambda_a and lambda_b must be strictly positive.")

    Lambda = lambda_a + lambda_b
    phi = lambda_a / Lambda
    psi = lambda_b / Lambda

    if init_plan is not None and tuple(init_plan.shape) == tuple(cost.shape):
        G = init_plan
    else:
        G = a.unsqueeze(2) * b.unsqueeze(1)

    K = (a.unsqueeze(2) ** phi) * (b.unsqueeze(1) ** psi) * torch.exp(-cost / Lambda)

    for iter_idx in range(num_iter):
        G_prev = G
        row_sum = G.sum(dim=2, keepdim=True)
        col_sum = G.sum(dim=1, keepdim=True)
        Gd = (row_sum**phi) * (col_sum**psi) + eps
        G = K * G / Gd

        if iter_idx % 10 == 0:
            err = torch.sqrt(torch.sum((G - G_prev) ** 2, dim=(1, 2)))
            if bool(torch.all(err < stop_thr)):
                break

    linear_cost = torch.sum(G * cost, dim=(1, 2))
    row_marginal = G.sum(dim=2)
    col_marginal = G.sum(dim=1)
    total_cost = linear_cost + lambda_a * kl_div_batched(row_marginal, a) + lambda_b * kl_div_batched(col_marginal, b)
    return total_cost, G


def dual_repair_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    plan: torch.Tensor,
    cost: torch.Tensor,
    lambda_a: float,
    lambda_b: float,
    feasibility_tol: float = 1e-3,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    row_marginal = plan.sum(dim=2)
    col_marginal = plan.sum(dim=1)

    f = -lambda_a * torch.log(torch.clamp(row_marginal, min=eps) / torch.clamp(a, min=eps))
    g = -lambda_b * torch.log(torch.clamp(col_marginal, min=eps) / torch.clamp(b, min=eps))

    violation = f.unsqueeze(2) + g.unsqueeze(1) - cost
    roi = violation.amax(dim=2)
    kai = violation.amax(dim=1)
    f_new = f - roi / 2.0
    g_new = g - kai / 2.0

    repaired_violation = f_new.unsqueeze(2) + g_new.unsqueeze(1) - cost
    max_violation = repaired_violation.amax(dim=(1, 2))
    valid_mask = max_violation <= feasibility_tol

    Lambda = lambda_a + lambda_b
    phi = lambda_a / Lambda
    psi = lambda_b / Lambda
    X = torch.sum(a * torch.exp(-f_new / lambda_a), dim=1)
    Y = torch.sum(b * torch.exp(-g_new / lambda_b), dim=1)
    geometric_mean = (X**psi) * (Y**phi)
    dual_value = lambda_a * (torch.sum(a, dim=1) - geometric_mean) + lambda_b * (torch.sum(b, dim=1) - geometric_mean)
    return dual_value, valid_mask


def compute_mm_dual_bounds_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    lambda_a: float,
    lambda_b: float,
    iters: int,
    init_plan: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    primal_ub, plan = mm_unbalanced_kl_batched(
        a,
        b,
        cost,
        lambda_a,
        lambda_b,
        iters,
        init_plan=init_plan,
    )
    dual_lb, valid_mask = dual_repair_batched(a, b, plan, cost, lambda_a, lambda_b)
    return dual_lb, primal_ub, plan, valid_mask


def batched_bounding_ball_cost(
    center_a: torch.Tensor,
    radii_a: torch.Tensor,
    center_b: torch.Tensor,
    radii_b: torch.Tensor,
) -> torch.Tensor:
    dist = batched_pairwise_dists(center_a, center_b)
    slack = torch.clamp(dist - (radii_a.unsqueeze(2) + radii_b.unsqueeze(1)), min=0.0)
    return slack * slack


def batched_product_lift_cost(
    mu_a: torch.Tensor,
    v_a: torch.Tensor,
    mu_b: torch.Tensor,
    v_b: torch.Tensor,
) -> torch.Tensor:
    return batched_pairwise_sq_dists(mu_a, mu_b) + v_a.unsqueeze(2) + v_b.unsqueeze(1)


def batched_exact_cost(coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
    return batched_pairwise_sq_dists(coords_a, coords_b)
    

def run_candidate_coarse_bounds_batched(
    dataset: QuantizedDataset,
    query_index: int,
    candidates: Sequence[CandidateState],
    block_number: int,
    lambda_a: float,
    lambda_b: float,
    coarse_iters: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_candidate_batch_size: int,
    min_candidate_batch_size: int,
    reference_block_number: int,
    sweeps: int,
    ready_ratio_threshold: float,
) -> dict:
    if not candidates:
        return {
            "block_number": int(block_number),
            "num_candidates": 0,
            "sweeps_used": 0,
            "ready_ratio": 1.0,
            "candidate_batch_size": 0,
            "remaining_active": 0,
        }

    if coarse_iters <= 0:
        raise ValueError("coarse_iters must be positive.")

    if sweeps <= 0:
        raise ValueError("sweeps must be positive.")

    if not 0.0 <= ready_ratio_threshold <= 1.0:
        raise ValueError("ready_ratio_threshold must be between 0 and 1.")

    level = get_level(dataset, block_number)

    query_weights_np = np.asarray(level["cluster_weights"][query_index], dtype=np.float32)
    query_centers_np = np.asarray(level["cluster_centers"][query_index], dtype=np.float32)
    query_radii_np = np.asarray(level["cluster_radii"][query_index], dtype=np.float32)
    query_means_np = np.asarray(level["weighted_means"][query_index], dtype=np.float32)
    query_vars_np = np.asarray(level["cluster_variances"][query_index], dtype=np.float32)

    batch_size = compute_candidate_batch_size(
        problem_size=block_number,
        reference_problem_size=reference_block_number,
        base_batch_size=base_candidate_batch_size,
        min_batch_size=min_candidate_batch_size,
    )
    batch_size = min(batch_size, len(candidates))

    query_weights = torch_from_numpy(query_weights_np, device, dtype).unsqueeze(0)
    query_centers = torch_from_numpy(query_centers_np, device, dtype).unsqueeze(0)
    query_radii = torch_from_numpy(query_radii_np, device, dtype).unsqueeze(0)
    query_means = torch_from_numpy(query_means_np, device, dtype).unsqueeze(0)
    query_vars = torch_from_numpy(query_vars_np, device, dtype).unsqueeze(0)

    candidate_indices = np.asarray([candidate.sample_index for candidate in candidates], dtype=np.int32)

    num_candidates = len(candidates)

    batches = []
    for start in range(0, num_candidates, batch_size):
        end = min(start + batch_size, num_candidates)
        batches.append(np.arange(start, end, dtype=np.int32))

    total_ready = 0
    max_sweeps_used = 0

    with torch.no_grad():
        for original_indices in batches:
            sample_indices = candidate_indices[original_indices]
            batch_len = len(original_indices)

            cand_weights = torch_from_numpy(
                np.asarray(level["cluster_weights"][sample_indices], dtype=np.float32),
                device, dtype,
            )
            cand_centers = torch_from_numpy(
                np.asarray(level["cluster_centers"][sample_indices], dtype=np.float32),
                device, dtype,
            )
            cand_radii = torch_from_numpy(
                np.asarray(level["cluster_radii"][sample_indices], dtype=np.float32),
                device, dtype,
            )
            cand_means = torch_from_numpy(
                np.asarray(level["weighted_means"][sample_indices], dtype=np.float32),
                device, dtype,
            )
            cand_vars = torch_from_numpy(
                np.asarray(level["cluster_variances"][sample_indices], dtype=np.float32),
                device, dtype,
            )

            query_weights_batch = query_weights.expand(batch_len, -1)
            query_centers_batch = query_centers.expand(batch_len, -1, -1)
            query_radii_batch = query_radii.expand(batch_len, -1)
            query_means_batch = query_means.expand(batch_len, -1, -1)
            query_vars_batch = query_vars.expand(batch_len, -1)

            lower_cost = batched_bounding_ball_cost(
                query_centers_batch,
                query_radii_batch,
                cand_centers,
                cand_radii,
            )
            upper_cost = batched_product_lift_cost(
                query_means_batch,
                query_vars_batch,
                cand_means,
                cand_vars,
            )

            lower_dual_gpu = torch_from_numpy(
                np.asarray(
                    [float(candidates[int(i)].lower_bound_dual) for i in original_indices],
                    dtype=np.float32,
                ),
                device, dtype,
            )
            lower_primal_gpu = torch_from_numpy(
                np.asarray(
                    [float(candidates[int(i)].lower_bound_primal) for i in original_indices],
                    dtype=np.float32,
                ),
                device, dtype,
            )
            upper_dual_gpu = torch_from_numpy(
                np.asarray(
                    [float(candidates[int(i)].upper_bound_dual) for i in original_indices],
                    dtype=np.float32,
                ),
                device, dtype,
            )
            upper_primal_gpu = torch_from_numpy(
                np.asarray(
                    [float(candidates[int(i)].upper_bound_primal) for i in original_indices],
                    dtype=np.float32,
                ),
                device, dtype,
            )

            lower_plan = None
            upper_plan = None
            ready_mask = torch.zeros(batch_len, dtype=torch.bool, device=device)
            batch_sweeps_used = 0

            for sweep_idx in range(sweeps):
                batch_sweeps_used = sweep_idx + 1

                (
                    lower_dual,
                    lower_primal,
                    lower_plan,
                    lower_valid,
                ) = compute_mm_dual_bounds_batched(
                    query_weights_batch,
                    cand_weights,
                    lower_cost,
                    lambda_a,
                    lambda_b,
                    coarse_iters,
                    init_plan=lower_plan,
                )

                (
                    upper_dual,
                    upper_primal,
                    upper_plan,
                    upper_valid,
                ) = compute_mm_dual_bounds_batched(
                    query_weights_batch,
                    cand_weights,
                    upper_cost,
                    lambda_a,
                    lambda_b,
                    coarse_iters,
                    init_plan=upper_plan,
                )

                lower_dual = lower_dual.to(dtype)
                lower_primal = lower_primal.to(dtype)
                upper_dual = upper_dual.to(dtype)
                upper_primal = upper_primal.to(dtype)
                lower_valid = lower_valid.to(torch.bool)
                upper_valid = upper_valid.to(torch.bool)

                lower_dual_gpu = torch.where(
                    lower_valid,
                    torch.maximum(lower_dual_gpu, lower_dual),
                    lower_dual_gpu,
                )
                lower_primal_gpu = torch.where(
                    lower_valid,
                    torch.minimum(lower_primal_gpu, lower_primal),
                    lower_primal_gpu,
                )
                upper_dual_gpu = torch.where(
                    upper_valid,
                    torch.maximum(upper_dual_gpu, upper_dual),
                    upper_dual_gpu,
                )
                upper_primal_gpu = torch.where(
                    upper_valid,
                    torch.minimum(upper_primal_gpu, upper_primal),
                    upper_primal_gpu,
                )

                lower_gap = torch.clamp(lower_primal_gpu - lower_dual_gpu, min=0.0)
                upper_gap = torch.clamp(upper_primal_gpu - upper_dual_gpu, min=0.0)
                delta_opt = lower_gap + upper_gap
                delta_block = torch.clamp(upper_dual_gpu - lower_primal_gpu, min=0.0)

                ready_mask = delta_opt < delta_block

                batch_ready_ratio = float(ready_mask.float().mean().item())
                if batch_ready_ratio >= ready_ratio_threshold:
                    break

            max_sweeps_used = max(max_sweeps_used, batch_sweeps_used)
            total_ready += int(ready_mask.sum().item())

            lbd_np = lower_dual_gpu.detach().cpu().numpy()
            lbp_np = lower_primal_gpu.detach().cpu().numpy()
            ubd_np = upper_dual_gpu.detach().cpu().numpy()
            ubp_np = upper_primal_gpu.detach().cpu().numpy()

            for k in range(batch_len):
                cand = candidates[int(original_indices[k])]
                cand.lower_bound_dual = float(lbd_np[k])
                cand.lower_bound_primal = float(lbp_np[k])
                cand.upper_bound_dual = float(ubd_np[k])
                cand.upper_bound_primal = float(ubp_np[k])

            del cand_weights, cand_centers, cand_radii, cand_means, cand_vars
            del lower_cost, upper_cost
            del lower_dual_gpu, lower_primal_gpu, upper_dual_gpu, upper_primal_gpu
            del lower_plan, upper_plan
            del ready_mask

    overall_ready_ratio = total_ready / num_candidates

    return {
        "block_number": int(block_number),
        "num_candidates": num_candidates,
        "sweeps_used": int(max_sweeps_used),
        "ready_ratio": float(overall_ready_ratio),
        "candidate_batch_size": int(batch_size)
    }

            

def run_candidate_exact_bounds_batched(
    dataset: QuantizedDataset,
    query_index: int,
    candidates: Sequence[CandidateState],
    lambda_a: float,
    lambda_b: float,
    exact_iters: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_exact_batch_size: int,
    min_exact_batch_size: int,
    reference_problem_size: int,
) -> int:
    if not candidates:
        return 0

    query_weights_np = np.asarray(dataset.measure[query_index], dtype=np.float32)
    query_coords_np = np.asarray(dataset.data_coords[query_index], dtype=np.float32)
    point_count = int(query_weights_np.shape[0])

    batch_size = compute_candidate_batch_size(
        problem_size=point_count,
        reference_problem_size=reference_problem_size,
        base_batch_size=base_exact_batch_size,
        min_batch_size=min_exact_batch_size,
    )
    batch_size = min(batch_size, len(candidates))

    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch_candidates = list(candidates[start : start + batch_size])
            candidate_indices = np.asarray([cand.sample_index for cand in batch_candidates], dtype=np.int64)
            batch_len = len(batch_candidates)

            cand_weights_np = np.asarray(dataset.measure[candidate_indices], dtype=np.float32)
            cand_coords_np = np.asarray(dataset.data_coords[candidate_indices], dtype=np.float32)
            query_weights_batch_np = np.repeat(query_weights_np[None, :], batch_len, axis=0)

            query_weights = torch_from_numpy(query_weights_batch_np, device, dtype)
            cand_weights = torch_from_numpy(cand_weights_np, device, dtype)
            query_coords = torch_from_numpy(np.repeat(query_coords_np[None, :, :], batch_len, axis=0), device, dtype)
            cand_coords = torch_from_numpy(cand_coords_np, device, dtype)

            cost = batched_exact_cost(query_coords, cand_coords)
            exact_dual, exact_primal, exact_plan, exact_valid = compute_mm_dual_bounds_batched(
                query_weights,
                cand_weights,
                cost,
                lambda_a,
                lambda_b,
                exact_iters,
                init_plan=None,
            )

            exact_dual_np = exact_dual.detach().cpu().numpy()
            exact_primal_np = exact_primal.detach().cpu().numpy()
            exact_plan_np = exact_plan.detach().cpu().numpy()
            exact_valid_np = exact_valid.detach().cpu().numpy()

            for local_idx, candidate in enumerate(batch_candidates):
                if bool(exact_valid_np[local_idx]):
                    candidate.exact_dual = max(candidate.exact_dual, float(exact_dual_np[local_idx]))
                exact_primal_value = float(exact_primal_np[local_idx])
                if exact_primal_value < candidate.exact_primal:
                    candidate.exact_primal = exact_primal_value
                    candidate.exact_plan = exact_plan_np[local_idx]

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return batch_size


def filter_topk_candidates(
    candidates: Sequence[CandidateState],
    keep_k: int,
) -> List[CandidateState]:
    if keep_k <= 0:
        raise ValueError("keep_k must be positive.")
    ranked = sorted(candidates, key=lambda cand: (cand.upper_bound, cand.sample_id))
    tau = ranked[keep_k - 1].upper_bound
    remaining = []
    for cand in candidates:
        if cand.lower_bound <= tau:
            remaining.append(cand)
    return remaining


def result_to_csv_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(result["remaining_candidates"], start=1):
        rows.append(
            {
                "query_id": result["query_id"],
                "query_index": result["query_index"],
                "final_rank": int(rank),
                "candidate_id": candidate.sample_id,
                "candidate_index": candidate.sample_index,
                "lower_bound": f"{candidate.lower_bound:.12f}",
                "upper_bound": f"{candidate.upper_bound:.12f}",
                "lower_bound_dual": f"{candidate.lower_bound_dual:.12f}",
                "lower_bound_primal": f"{candidate.lower_bound_primal:.12f}",
                "upper_bound_dual": f"{candidate.upper_bound_dual:.12f}",
                "upper_bound_primal": f"{candidate.upper_bound_primal:.12f}",
                "exact_dual": f"{candidate.exact_dual:.12f}",
                "exact_primal": f"{candidate.exact_primal:.12f}",
                "delta_block": f"{candidate.delta_block:.12f}",
                "delta_opt": f"{candidate.delta_opt:.12f}",
            }
        )
    return rows


def write_results_csv(output_csv: str, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "query_id",
        "query_index",
        "final_rank",
        "candidate_id",
        "candidate_index",
        "lower_bound",
        "upper_bound",
        "lower_bound_dual",
        "lower_bound_primal",
        "upper_bound_dual",
        "upper_bound_primal",
        "exact_dual",
        "exact_primal",
        "delta_block",
        "delta_opt",
    ]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_progress_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def write_query_progress_log(
    output_dir: str,
    summaries: Sequence[Dict[str, object]],
    total_elapsed: float,
) -> None:
    log_path = Path(output_dir) / "multi_gpu_query_progress.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"num_queries={len(summaries)}\n")
        f.write(f"total_elapsed={total_elapsed:.2f}s\n")
        for summary in summaries:
            top_ids = ",".join(str(v) for v in summary["top_candidate_ids"])
            f.write(
                f"query_id={summary['query_id']} "
                f"query_index={summary['query_index']} "
                f"gpu_id={summary['gpu_id']} "
                f"elapsed={summary['elapsed_seconds']:.2f}s "
                f"num_remaining={summary['num_remaining']} "
                f"exact_batch_size={summary['exact_batch_size']} "
                f"output_csv={summary['output_csv']} "
                f"top_candidate_ids={top_ids}\n"
            )
            for level in summary["level_history"]:
                f.write(
                    f"  level_pos={level['level_pos']} "
                    f"block_number={level['block_number']} "
                    f"sweeps_used={level['sweeps_used']} "
                    f"ready_ratio={level['ready_ratio']:.4f} "
                    f"num_candidates_before_filter={level['num_candidates_before_filter']} "
                    f"num_candidates_after_filter={level['num_candidates_after_filter']} "
                    f"candidate_batch_size={level['candidate_batch_size']} "
                    f"level_elapsed={level['level_elapsed_seconds']:.2f}s\n"
                )


def run_single_query_with_progress(
    dataset: QuantizedDataset,
    query_index: int,
    *,
    block_numbers: Sequence[int],           
    exact_block_number: int,               
    keep_k: int,
    lambda_a: float,
    lambda_b: float,
    coarse_iters: int,
    final_exact_iters: int,
    ready_ratio_threshold: float,
    max_level_sweeps: int,
    candidate_limit: Optional[int],
    device_name: str,
    dtype_name: str,
    base_candidate_batch_size: int,
    min_candidate_batch_size: int,
    base_exact_batch_size: int,
    min_exact_batch_size: int,
    gpu_id: int,
    output_dir: str,
    progress_queue: mp.Queue,
) -> Dict[str, object]:
    import torch

    device = resolve_device(device_name)
    dtype = get_torch_dtype(dtype_name)
    query_id = int(dataset.sample_ids[query_index])
    start_time = time.perf_counter()

    progress_queue.put(
        {
            "type": "query_started",
            "query_id": query_id,
            "query_index": int(query_index),
            "gpu_id": int(gpu_id),
        }
    )

    candidates = build_candidate_states(dataset, query_index, candidate_limit)
    if not candidates:
        raise ValueError("No non-query candidates are available.")

    level_history: List[Dict[str, object]] = []
    current_candidates: List[CandidateState] = candidates
    reference_block_number = int(block_numbers[0])


    levels: List[Tuple[str, int]] = (
        [("coarse", int(bn)) for bn in block_numbers]
        + [("exact", int(exact_block_number))]
    )
    num_levels = len(levels)

    exact_batch_size = 0 

    for level_pos, (level_kind, block_number) in enumerate(levels):
        level_start = time.perf_counter()
        num_before = len(current_candidates)

        if level_kind == "coarse":
            # -------- coarse level --------
            reset_coarse_level_state(current_candidates)
            level_summary = run_candidate_coarse_bounds_batched(
                dataset,
                query_index,
                current_candidates,
                block_number,
                lambda_a,
                lambda_b,
                coarse_iters,
                ready_ratio_threshold=ready_ratio_threshold,
                sweeps=max_level_sweeps,
                device=device,
                dtype=dtype,
                base_candidate_batch_size=base_candidate_batch_size,
                min_candidate_batch_size=min_candidate_batch_size,
                reference_block_number=reference_block_number
            )
        else:
            exact_batch_size = run_candidate_exact_bounds_batched(
                dataset,
                query_index,
                current_candidates,
                lambda_a,
                lambda_b,
                final_exact_iters,
                device=device,
                dtype=dtype,
                base_exact_batch_size=base_exact_batch_size,
                min_exact_batch_size=min_exact_batch_size,
                reference_problem_size=reference_block_number,
            )

            level_summary = {
                "block_number": int(block_number),
                "num_candidates": int(num_before),
                "sweeps_used": 0,
                "ready_ratio": 1.0,
                "candidate_batch_size": int(exact_batch_size),
            }

        level_summary["level_pos"] = int(level_pos)
        level_summary["level_kind"] = level_kind
        level_summary["block_number"] = int(block_number)
        level_summary["num_candidates_before_filter"] = int(num_before)

        current_candidates = filter_topk_candidates(current_candidates, keep_k)
        level_summary["num_candidates_after_filter"] = int(len(current_candidates))
        level_summary["level_elapsed_seconds"] = float(time.perf_counter() - level_start)
        level_history.append(level_summary)

        progress_queue.put(
            {
                "type": "level_done",
                "query_id": query_id,
                "query_index": int(query_index),
                "gpu_id": int(gpu_id),
                "level_pos": int(level_pos),
                "level_kind": level_kind,
                "block_number": int(block_number),
                "num_candidates_before_filter": int(level_summary["num_candidates_before_filter"]),
                "num_candidates_after_filter": int(level_summary["num_candidates_after_filter"]),
                "ready_ratio": float(level_summary.get("ready_ratio", 1.0)),
                "sweeps_used": int(level_summary.get("sweeps_used", 0)),
                "level_elapsed_seconds": float(level_summary["level_elapsed_seconds"]),
            }
        )

    ranked_candidates = sorted(
        current_candidates,
        key=lambda cand: (cand.upper_bound, cand.sample_id),
    )

    result = {
        "query_id": query_id,
        "query_index": int(query_index),
        "remaining_candidates": ranked_candidates,
        "level_history": level_history,
        "exact_batch_size": int(exact_batch_size),
    }
    rows = result_to_csv_rows(result)
    output_csv = os.path.join(output_dir, f"query_{query_id}_multi_gpu.csv")
    write_results_csv(output_csv, rows)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "query_id": query_id,
        "query_index": int(query_index),
        "gpu_id": int(gpu_id),
        "elapsed_seconds": float(elapsed_seconds),
        "num_remaining": len(ranked_candidates),
        "top_candidate_ids": [cand.sample_id for cand in ranked_candidates[: min(keep_k, 10)]],
        "level_history": level_history,
        "exact_batch_size": int(exact_batch_size),
        "output_csv": output_csv,
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    return summary


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    progress_queue: mp.Queue,
    processed_h5: str,
    output_dir: str,
    block_numbers_config: Optional[Sequence[int]],
    exact_block_number: int,
    keep_k: int,
    lambda_a: float,
    lambda_b: float,
    coarse_iters: int,
    final_exact_iters: int,
    ready_ratio_threshold: float,
    max_level_sweeps: int,
    candidate_limit: Optional[int],
    torch_dtype_name: str,
    base_candidate_batch_size: int,
    min_candidate_batch_size: int,
    base_exact_batch_size: int,
    min_exact_batch_size: int,
) -> None:
    try:
        import torch

        device_name = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        device = resolve_device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        dataset = read_quantized_point_cloud_batch(processed_h5)
        block_numbers = resolve_levels(dataset.num_clusters_list, block_numbers_config)

        while True:
            task = task_queue.get()
            if task is None:
                break
            query_index = int(task["query_index"])
            try:
                summary = run_single_query_with_progress(
                    dataset,
                    query_index,
                    block_numbers=block_numbers,
                    exact_block_number=exact_block_number,
                    keep_k=keep_k,
                    lambda_a=lambda_a,
                    lambda_b=lambda_b,
                    coarse_iters=coarse_iters,
                    final_exact_iters=final_exact_iters,
                    ready_ratio_threshold=ready_ratio_threshold,
                    max_level_sweeps=max_level_sweeps,
                    candidate_limit=candidate_limit,
                    device_name=device_name,
                    dtype_name=torch_dtype_name,
                    base_candidate_batch_size=base_candidate_batch_size,
                    min_candidate_batch_size=min_candidate_batch_size,
                    base_exact_batch_size=base_exact_batch_size,
                    min_exact_batch_size=min_exact_batch_size,
                    gpu_id=gpu_id,
                    output_dir=output_dir,
                    progress_queue=progress_queue,
                )
                progress_queue.put({"type": "summary", "summary": summary})
                progress_queue.put(
                    {
                        "type": "query_done",
                        "query_id": summary["query_id"],
                        "query_index": summary["query_index"],
                        "gpu_id": int(gpu_id),
                        "elapsed_seconds": float(summary["elapsed_seconds"]),
                        "num_remaining": int(summary["num_remaining"]),
                        "output_csv": summary["output_csv"],
                    }
                )
            except Exception as exc:
                progress_queue.put(
                    {
                        "type": "query_error",
                        "query_index": query_index,
                        "gpu_id": int(gpu_id),
                        "error": repr(exc),
                    }
                )
    finally:
        progress_queue.put({"type": "worker_exit", "gpu_id": int(gpu_id)})


def main(ready_ratio_threshold, output_dir) -> None:
    overall_start = time.perf_counter()

    # Path to the quantized HDF5 produced by your preprocessing notebook.
    processed_h5 = "./data/mvp_processed_1000_480_16classes.h5"

    # Either set query_indices or query_ids.
    query_indices = list(range(480))
    query_ids = None

    # Use exactly these GPUs for query-parallel workers.
    gpu_ids = [0, 1, 2]

    # Fixed block numbers used by the level-wise schedule.
    block_numbers_config = [50, 200, 500, 700]
    exact_block_number = 1000

    # Number of candidates kept after each non-final level.
    keep_k = 15

    # KL marginal relaxation parameters in UOT.
    lambda_a = 0.05
    lambda_b = 0.05

    # MM iterations added to each candidate during one level sweep.
    coarse_iters = 20

    # Exact MM iterations used once on the final surviving candidates.
    final_exact_iters = 3000

    # Maximum number of whole-candidate sweeps per level.
    max_level_sweeps = 100

    # Optional cap on how many non-query candidates are used initially.
    candidate_limit = None

    # Torch floating-point dtype inside each GPU worker.
    torch_dtype_name = "float32"

    # Base candidate batch size at the coarsest block number. Finer levels use smaller batches.
    base_candidate_batch_size = 40960
    min_candidate_batch_size = 2048

    # Base batch size for the final exact stage.
    base_exact_batch_size = 128
    min_exact_batch_size = 64

    dataset_meta = read_quantized_point_cloud_batch(processed_h5)
    query_index_list = resolve_query_indices(
        dataset_meta.sample_ids,
        query_indices=query_indices,
        query_ids=query_ids,
    )
    total_queries = len(query_index_list)
    if total_queries == 0:
        raise ValueError("No queries selected.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_path / "multi_gpu_progress.log"
    write_progress_log_line(progress_log_path, "=" * 80)
    write_progress_log_line(progress_log_path, f"start_time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    write_progress_log_line(progress_log_path, f"gpu_ids={gpu_ids}")
    write_progress_log_line(progress_log_path, f"num_queries={total_queries}")

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    progress_queue: mp.Queue = ctx.Queue()

    for query_index in query_index_list:
        task_queue.put({"query_index": int(query_index)})
    for _ in gpu_ids:
        task_queue.put(None)

    workers: List[mp.Process] = []
    for gpu_id in gpu_ids:
        process = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                progress_queue,
                processed_h5,
                output_dir,
                block_numbers_config,
                exact_block_number,
                keep_k,
                lambda_a,
                lambda_b,
                coarse_iters,
                final_exact_iters,
                ready_ratio_threshold,
                max_level_sweeps,
                candidate_limit,
                torch_dtype_name,
                base_candidate_batch_size,
                min_candidate_batch_size,
                base_exact_batch_size,
                min_exact_batch_size,
            ),
        )
        process.start()
        workers.append(process)

    summaries: List[Dict[str, object]] = []
    completed_queries = 0
    failed_queries = 0
    active_queries: Dict[int, int] = {}

    while completed_queries + failed_queries < total_queries:
        event = progress_queue.get()
        event_type = event["type"]

        if event_type == "query_started":
            active_queries[int(event["query_id"])] = int(event["gpu_id"])
            line = (
                f"[START] query_id={event['query_id']} query_index={event['query_index']} "
                f"gpu={event['gpu_id']} completed={completed_queries}/{total_queries}"
            )
            print(line)
            write_progress_log_line(progress_log_path, line)
            continue

        if event_type == "level_done":
            line = (
                f"[LEVEL] query_id={event['query_id']} gpu={event['gpu_id']} "
                f"level={event['level_pos']} block={event['block_number']} "
                f"remaining={event['num_candidates_after_filter']} "
                f"before={event['num_candidates_before_filter']} "
                f"ready_ratio={event['ready_ratio']:.3f} "
                f"sweeps={event['sweeps_used']} "
                f"elapsed={event['level_elapsed_seconds']:.2f}s "
                f"done={completed_queries}/{total_queries}"
            )
            print(line)
            write_progress_log_line(progress_log_path, line)
            continue

        if event_type == "query_done":
            completed_queries += 1
            active_queries.pop(int(event["query_id"]), None)
            line = (
                f"[DONE] query_id={event['query_id']} gpu={event['gpu_id']} "
                f"elapsed={event['elapsed_seconds']:.2f}s "
                f"num_remaining={event['num_remaining']} "
                f"completed={completed_queries}/{total_queries}"
            )
            print(line)
            write_progress_log_line(progress_log_path, line)
            continue

        if event_type == "summary":
            summaries.append(event["summary"])
            continue

        if event_type == "query_error":
            failed_queries += 1
            line = (
                f"[ERROR] query_index={event['query_index']} gpu={event['gpu_id']} "
                f"error={event['error']} failed={failed_queries}"
            )
            print(line)
            write_progress_log_line(progress_log_path, line)
            continue

        if event_type == "worker_exit":
            line = f"[WORKER_EXIT] gpu={event['gpu_id']}"
            write_progress_log_line(progress_log_path, line)
            continue

    for process in workers:
        process.join()

    total_elapsed = time.perf_counter() - overall_start
    summaries.sort(key=lambda item: int(item["query_index"]))
    write_query_progress_log(output_dir, summaries, total_elapsed)

    final_line = (
        f"finished total_elapsed={total_elapsed:.2f}s "
        f"completed={completed_queries} failed={failed_queries} total={total_queries}"
    )

    print(f"Progress log saved to: {progress_log_path}")
    print(f"Query progress log saved to: {Path(output_dir) / 'multi_gpu_query_progress.log'}")
    write_progress_log_line(progress_log_path, final_line)


if __name__ == "__main__":
    main(0.5, './knn_uot_results_50_new')
    # main(0.75, './knn_uot_results_75')
    # main(0.8, './knn_uot_results_80')
    # main(0.9, './knn_uot_results_90')
    # main(1, './knn_uot_results_100')
