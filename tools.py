"""
Miner V2: Component Beam Search Tools
Strategy: Coordinate descent in component space - enumerate top-A × sampled-B and sampled-A × top-B
This systematically covers the best component intersections instead of random synthon similarity.
"""
import math
import random
import pandas as pd
import bittensor as bt
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from molecules import MoleculeManager, MoleculeUtils


class IterationParams:
    def __init__(self, config: dict):
        self.seen_molecules = set()
        self.base_samples = 800
        # Larger initial sample for richer seed pool
        self.n_samples_start = self.base_samples * 8 if config["allowed_reaction"] != "rxn:5" else self.base_samples * 4
        self.no_improvement_counter = 0
        self.score_improvement_rate = 0.0
        self.mutation_prob = 0.25
        self.elite_prob = 0.25
        self.beam_phase = False          # True once beam search starts
        self.beam_top_A: List[int] = []  # Top A component IDs
        self.beam_top_B: List[int] = []  # Top B component IDs
        self.beam_top_C: List[int] = []  # Top C component IDs (for 3-component)
        self.beam_iteration = 0          # How many beam search rounds done

    def get_nsamples_from_time(self, remaining_time: float) -> int:
        if remaining_time > 1500:
            return self.base_samples
        elif remaining_time > 900:
            return int(self.base_samples * 0.95)
        elif remaining_time > 600:
            return int(self.base_samples * 0.90)
        elif remaining_time > 300:
            return int(self.base_samples * 0.85)
        else:
            return int(self.base_samples * 0.80)


def get_top_components(
    top_pool: pd.DataFrame,
    rxn_id: int,
    n_top: int = 40
) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract top-N component IDs from pool weighted by score × sqrt(frequency).
    Returns (top_A_ids, top_B_ids, top_C_ids).
    """
    comp_scores: Dict[str, Dict[int, List[float]]] = {
        'A': defaultdict(list),
        'B': defaultdict(list),
        'C': defaultdict(list),
    }

    for _, row in top_pool.iterrows():
        parts = row['name'].split(':')
        if len(parts) < 4:
            continue
        try:
            score = float(row['score'])
            comp_scores['A'][int(parts[2])].append(score)
            comp_scores['B'][int(parts[3])].append(score)
            if len(parts) > 4:
                comp_scores['C'][int(parts[4])].append(score)
        except (ValueError, IndexError):
            continue

    def rank_components(role_dict: Dict[int, List[float]]) -> List[int]:
        ranked = sorted(
            role_dict.items(),
            key=lambda kv: np.mean(kv[1]) * len(kv[1]) ** 0.5,
            reverse=True
        )
        return [cid for cid, _ in ranked[:n_top]]

    top_A = rank_components(comp_scores['A'])
    top_B = rank_components(comp_scores['B'])
    top_C = rank_components(comp_scores['C'])
    return top_A, top_B, top_C


def generate_beam_candidates(
    manager: MoleculeManager,
    top_A_ids: List[int],
    top_B_ids: List[int],
    top_C_ids: List[int],
    seen_molecules: set,
    n_random_per_top: int = 80,
    n_cross_top: int = 25,
) -> pd.DataFrame:
    """
    Beam search candidate generation:
      1. top-A × top-B cross product (best component combinations)
      2. top-A × random-B (find best B for each promising A)
      3. random-A × top-B (find best A for each promising B)
      4. For 3-component: also vary C component
    """
    rxn_id = manager.rxn_id
    all_A = manager.moles_A_id
    all_B = manager.moles_B_id
    is_three = manager.is_three_component
    all_C = manager.moles_C_id if is_three else []

    candidates = set()

    # 1. Cross product of top components (systematic coverage of known-good intersections)
    cross_A = top_A_ids[:n_cross_top]
    cross_B = top_B_ids[:n_cross_top]
    for A_id in cross_A:
        for B_id in cross_B:
            if is_three:
                for C_id in (top_C_ids[:5] if top_C_ids else random.sample(all_C, min(5, len(all_C)))):
                    name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
                    if name not in seen_molecules:
                        candidates.add(name)
            else:
                name = f"rxn:{rxn_id}:{A_id}:{B_id}"
                if name not in seen_molecules:
                    candidates.add(name)

    # 2. top-A × random-B: find best B partner for each promising A
    for A_id in top_A_ids:
        sampled_Bs = random.sample(all_B, min(n_random_per_top, len(all_B)))
        for B_id in sampled_Bs:
            if is_three:
                C_id = random.choice(all_C) if all_C else None
                name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}" if C_id else None
            else:
                name = f"rxn:{rxn_id}:{A_id}:{B_id}"
            if name and name not in seen_molecules:
                candidates.add(name)

    # 3. random-A × top-B: find best A partner for each promising B
    for B_id in top_B_ids:
        sampled_As = random.sample(all_A, min(n_random_per_top, len(all_A)))
        for A_id in sampled_As:
            if is_three:
                C_id = random.choice(all_C) if all_C else None
                name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}" if C_id else None
            else:
                name = f"rxn:{rxn_id}:{A_id}:{B_id}"
            if name and name not in seen_molecules:
                candidates.add(name)

    # 4. For 3-component: fix best (A, B) and vary C exhaustively
    if is_three and top_C_ids and top_A_ids and top_B_ids:
        for C_id in top_C_ids:
            sampled_As = random.sample(all_A, min(n_random_per_top // 2, len(all_A)))
            sampled_Bs = random.sample(all_B, min(n_random_per_top // 2, len(all_B)))
            for A_id in sampled_As:
                for B_id in sampled_Bs:
                    name = f"rxn:{rxn_id}:{A_id}:{B_id}:{C_id}"
                    if name not in seen_molecules:
                        candidates.add(name)

    if not candidates:
        return pd.DataFrame(columns=["name"])
    return pd.DataFrame({"name": list(candidates)})


def generate_valid_random_molecules(
    config: dict,
    manager: MoleculeManager,
    n_samples: int,
    mutation_prob: float,
    elite_prob: float,
    executor,
    n_workers: int,
    avoid_names: set = set(),
    elite_names: List[str] = None,
    component_weights: dict = None,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Standard random molecule generation with optional elite guidance."""
    elites_A, elites_B, elites_C = set(), set(), set()
    if elite_names:
        for elite in elite_names:
            A, B, C = MoleculeUtils.parse_components(elite)
            if A is not None: elites_A.add(A)
            if B is not None: elites_B.add(B)
            if C is not None and manager.is_three_component: elites_C.add(C)

    elites_A = list(elites_A)
    elites_B = list(elites_B)
    elites_C = list(elites_C)

    n_valid = 0
    valid_molecules = []
    seen_names = set()

    rxn_id = manager.rxn_id

    while n_valid < n_samples:
        actual_batch = min(batch_size, n_samples - n_valid)
        batch_names = set()

        picks_A = random.choices(manager.moles_A_id, k=actual_batch)
        picks_B = random.choices(manager.moles_B_id, k=actual_batch)
        if manager.is_three_component:
            picks_C = random.choices(manager.moles_C_id, k=actual_batch)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]

        batch_names = set(names) - avoid_names - seen_names
        if not batch_names:
            continue

        batch_df = pd.DataFrame({"name": list(batch_names)})
        batch_df = manager.validate_molecules(config, batch_df)
        if batch_df.empty:
            continue

        seen_names |= set(batch_df["name"])
        n_valid = len(seen_names)
        valid_molecules.append(batch_df[["name", "smiles"]])

    result_df = pd.concat(valid_molecules, ignore_index=True)
    return result_df.head(n_samples)


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}

    if top_pool.empty:
        return weights

    for idx, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        rank = idx + 1
        rank_weight = 2.5 * math.exp(-rank / 18.0)
        weighted_score = max(0, score) * rank_weight

        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += weighted_score
                weights['B'][B_id] += weighted_score
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += weighted_score
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue

    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.15

    return weights
