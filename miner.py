"""
Miner V2: Component Beam Search
===================================
Strategy: Coordinate descent in the discrete A×B (×C) component space.

Instead of random sampling + synthon similarity (local random walk), we:
1. Identify the top-K A and top-K B components from the current pool.
2. Generate candidates by crossing these top components with random partners:
   - top-A × random-B  (find best B for each good A)
   - random-A × top-B  (find best A for each good B)
   - top-A × top-B cross product (explore known-good intersections)
3. Score all candidates, update pool, repeat.

This converges toward the globally best component combinations in the
discrete lattice, not just a local neighbourhood of a known molecule.
"""

import os
import time
import json
import threading
import pandas as pd
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
TIME_LIMIT = 1800

from molecules import MoleculeManager, MoleculeUtils
from tools import (
    IterationParams,
    get_top_components,
    generate_beam_candidates,
    generate_valid_random_molecules,
    build_component_weights,
)
from models import ModelManager
from exploit import get_top_n_unexploited, run_exploit


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


model_manager = None
molecule_manager = None


def initialize_solution(config: dict):
    global molecule_manager, model_manager
    molecule_manager = MoleculeManager(config=config, db_path=DB_PATH)
    model_manager = ModelManager(config)


def find_solution(config: dict, time_start: float):
    global molecule_manager, model_manager

    iteration = 0
    n_workers = os.cpu_count() or 1
    bt.logging.info(f"[V2-Beam] CPU Workers: {n_workers}")

    params = IterationParams(config=config)
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score", "target", "anti"])

    # Phase-1: pure random sampling to build initial pool
    # Phase-2: component beam search (starts after first scoring round)
    BEAM_START_ITER = 2   # start beam search from iteration 2
    BEAM_N_TOP = 40       # number of top components to track
    BEAM_N_RAND = 80      # random partners per top component
    BEAM_N_CROSS = 25     # cross-product size per side

    # Exploit state
    use_exploit = False
    exploited_reactants: set = set()
    no_improvement_counter = 0
    seen_molecules: set = set()

    with ProcessPoolExecutor(max_workers=n_workers) as cpu_executor:
        while time.time() - time_start < TIME_LIMIT:
            iteration += 1
            iter_start = time.time()
            remaining = TIME_LIMIT - (iter_start - time_start)
            bt.logging.info(f"[V2-Beam] --- Iteration {iteration} | remaining={remaining:.0f}s ---")

            # ── Exploit mode ────────────────────────────────────────────────
            exploited_status = False
            exploit_summary = None

            if no_improvement_counter >= 1 and not top_pool.empty:
                use_exploit = True

            if use_exploit and not top_pool.empty:
                bt.logging.info("[V2-Beam] Exploit mode active.")
                all_top_mols = top_pool.to_dict("records")
                try:
                    unexploited = get_top_n_unexploited(all_top_mols, exploited_reactants, n=3)
                    if unexploited:
                        ex_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited,
                            top_n=3,
                            limit_per_reactant=1000,
                            avoid_names=seen_molecules,
                            exploited_reactants=exploited_reactants,
                        )
                        if ex_results:
                            data = pd.DataFrame(ex_results)
                            exploited_status = True
                            bt.logging.info(f"[V2-Beam] Exploit: {len(data)} candidates")
                        else:
                            raise Exception("Exploit returned nothing.")
                    else:
                        raise Exception("No unexploited molecules.")
                except Exception as e:
                    bt.logging.warning(f"[V2-Beam] Exploit failed: {e}")

            # ── Generation strategy ─────────────────────────────────────────
            if not exploited_status:
                if iteration == 1:
                    # Phase 1: large random sampling to seed the pool
                    bt.logging.info(f"[V2-Beam] Phase 1: random sampling ({params.n_samples_start} molecules)")
                    data = generate_valid_random_molecules(
                        config=config,
                        manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        mutation_prob=0,
                        elite_prob=0,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=seen_molecules,
                    )

                elif iteration >= BEAM_START_ITER and not top_pool.empty:
                    # Phase 2: component beam search
                    params.beam_phase = True
                    params.beam_iteration += 1
                    bt.logging.info(f"[V2-Beam] Phase 2: Beam search round {params.beam_iteration}")

                    top_A, top_B, top_C = get_top_components(
                        top_pool.head(config["num_molecules"] + 50),
                        molecule_manager.rxn_id,
                        n_top=BEAM_N_TOP,
                    )
                    params.beam_top_A = top_A
                    params.beam_top_B = top_B
                    params.beam_top_C = top_C
                    bt.logging.info(
                        f"[V2-Beam] Top components: A={len(top_A)}, B={len(top_B)}, C={len(top_C)}"
                    )

                    beam_df = generate_beam_candidates(
                        manager=molecule_manager,
                        top_A_ids=top_A,
                        top_B_ids=top_B,
                        top_C_ids=top_C,
                        seen_molecules=seen_molecules,
                        n_random_per_top=BEAM_N_RAND,
                        n_cross_top=BEAM_N_CROSS,
                    )
                    bt.logging.info(f"[V2-Beam] Beam candidates (pre-validate): {len(beam_df)}")

                    if not beam_df.empty:
                        beam_df = molecule_manager.validate_molecules(
                            config, beam_df,
                            time_elapsed=iter_start - time_start,
                        )
                        bt.logging.info(f"[V2-Beam] Beam candidates (post-validate): {len(beam_df)}")

                    # Fill remaining budget with random GA to maintain diversity
                    n_target = params.get_nsamples_from_time(remaining)
                    n_remaining = max(0, n_target - len(beam_df))
                    if n_remaining > 0:
                        elite_df = MoleculeUtils.select_diverse_elites(top_pool, min(150, len(top_pool)))
                        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
                        cw = build_component_weights(
                            top_pool.head(config["num_molecules"]), molecule_manager.rxn_id
                        )
                        rand_df = generate_valid_random_molecules(
                            config=config,
                            manager=molecule_manager,
                            n_samples=n_remaining,
                            mutation_prob=params.mutation_prob,
                            elite_prob=params.elite_prob,
                            executor=cpu_executor,
                            n_workers=n_workers,
                            avoid_names=seen_molecules,
                            elite_names=elite_names,
                            component_weights=cw,
                        )
                        data = pd.concat(
                            [beam_df, rand_df], ignore_index=True
                        ).drop_duplicates(subset=["name"])
                    else:
                        data = beam_df

                else:
                    # Fallback: pure random
                    data = generate_valid_random_molecules(
                        config=config,
                        manager=molecule_manager,
                        n_samples=params.get_nsamples_from_time(remaining),
                        mutation_prob=params.mutation_prob,
                        elite_prob=params.elite_prob,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=seen_molecules,
                    )

            # ── Pre-score deduplication ─────────────────────────────────────
            if data.empty:
                bt.logging.warning(f"[V2-Beam] Iteration {iteration}: no molecules generated.")
                no_improvement_counter += 1
                continue

            data = data[~data["name"].isin(seen_molecules)].reset_index(drop=True)
            if data.empty:
                bt.logging.warning(f"[V2-Beam] Iteration {iteration}: all molecules already seen.")
                no_improvement_counter += 1
                continue

            # ── GPU scoring ─────────────────────────────────────────────────
            bt.logging.info(f"[V2-Beam] Scoring {len(data)} molecules with PSICHIC...")
            gpu_start = time.time()
            data["target"] = model_manager.get_target_score_from_data(data["smiles"])
            data["anti"] = model_manager.get_antitarget_score()
            data["score"] = data["target"] - (config["antitarget_weight"] * data["anti"])
            bt.logging.info(f"[V2-Beam] GPU scoring: {time.time() - gpu_start:.2f}s")

            data["inchi"] = data["smiles"].map(MoleculeUtils.generate_inchikey)
            seen_molecules |= set(data["name"])

            # ── Pool update ─────────────────────────────────────────────────
            prev_avg = top_pool.head(config["num_molecules"])["score"].mean() if not top_pool.empty else None
            total_data = data[["name", "smiles", "inchi", "score", "target", "anti"]]

            if not top_pool.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
            else:
                top_pool = total_data.copy()

            top_pool = (
                top_pool
                .sort_values("score", ascending=False)
                .drop_duplicates(subset=["inchi"], keep="first")
                .head(config["num_molecules"] + 150)
            )

            current_avg = top_pool.head(config["num_molecules"])["score"].mean()

            if prev_avg is not None:
                improvement = (current_avg - prev_avg) / max(abs(prev_avg), 1e-6)
                params.score_improvement_rate = improvement
                if improvement <= 0:
                    no_improvement_counter += 1
                else:
                    no_improvement_counter = 0
            else:
                params.score_improvement_rate = 1.0
                no_improvement_counter = 0

            if exploit_summary and exploit_summary.get("exploited_reactant_ids") and params.score_improvement_rate <= 0:
                exploited_reactants.update(exploit_summary["exploited_reactant_ids"])

            # ── Logging & save ──────────────────────────────────────────────
            pool_max = top_pool["score"].max()
            try:
                pool_entropy = MoleculeUtils.compute_maccs_entropy(
                    top_pool.head(config["num_molecules"])["smiles"].tolist()
                )
            except Exception:
                pool_entropy = 0.0

            iter_time = time.time() - iter_start
            total_time = time.time() - time_start
            mode = "BEAM" if params.beam_phase else "RANDOM"
            bt.logging.info(
                f"[V2-Beam] Iter {iteration} | {iter_time:.1f}s | total={total_time:.0f}s | "
                f"mode={mode} | avg={current_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f} "
                f"no_imp={no_improvement_counter}"
            )

            if pool_entropy > config["entropy_min_threshold"]:
                top_entries = {"molecules": top_pool.head(config["num_molecules"])["name"].tolist()}
                with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                    json.dump(top_entries, f, ensure_ascii=False, indent=2)
                bt.logging.info("[V2-Beam] Results saved.")


if __name__ == "__main__":
    config = get_config()
    time_start = time.time()
    initialize_solution(config)
    bt.logging.info(f"[V2-Beam] Init time: {time.time() - time_start:.2f}s")
    find_solution(config, time_start)
