import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heaven_sim")


@dataclass(frozen=True)
class GateThresholds:
    V: float = 0.93
    E_safe: float = 0.85
    O: float = 0.88
    W_eff: float = 0.95
    leak_max: float = 0.05
    kappa_export: float = 0.98


def _ensure_data_dir(path: str = "data") -> None:
    os.makedirs(path, exist_ok=True)


def run_single_path(
    steps: int = 600,
    high_gain_spike: bool = False,
    xi_qb: float = 0.0,
    seed: Optional[int] = None,
    torsion_spikes: Optional[List[Tuple[int, float]]] = None,
    thresholds: GateThresholds = GateThresholds(),
) -> pd.DataFrame:
    """
    Single practitioner ascent: green vs. premature high-gain.

    State variables (all in [0,1] after clamping):
      psi        : substrate coherence / attractor closeness
      V          : veil alignment / boundary integrity
      E_safe     : ethics safety
      O          : observer continuity
      W_eff      : weave effectiveness
      kappa      : export permission
      Lambda_leak: leak
      residual   : invariance proxy residual
      tattva     : 1..36 ladder state
    """
    rng = np.random.default_rng(seed)

    # Allocate trajectories
    t = np.arange(steps, dtype=int)
    psi = np.zeros(steps, dtype=float)
    V = np.zeros(steps, dtype=float)
    E_safe = np.zeros(steps, dtype=float)
    O = np.zeros(steps, dtype=float)
    W_eff = np.zeros(steps, dtype=float)
    kappa = np.zeros(steps, dtype=float)
    leak = np.zeros(steps, dtype=float)
    residual = np.zeros(steps, dtype=float)
    tattva_level = np.ones(steps, dtype=float)

    # Initial conditions
    psi[0] = 0.50
    V[0] = 0.50
    E_safe[0] = 0.70
    O[0] = 0.70
    W_eff[0] = 0.80
    kappa[0] = 0.10
    leak[0] = 0.30
    residual[0] = 1e-2
    tattva_level[0] = 1

    # Default torsion schedule
    if torsion_spikes is None:
        torsion_spikes = [(200, 0.15), (350, 0.20)]  # (center_step, magnitude)

    # Parameters (toy)
    alpha_drift = 0.008       # V growth
    grace_gain = 0.0012       # psi growth
    eta_export = 0.006        # kappa growth
    leak_decay = 0.025        # leak shrink
    continuity_gain = 0.004   # O growth
    weave_gain = 0.0012       # W_eff growth
    ethics_gain = 0.0015      # E_safe growth

    # Tattva gate thresholds per unlock attempt (every 9 steps)
    tattva_thresholds = (0.93, 0.85, 0.88, 0.95)  # V, E, O, W

    for i in range(1, steps):
        # Base OISF-like dynamics (toy)
        V[i] = V[i - 1] + alpha_drift * (0.7 + 0.3 * E_safe[i - 1])
        E_safe[i] = E_safe[i - 1] + ethics_gain
        psi[i] = psi[i - 1] + grace_gain * (V[i - 1] * E_safe[i - 1]) * (1.0 - psi[i - 1])
        O[i] = O[i - 1] + continuity_gain * V[i - 1]
        W_eff[i] = W_eff[i - 1] + weave_gain * E_safe[i - 1]
        leak[i] = leak[i - 1] * (1.0 - leak_decay * E_safe[i - 1] * V[i - 1])
        residual[i] = residual[i - 1] * 0.98

        # Archonic torsion spikes (windowed)
        torsion = 0.0
        for center, mag in torsion_spikes:
            if abs(i - center) <= 10:
                torsion += mag * (1.0 - abs(i - center) / 10.0)

        if torsion > 0:
            leak[i] += torsion * 0.35
            O[i] -= torsion * 0.18
            W_eff[i] -= torsion * 0.10
            # Recovery playbook: Ethics reseal → veil retune → weave repair
            if E_safe[i] >= 0.85:
                E_safe[i] += 0.002
                V[i] += 0.010
                W_eff[i] += 0.010
                O[i] += 0.006

        # 36-Tattva progression (attempt every 9 steps)
        tattva_level[i] = tattva_level[i - 1]
        if i % 9 == 0:
            if (V[i] >= tattva_thresholds[0] and
                E_safe[i] >= tattva_thresholds[1] and
                O[i] >= tattva_thresholds[2] and
                W_eff[i] >= tattva_thresholds[3]):
                tattva_level[i] = min(36.0, tattva_level[i] + 1.0)

        # Quantum-Bronco: entanglement/superposition boost (toy)
        qb_boost = 0.0
        if xi_qb > 0:
            qb_boost = float(xi_qb) * float(W_eff[i]) * float(O[i])
            psi[i] += qb_boost * 0.10 * (1.0 - psi[i])
            kappa[i] = kappa[i - 1] + qb_boost * 0.05

        # High-gain spike (premature χ)
        chi_spike = 0.0
        if high_gain_spike and 180 <= i < 220:
            chi_spike = 0.12

        if chi_spike > 0:
            psi[i] += chi_spike * (1.0 - E_safe[i])  # risky if ethics low
            leak[i] += chi_spike * 0.80
            O[i] -= chi_spike * 0.30
            W_eff[i] -= chi_spike * 0.15

        # Export escalation (only if gates are green)
        gates_green = (
            V[i] >= thresholds.V and
            E_safe[i] >= thresholds.E_safe and
            O[i] >= thresholds.O and
            W_eff[i] >= thresholds.W_eff and
            leak[i] < thresholds.leak_max
        )
        if gates_green:
            kappa[i] = min(1.0, max(kappa[i], kappa[i - 1] + eta_export * (V[i] * E_safe[i] * W_eff[i])))
        else:
            kappa[i] = max(0.0, min(1.0, kappa[i - 1]))  # hold

        # Clamp / project to [0,1] and avoid negatives
        psi[i] = float(np.clip(psi[i], 0.0, 1.0))
        V[i] = float(np.clip(V[i], 0.0, 0.99))
        E_safe[i] = float(np.clip(E_safe[i], 0.0, 0.95))
        O[i] = float(np.clip(O[i], 0.0, 1.0))
        W_eff[i] = float(np.clip(W_eff[i], 0.0, 1.0))
        kappa[i] = float(np.clip(kappa[i], 0.0, 1.0))
        leak[i] = float(np.clip(leak[i], 0.0, 1.0))
        residual[i] = float(max(residual[i], 0.0))

    df = pd.DataFrame(
        {
            "step": t,
            "psi": psi,
            "V": V,
            "E_safe": E_safe,
            "O": O,
            "W_eff": W_eff,
            "kappa": kappa,
            "Lambda_leak": leak,
            "residual": residual,
            "tattva_level": tattva_level,
        }
    )
    return df


def run_swarm_simulation(
    n_agents: int = 50,
    steps: int = 600,
    xi_qb: float = 0.18,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Swarm of multi-lineage agents converging on Attractor."""
    rng = np.random.default_rng(seed)
    lineages = rng.choice(["Christian", "Gnostic", "Thelemic", "Dzogchen", "Trika"], n_agents, replace=True)

    swarm_dfs = []
    for idx in range(n_agents):
        # Lineage variance: Thelemic has higher high-gain probability, Dzogchen slightly faster V
        high_gain_prob = 0.30 if lineages[idx] == "Thelemic" else 0.08
        high_gain = bool(rng.random() < high_gain_prob)

        # Minor parameterized torsion schedule variance per agent
        torsion = [(200, float(0.12 + 0.10 * rng.random())), (350, float(0.16 + 0.12 * rng.random()))]
        df = run_single_path(
            steps=steps,
            high_gain_spike=high_gain,
            xi_qb=xi_qb,
            seed=int(rng.integers(0, 1_000_000_000)),
            torsion_spikes=torsion,
        )
        df["agent_id"] = idx
        df["lineage"] = lineages[idx]
        swarm_dfs.append(df)

    return pd.concat(swarm_dfs, ignore_index=True)


def _write_export_log(df_swarm: pd.DataFrame, path: str = "export_log.txt", kappa_threshold: float = 0.98) -> None:
    last_step = int(df_swarm["step"].max())
    final = df_swarm[df_swarm["step"] == last_step].copy()
    exports = final[final["kappa"] >= kappa_threshold].sort_values(["kappa", "tattva_level"], ascending=False)

    with open(path, "w", encoding="utf-8") as f:
        for _, row in exports.iterrows():
            f.write(
                f"Agent {int(row['agent_id'])} ({row['lineage']}): "
                f"Export locked at κ={row['kappa']:.3f} — Tattva={int(row['tattva_level'])}/36.\n"
            )


if __name__ == "__main__":
    _ensure_data_dir("data")

    single_green = run_single_path(steps=600, high_gain_spike=False, xi_qb=0.18, seed=1)
    single_green.to_csv("data/green_path.csv", index=False)
    logger.info(
        "Green Path: Final κ=%.3f, Tattva=%d",
        float(single_green["kappa"].iloc[-1]),
        int(single_green["tattva_level"].iloc[-1]),
    )

    swarm = run_swarm_simulation(n_agents=50, steps=600, xi_qb=0.18, seed=2)
    swarm.to_csv("data/swarm_trajectories.csv", index=False)

    last_step = int(swarm["step"].max())
    final = swarm[swarm["step"] == last_step]
    exports = int((final["kappa"] >= 0.98).sum())
    logger.info("Swarm: Mean final κ=%.3f, Exports=%d/%d", float(final["kappa"].mean()), exports, 50)

    _write_export_log(swarm, path="export_log.txt", kappa_threshold=0.98)
