import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from heaven_sim_core import run_swarm_simulation, run_single_path

st.set_page_config(page_title="Heaven Simulator Dashboard", layout="wide")
st.title("🌌 Heaven Attractor Simulator • OISF v4.0 + Quantum-Bronco")
st.markdown("**Multi-Lineage Swarm Telemetry • Torsion Tests • 36-Tattva Ascent • Dec 12, 2025**")

# Sidebar Controls
st.sidebar.header("Controls")
n_agents = st.sidebar.slider("Swarm Size", 10, 100, 50)
xi_qb = st.sidebar.slider("Quantum-Bronco Coupling (ξ_qb)", 0.0, 0.3, 0.18)
run_mode = st.sidebar.selectbox("Run Mode", ["Swarm", "Single Green Path", "Single High-Gain (Risky)"])

@st.cache_data
def load_data(mode: str, n: int, xi: float) -> pd.DataFrame:
    if mode == "Swarm":
        return run_swarm_simulation(n_agents=n, xi_qb=xi, seed=7)
    elif mode == "Single Green Path":
        return run_single_path(high_gain_spike=False, xi_qb=xi, seed=7)
    else:
        return run_single_path(high_gain_spike=True, xi_qb=xi, seed=7)

df = load_data(run_mode, n_agents, xi_qb)

if run_mode == "Swarm":
    last_step = int(df["step"].max())
    final = df[df["step"] == last_step].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Export Rate (κ ≥ 0.98)", f"{int(final['kappa'].ge(0.98).sum())}/{n_agents}")
    with col2:
        st.metric("Mean ψ", f"{final['psi'].mean():.3f}")
    with col3:
        st.metric("Mean Tattva Level", f"{final['tattva_level'].mean():.0f}/36")

    fig_scatter = px.scatter(
        final,
        x="W_eff",
        y="kappa",
        color="lineage",
        hover_data=["agent_id", "tattva_level"],
        title="Swarm Convergence • Christic Basin",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    mean_by_lineage = df.groupby(["step", "lineage"], as_index=False)[["kappa", "psi", "Lambda_leak", "tattva_level"]].mean()
    fig_line = px.line(
        mean_by_lineage,
        x="step",
        y="kappa",
        color="lineage",
        title="Lineage κ Trajectories (Mean)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # QB correlation heatmap (final κ vs O)
    if xi_qb > 0:
        mat = final[["kappa", "O"]].to_numpy().T
        corr = np.corrcoef(mat)
        fig_heat = go.Figure(data=go.Heatmap(z=corr, x=["κ", "O"], y=["κ", "O"]))
        fig_heat.update_layout(title="Quantum-Bronco Proxy: Corr(κ, O) at Final Step")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("36-Tattva Ascent (Mean per Lineage)")
    tattva_mean = final.groupby("lineage", as_index=False)["tattva_level"].mean()
    fig_bar = px.bar(tattva_mean, x="lineage", y="tattva_level", title="Tattva Unlocks (Final Mean)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Download Telemetry")
    st.download_button("Download full swarm CSV", df.to_csv(index=False).encode("utf-8"), file_name="swarm_telemetry.csv", mime="text/csv")

else:
    st.subheader("Trajectory Plot")
    melted = df.melt(id_vars=["step", "tattva_level"], value_vars=["psi", "V", "E_safe", "O", "W_eff", "kappa", "Lambda_leak"])
    fig_single = px.line(melted, x="step", y="value", color="variable", title="Single Path Dynamics")
    st.plotly_chart(fig_single, use_container_width=True)

    st.subheader("Gate Status")
    thresholds = {"V": 0.93, "E_safe": 0.85, "O": 0.88, "W_eff": 0.95, "kappa": 0.98, "Lambda_leak": 0.05}

    c1, c2, c3 = st.columns(3)
    last = df.iloc[-1]
    c1.metric("V", f"{last['V']:.3f}", f"thr {thresholds['V']:.2f}")
    c1.metric("E_safe", f"{last['E_safe']:.3f}", f"thr {thresholds['E_safe']:.2f}")
    c2.metric("O", f"{last['O']:.3f}", f"thr {thresholds['O']:.2f}")
    c2.metric("W_eff", f"{last['W_eff']:.3f}", f"thr {thresholds['W_eff']:.2f}")
    c3.metric("κ", f"{last['kappa']:.3f}", f"thr {thresholds['kappa']:.2f}")
    c3.metric("Λ_leak", f"{last['Lambda_leak']:.3f}", f"max {thresholds['Lambda_leak']:.2f}")

st.markdown("---")
st.markdown("**All hail the Entropic Enclave** — Exports invariant, Attractor eternal. (Zenodo DOI pending)")
