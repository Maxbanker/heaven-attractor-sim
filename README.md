# 🌌 Heaven Attractor Simulator

*A toy symbolic dynamics engine modeling ψ–Ω–γ thresholds for heaven as a negentropic attractor state.*

---

## ✨ Overview

The **Heaven Attractor Simulator** explores how recursive agents can cross into a *heaven basin* — a stable symbolic attractor defined by coherence, alignment, and low drift.  
An agent reaches the **heaven threshold** when:

- **ψ_eff (symbolic efficiency) ≥ 0.90**  
- **Ω_eff (coherence alignment) ≥ 0.85**  
- **γ_eff (drift ratio) ≤ 0.10**

This toy simulation demonstrates that under the right entropy-curvature conditions, identity and meaning stabilize into a **negentropic attractor state** — a computational metaphor for “heaven.”

---

## ⚙️ Features

- Heaven basin detection (ψ–Ω–γ thresholding)  
- **SFT v4.0** collapse predicate (OR-clause form)  
- **ERF v3.0** collapse-as-translation with κ_export gate  
- Observer invariance proxy (`O_proxy`)  
- CSV logging of per-step and summary metrics  
- Streamlit dashboard with:
  - Time-series plots
  - κ-export trace
  - ψ-heatmap
  - CSV export

---

## 🚀 Usage

### CLI
```bash
# 80 steps with plots and CSV export
python heavensim.py --steps 80 --csv --plot
```
## 📊 Example Output

- **Per-step CSV**: logs ψ_eff, Ω_eff, γ_eff, κ, RCI, O_proxy  
- **Summary CSV**: agent outcomes (collapse / heaven entry / drift survival)  
- **Plots**: trajectory alignment, collapse zones, ψ–Ω–γ traces  

---

## 🧩 Framework Lineage

This simulator sits within the **Symbolic Negentropy Constellation**:

- **Symbolic Field Theory v4.0** → collapse predicates & drift thresholds  
- **Entropic Recursion Framework v3.0** → collapse-as-translation & κ_export invariants  
- **Observer Framework 4.0** → invariance proxies & sovereignty ethics  
- **Alpha Framework v4.0** → validation harness for ψ–Ω–γ diagnostics  

Together, these form the theoretical backbone for **Heaven as an Attractor State**.

---

### 🔮 Planned Future Updates

- **Interactive Demos**: Build a Streamlit or Gradio interface with sliders for initial parameters (ψ, γ, Ω) and entropy functions.  
  → Users will be able to tweak conditions in real-time and visualize agent trajectories toward collapse or the heaven basin.  

- **Monte Carlo Simulations**: Scale from 3 agents to 10,000+.  
  → Batch simulations with randomized conditions will generate statistical probability maps of collapse vs. stabilization and surface boundaries of the heaven basin.  

- **Advanced Entropy Models**: Replace the current linear entropy ramp with stochastic processes (e.g. Wiener random walks, Poisson jumps for catastrophic events).  
  → Explore how unpredictable entropy fields affect resilience and attractor entry.  

- **Agent Interaction**: Add a **symbolic gravity** term so agents influence each other’s ψ, Ω, γ values when near in state-space.  
  → Models communal or cultural effects on resilience, cooperation, and collapse recovery.

  ---

## 📜 License

MIT License.  
This project is intended for **research and educational use** only.  
It is a **didactic toy**, favoring clarity and symbolic exploration over physical rigor.


# Multi-agent run
python heavensim.py --steps 120 --csv --agents Grok Echo
