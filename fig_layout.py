"""Combined figures for page layout: key material (chaos + BB84) and the four circuits."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from core import *; from qcirc import *
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .03})
BLUE, VERM, INK = "#0072B2", "#D55E00", "#1f2328"; W = 6.3
def finish(fig, path, labels, gap=.02):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(fig.axes, labels):
        bb = ax.get_tightbbox(r).transformed(inv); pos = ax.get_position()
        fig.text((pos.x0+pos.x1)/2, bb.y0-gap, lab, ha="center", va="top", fontsize=8.5)
    fig.savefig(path); plt.close(fig)
# ---------------- key material: chaos (a-c) + BB84 (d), one row
c = np.load("results/chaos.npz"); As, LL, orb = c["As"], c["LL"], c["orb"]; q = json.load(open("results/qber_runs.json"))
fig, ax = plt.subplots(1, 4, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
for a_, o in zip(As, orb):
    if not np.isnan(o).any() and np.abs(o).max() < 50: ax[0].plot([a_]*len(o), o, ".", ms=.5, color=INK, alpha=.4, rasterized=True)
ax[0].axvline(-2.8, color=VERM, lw=.8); ax[0].set(xlabel="$a$", ylabel="$x$")
ax[1].plot(As, LL, color=BLUE, lw=.8); ax[1].axhline(0, color=INK, lw=.5); ax[1].axvline(-2.8, color=VERM, lw=.8); ax[1].set(xlabel="$a$", ylabel="largest Lyapunov exp.")
ax[2].scatter(c["xs"], c["ys"], s=.3, color=INK, alpha=.5, lw=0, rasterized=True); ax[2].set(xlabel="$x$", ylabel="$y$")
ax[3].axhline(.11, color=VERM, ls="--", lw=.8); ax[3].text(20.5, .125, "abort 11 %", ha="right", fontsize=6.5)
ax[3].plot(range(1, 21), q["eve"], "s-", color=INK, ms=2.6, lw=.7, mfc="white", label="intercept-resend")
ax[3].plot(range(1, 21), q["honest"], "o-", color=BLUE, ms=2.6, lw=.7, label="no eavesdropper")
ax[3].set(xlabel="BB84 run", ylabel="QBER", ylim=(-.01, .50), xticks=[1, 10, 20]); ax[3].legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=1.6, borderaxespad=0)
fig.tight_layout(w_pad=1.0)
finish(fig, "figs/fig_keys.pdf", ["(a) orbit vs. $a$", "(b) Lyapunov exponent", "(c) orbit in $(x,y)$", "(d) BB84 QBER"])
# ---------------- circuits with print-legible fonts
demo = np.array([[3, 1], [2, 0]]); y, x, c_ = registers(2)   # small illustrative cover: few set bits
sty = {"name": "bw", "fontsize": 13, "subfontsize": 10, "displaytext": {"R_2": "$R_2$", "qft": "QFT", "qft_dg": "QFT†"}}
def save(qc, name, fold=-1, scale=1.0):
    f = qc.draw("mpl", fold=fold, scale=scale, idle_wires=False, style=sty); f.savefig(f"figs/{name}.pdf", bbox_inches="tight", pad_inches=.02); plt.close(f)
qa = QuantumCircuit(y, x, c_); neqr_prep(qa, y, x, c_, demo); qa.barrier(); scale_up(qa, y, x); save(qa, "circ_a_prep_scale")
from qiskit.circuit import Gate
qb = QuantumCircuit(y, x, c_); qb.append(Gate("R_2", 10, []), list(c_)); qb.barrier(); write_D(qb, y, x, c_, np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,0]])); save(qb, "circ_b_embed")
qcr = QuantumCircuit(y, x, c_); draper_affine(qcr, x, y, 3, 1); qcr.barrier(); draper_affine(qcr, y, x, 2, 3); save(qcr, "circ_c_scramble")
qd = QuantumCircuit(y, x, c_); key_oracle(qd, y, x, c_, np.array([[5,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,769]])); save(qd, "circ_d_oracle")
print("ok")
