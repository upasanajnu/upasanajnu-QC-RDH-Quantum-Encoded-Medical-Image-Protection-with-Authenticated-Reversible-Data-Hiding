import os, json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit
from core import *
from qcirc import *

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": .03, "axes.grid": False})
BLUE, VERM, GREEN, AMBER, PINK, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#8a8f98", "#1f2328"
W = 6.3                                                         # textwidth in inches (preprint, 12 pt)
R = json.load(open("results/results.json"))
def panel(ax, s): ax._sublabel = s                               # placed below the axes by finish()
def finish(fig, path, gap=0.018):
    """Put each axes' (a), (b) ... label centred BELOW the axes (under tick labels / x-label), then save."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax in fig.axes:
        lab = getattr(ax, "_sublabel", None)
        if lab is None: continue
        bb = ax.get_tightbbox(r).transformed(inv); pos = ax.get_position()
        fig.text((pos.x0 + pos.x1)/2, bb.y0 - gap, lab, ha="center", va="top", fontsize=9)
    fig.savefig(path); plt.close(fig)

# ---------------------------------------------------------------- chaos
c = np.load("results/chaos.npz"); As, LL, orb = c["As"], c["LL"], c["orb"]
fig, ax = plt.subplots(1, 3, figsize=(W, 1.9))
for a_, o in zip(As, orb):
    if not np.isnan(o).any() and np.abs(o).max() < 50: ax[0].plot([a_]*len(o), o, ".", ms=.6, color=INK, alpha=.4, rasterized=True)
ax[0].axvline(-2.8, color=VERM, lw=.8); ax[0].set(xlabel="$a$", ylabel="$x$"); panel(ax[0], "(a)")
ax[1].plot(As, LL, color=BLUE, lw=.9); ax[1].axhline(0, color=INK, lw=.5); ax[1].axvline(-2.8, color=VERM, lw=.8)
ax[1].set(xlabel="$a$", ylabel="largest Lyapunov exp."); panel(ax[1], "(b)")
ax[2].scatter(c["xs"], c["ys"], s=.4, color=INK, alpha=.5, lw=0, rasterized=True); ax[2].set(xlabel="$x$", ylabel="$y$"); panel(ax[2], "(c)")
fig.tight_layout(w_pad=1.6); finish(fig, "figs/fig_chaos.pdf")

# ---------------------------------------------------------------- BB84 QBER
q = json.load(open("results/qber_runs.json"))
fig, ax = plt.subplots(figsize=(W*.55, 1.9))
ax.axhline(.11, color=VERM, ls="--", lw=.9); ax.text(20.3, .118, "abort threshold 11 %", color=INK, ha="right", fontsize=7.5)
ax.plot(range(1, 21), q["honest"], "o-", color=BLUE, ms=3.5, lw=.9, label="no eavesdropper")
ax.plot(range(1, 21), q["eve"], "s-", color=INK, ms=3.5, lw=.9, mfc="white", label="intercept-resend")
ax.set(xlabel="independent BB84 run (2048 raw qubits)", ylabel="QBER", ylim=(-.01, .36)); ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.13))
fig.savefig("figs/fig_bb84.pdf"); plt.close()

# ---------------------------------------------------------------- circuits (2x2 demo cover, n = 2)
demo = np.array([[236, 215], [239, 203]]); y, x, c_ = registers(2)
sty = {"fontsize": 10, "subfontsize": 8, "displaycolor": {"h": ("#0072B2", "#ffffff"), "swap": ("#009E73", "#ffffff"),
       "mcx": ("#D55E00", "#ffffff")}}
def save_circ(qc, name, fold=-1, scale=.75):
    f = qc.draw("mpl", fold=fold, scale=scale, idle_wires=False, style={"name": "bw", "fontsize": 9}); f.savefig(f"figs/{name}.pdf"); plt.close(f)
qa = QuantumCircuit(y, x, c_); neqr_prep(qa, y, x, c_, demo); qa.barrier(label="NEQR$(I)$"); scale_up(qa, y, x); qa.barrier(label="scale")
save_circ(qa, "circ_a_prep_scale", fold=60, scale=.62)
qb = QuantumCircuit(y, x, c_); shift_by_4(qb, c_); qb.barrier(label="$|c\\rangle\\!\\to\\!|4c\\rangle$")
write_D(qb, y, x, c_, np.array([[2,1,3,0],[1,0,2,3],[0,3,1,2],[3,2,0,1]]))
save_circ(qb, "circ_b_embed", fold=60, scale=.62)
qcr = QuantumCircuit(y, x, c_); draper_affine(qcr, x, y, 3, 1); qcr.barrier(); draper_affine(qcr, y, x, 2, 3)
f = qcr.decompose().draw("mpl", fold=40, scale=.6, style={"name": "bw", "fontsize": 9}) if False else qcr.draw("mpl", fold=40, scale=.7, idle_wires=False, style={"name": "bw"})
f.savefig("figs/circ_c_scramble.pdf"); plt.close(f)
qd = QuantumCircuit(y, x, c_); key_oracle(qd, y, x, c_, np.array([[5,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,769]]))
save_circ(qd, "circ_d_oracle", fold=40, scale=.62)
ex = QuantumCircuit(4, 4)
for i, (b, ab, bb) in enumerate([(0,0,0),(1,0,1),(1,1,1),(0,1,0)]):
    if b: ex.x(i)
    if ab: ex.h(i)
    ex.barrier(i)
    if bb: ex.h(i)
    ex.measure(i, i)
f = ex.draw("mpl", scale=.75, style={"name": "bw"}); f.savefig("figs/circ_bb84.pdf"); plt.close(f)

# ---------------------------------------------------------------- Theorem 1 verification
tt = pd.read_csv("results/theorem1_trials.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.15]})
for n_, mk, col, lab in [(2, "o", BLUE, "$n=2$ (16 pixels)"), (3, "s", VERM, "$n=3$ (64 pixels)")]:
    d = tt[tt.n == n_]; ax[0].scatter(d.changed_in + np.random.default_rng(n_).normal(0, .12, len(d)), d.changed_out + np.random.default_rng(n_+9).normal(0, .12, len(d)),
                                      s=11, marker=mk, color=col, alpha=.7, lw=0, label=lab)
mx = int(tt.changed_in.max()) + 1; ax[0].plot([0, mx], [0, mx], color=INK, lw=.7)
ax[0].set(xlabel="pixels changed in the plaintext, $d_H(I,J)$", ylabel="pixels changed in the cipher image", xlim=(0, mx), ylim=(0, mx))
ax[0].legend(frameon=False, loc="upper left"); panel(ax[0], "(a)")
labels = ["fixed-key\ncipher\n(Thm. 3)", "NEQRX-style\n(re-impl.)", "QC-RDH\n(SIV nonce)"]
vals = [100*R["fixed_key_changed"]/1048576, R["neqrx"]["npcr"], 100*R["siv_changed"][0]/R["siv_changed"][1]]
ax[1].bar([0, 1], vals[:2], color=[VERM, VERM], width=.55); ax[1].bar([2], [vals[2]], color=BLUE, width=.55)
ax[1].set_yscale("log"); ax[1].set(ylabel="NPCR for a 1-pixel change (%)", ylim=(1e-5, 300))
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels)
for xk, v in zip(range(3), vals): ax[1].text(xk, v*1.5, f"{v:.5f}" if v < 1 else f"{v:.2f}", ha="center", fontsize=7.5)
ax[1].axhline(100/(512*512), color=INK, ls=":", lw=.8, label="ceiling $100/N$ (Cor. 1)"); ax[1].legend(frameon=False, loc="upper left", fontsize=7)
panel(ax[1], "(b)"); fig.tight_layout(w_pad=2.2); finish(fig, "figs/fig_thm1.pdf")

# ---------------------------------------------------------------- chosen-cover attacks
a = np.load("results/attack_imgs.npz")
fig, ax = plt.subplots(2, 3, figsize=(W, 4.6))
items = [(a["victim"], "(a) hidden cover (victim)"),
         (a["rec_fixed"], f"(b) fixed-key QC-RDH: attacker output\n({R['attack_fixed']['queries']} chosen covers, exact)"),
         (a["rec_siv"], "(c) QC-RDH with SIV: attacker output\n(%.3f %% correct; chance %.2f %%)" % (R["attack_siv"]["correct_pct"], R["attack_siv"]["chance_pct"])),
         (a["victim"], "(d) hidden image (victim)"), (a["marked"], "(e) NEQRX-style marked cipher image"),
         (a["rec_nx"], f"(f) attacker output, {R['neqrx']['queries']} chosen\nplaintexts (PSNR %.1f dB)" % R["neqrx"]["psnr"])]
for axx, (im, t) in zip(ax.ravel(), items):
    axx.imshow(im, cmap="gray", vmin=0, vmax=255, rasterized=True); axx.set_xticks([]); axx.set_yticks([])
    for sp in axx.spines.values(): sp.set_visible(False)
    axx.set_xlabel(t, fontsize=8, labelpad=4)
fig.tight_layout(h_pad=1.2, w_pad=.8); fig.savefig("figs/fig_attack.pdf"); plt.close()

# ---------------------------------------------------------------- visual results (3 images)
IMG = np.load("results/images.npz"); ST = np.load("results/stego3.npz")
fig, ax = plt.subplots(3, 5, figsize=(W, 3.9))
for r, n in enumerate(["camera", "astronaut", "moon"]):
    cv, st, rc = IMG[n], ST[f"{n}_s"], ST[f"{n}_r"]
    ax[r,0].imshow(cv, cmap="gray", vmin=0, vmax=255, rasterized=True)
    ax[r,1].hist(cv.ravel(), 256, range=(0, 256), color=BLUE, lw=0); ax[r,1].set_xlim(0, 255)
    ax[r,2].imshow(st, cmap="gray", vmin=0, vmax=1023, rasterized=True)
    ax[r,3].hist(st.ravel(), 256, range=(0, 1024), color=VERM, lw=0); ax[r,3].set_xlim(0, 1023)
    ax[r,4].imshow(rc, cmap="gray", vmin=0, vmax=255, rasterized=True)
    ax[r,0].set_ylabel(n, fontsize=8)
    for k in (0, 2, 4): ax[r,k].set_xticks([]); ax[r,k].set_yticks([])
    for k in (1, 3): ax[r,k].set_yticks([]); ax[r,k].tick_params(labelsize=6)
for k, t in enumerate(["cover $512^2$", "cover histogram", "stego $1024^2$ (10-bit)", "stego histogram", "recovered ($\\max|\\Delta|=0$)"]):
    ax[0,k].set_title(t, fontsize=7.6)
fig.tight_layout(w_pad=.4, h_pad=.5); fig.savefig("figs/fig_visual.pdf"); plt.close()

# ---------------------------------------------------------------- correlation scatter (camera)
cv, st = IMG["camera"], ST["camera_s"]; rr = np.random.default_rng(0)
fig, ax = plt.subplots(2, 3, figsize=(W, 3.5))
for j, (dx, dy, nm) in enumerate([(1,0,"horizontal"), (0,1,"vertical"), (1,1,"diagonal")]):
    for i, (im, lab, lim, col) in enumerate([(cv, "cover", 255, BLUE), (st, "stego", 1023, VERM)]):
        ys_ = rr.integers(0, im.shape[0]-dy, 4000); xs_ = rr.integers(0, im.shape[1]-dx, 4000)
        ax[i,j].scatter(im[ys_, xs_], im[ys_+dy, xs_+dx], s=1.2, alpha=.35, color=col, lw=0, rasterized=True)
        ax[i,j].set(xlim=(0, lim), ylim=(0, lim)); ax[i,j].tick_params(labelsize=6.5); ax[i,j].set_aspect("equal")
        panel(ax[i,j], f"({'abcdef'[3*i+j]}) {nm}, {lab}: $r={corr(im, dx, dy):+.4f}$")
fig.tight_layout(h_pad=2.6, w_pad=1.0); finish(fig, "figs/fig_corr.pdf", gap=.01)

# ---------------------------------------------------------------- sensitivity (user's CSV)
df = pd.read_csv("/mnt/user-data/uploads/a6bccbd9-qcrdh_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3)); xs = np.arange(len(df)); wd = .2
sets = [("chaos", "chaotic key", BLUE, "o"), ("qkd", "QKD key", VERM, "s"), ("1px", "1 cover pixel", GREEN, "^"), ("1char", "1 text char", AMBER, "D")]
for k, (suf, lab, col, mk) in enumerate(sets):
    ax[0].plot(xs + (k-1.5)*.1, df[f"NPCR_{suf}"], mk, color=col, ms=4, label=lab, mfc="white" if k % 2 else col)
    ax[1].plot(xs + (k-1.5)*.1, df[f"UACI_{suf}"], mk, color=col, ms=4, label=lab, mfc="white" if k % 2 else col)
ax[0].axhline(100*(1-1/1024), color=INK, lw=.8, ls="--"); ax[0].text(-.45, 99.925, "dashed: ideal 99.902", ha="left", fontsize=7)
ax[1].axhline(100*1025/3072, color=INK, lw=.8, ls="--"); ax[1].text(-.45, 33.48, "dashed: ideal 33.366", ha="left", fontsize=7)
for a_, yl, lim in [(ax[0], "NPCR (%)", (99.89, 99.93)), (ax[1], "UACI (%)", (33.28, 33.50))]:
    a_.set_xticks(xs); a_.set_xticklabels(df.Image, rotation=35, ha="right"); a_.set_ylabel(yl); a_.set_ylim(*lim)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.07), fontsize=7.5)
panel(ax[0], "(a)"); panel(ax[1], "(b)")
fig.tight_layout(w_pad=1.6); finish(fig, "figs/fig_sens.pdf")

# ---------------------------------------------------------------- capacity comparison
yao = [1.56]*6 + [.79,1.30,1.22,1.23,1.16,.56,1.17,1.46]
lu = [1.05,1.49,1.50,1.50,1.05,1.49,1.50,1.22,1.17,1.18,1.00,.53,1.04,1.29]
lee = [1.07]*6 + [.54,.89,.83,.84,.80,.37,.80,1.03]
chi = [1.50,1.49,1.50,1.50,1.48,1.49,.75,1.22,1.17,1.18,1.00,.53,1.04,1.29]
base = {"Parah et al. 2017": .75, "Chen et al. 2014": .50, "Wu et al. 2016": 1.00, "Bhardwaj 2021 (patient data)": 2.00,
        "Bhardwaj 2021 (e-health)": 2.00, "Bhardwaj 2022": 2.24, "Yao et al. 2017": np.mean(yao), "Lu et al. 2017": np.mean(lu),
        "Lee–Huang 2013": np.mean(lee), "Chi et al. 2018": np.mean(chi)}
json.dump({k: float(v) for k, v in base.items()}, open("results/classical_means.json", "w"), indent=1)
rdhei = {"Puteaux–Puech (2017)": 1.0, "Yue et al. (2024)": 2.4731, "Zhang–Luo EMR (2022)": 3.2457, "Alqahtani–Masmoudi (2023)": 3.6, "Qiu (2025)": 4.240}
fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
labs = list(base) ; vals = list(base.values())
ax[0].barh(range(len(labs)), vals, color=GREY, height=.62)
ax[0].barh([len(labs)], [R["payload"]["cap"]/1048576], color=BLUE, height=.62); ax[0].barh([len(labs)+1], [R["payload"]["cap"]/262144], color=BLUE, height=.62, hatch="////", fc="white", ec=BLUE)
labs += ["QC-RDH (per stego pixel)", "QC-RDH (per cover pixel, not comparable)"]; vals += [2.0, 8.0]
for i, v in enumerate(vals): ax[0].text(v+.08, i, f"{v:.2f}", va="center", fontsize=7)
ax[0].set_yticks(range(len(labs))); ax[0].set_yticklabels(labs, fontsize=7.2); ax[0].invert_yaxis()
ax[0].set(xlabel="embedding rate, bits per pixel", xlim=(0, 9.8)); panel(ax[0], "(a)")
names = list(rdhei) + ["NEQRX (mechanism)", "Bhardwaj (2022)", "QC-RDH"]
per_bit = [v/8 for v in rdhei.values()] + [1/8, 2.24/8, 2.0/10]
cols = [GREY]*(len(names)-1) + [BLUE]
ax[1].barh(range(len(names)), per_bit, color=cols, height=.62)
for i, v in enumerate(per_bit): ax[1].text(v+.01, i, f"{v:.3f}", va="center", fontsize=7)
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7.2); ax[1].invert_yaxis()
ax[1].set(xlabel="payload bits per transmitted bit", xlim=(0, .66)); panel(ax[1], "(b)")
fig.tight_layout(w_pad=1.4); finish(fig, "figs/fig_capacity.pdf")
print("figs done", sorted(os.listdir("figs")))
