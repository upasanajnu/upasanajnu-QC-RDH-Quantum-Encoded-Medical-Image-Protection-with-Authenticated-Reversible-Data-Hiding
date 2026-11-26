import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .03})
D = np.load("results/medical_imgs.npz"); rows = json.load(open("results/medical.json"))
pick = ["CT-1-20", "CT-2-40", "MR-1-10", "MR-2-14"]
fig, ax = plt.subplots(3, 4, figsize=(6.3, 4.9))
for j, n in enumerate(pick):
    for i, (k, vmax, lab) in enumerate([(n, 255, "cover"), (n + "_s", 1023, "stego (10-bit)"), (n + "_r", 255, "recovered")]):
        a = ax[i, j]; a.imshow(D[k], cmap="gray", vmin=0, vmax=vmax, rasterized=True, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        sz = D[k].shape[0]
        a.set_xlabel(f"({'abcdefghijkl'[4*i+j]}) {n}, {lab}", fontsize=7.5, labelpad=3)
fig.tight_layout(h_pad=.9, w_pad=.5); fig.savefig("figs/fig_medical.pdf"); plt.close()
lines = []
for r in rows:
    lines.append(f"{r['image']} & ${r['size']}^2$ & {r['cap']:,} & {r['load']:.2f} & \\cmark & \\cmark & \\cmark & {r['H_cover']:.3f} & {r['H_stego']:.4f} & {r['chi2']:.1f} & {max(abs(r['rH']),abs(r['rV']),abs(r['rD'])):.4f} & {r['npcr_px']:.3f} & {r['uaci_px']:.3f}\\\\".replace(",", "\\,"))
open("tables/t_medical.tex", "w").write("\n".join(lines) + "\n")
