"""Figure 9: payload bits per transmitted bit (the embedding rates are in Table 12).
Full text width, panels stacked, all text >= 8.5 pt at 100 % scale."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .03})
BLUE, GREY, INK = "#0072B2", "#9aa0a8", "#1f2328"
W = 6.3
base = json.load(open("results/classical_means.json"))          # per-scheme means written by figs.py
R = json.load(open("results/results.json"))
REN={"Parah et al. 2017":"Parah et al. (2017)","Chen et al. 2014":"Chen et al. (2014)","Wu et al. 2016":"Wu et al. (2016)","Bhardwaj 2021 (patient data)":"Bhardwaj (2021, patient data)","Bhardwaj 2021 (e-health)":"Bhardwaj (2021, e-health)","Bhardwaj 2022":"Bhardwaj (2022)","Yao et al. 2017":"Yao et al. (2017)","Lu et al. 2017":"Lu et al. (2017)","Lee–Huang 2013":"Lee–Huang (2013)","Chi et al. 2018":"Chi et al. (2018)"}
RR = json.load(open("results/rr.json"))["main_w8"]
labs = [REN[k] for k in base] + ["QC-RDH (2,2), per stego pixel", "QC-RDH RR, per stego pixel", "QC-RDH (2,2), per cover pixel$^{\\dagger}$"]
vals = list(base.values()) + [R["payload"]["cap"]/1048576, float(np.mean([x["payload"]/x["size"]**2 for x in RR])), R["payload"]["cap"]/262144]
rdhei = {"Puteaux–Puech (2017)": 1.0, "Yue et al. (2024)": 2.4731, "Zhang–Luo EMR (2022)": 3.2457,
         "Alqahtani–Masmoudi (2023)": 3.6, "Qiu (2025)": 4.240}
names = list(rdhei) + ["NEQRX (mechanism)", "Bhardwaj (2022)", "QC-RDH (2,2)", "QC-RDH (1,8)", "QC-RDH RR"]
per_bit = [v/8 for v in rdhei.values()] + [1/8, 2.24/8, 2.0/10, 8.0/16, float(np.mean([x["rate"] for x in RR]))]

fig, ax1 = plt.subplots(1, 1, figsize=(W, 2.0))
h = .66
cols = [GREY]*(len(names)-3) + [BLUE, BLUE, BLUE]
ax1.barh(range(len(names)), per_bit, color=cols, height=h)
for i, v in enumerate(per_bit): ax1.text(v + .006, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names); ax1.invert_yaxis()
ax1.set_xlim(0, .7); ax1.set_xlabel("payload bits per transmitted bit")
ax1.tick_params(axis="y", length=0); ax1.xaxis.grid(True, color="#e3e5e8", lw=.6); ax1.set_axisbelow(True)
fig.savefig("figs/fig_capacity.pdf")
