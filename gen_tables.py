import json, pandas as pd, numpy as np
df = pd.read_csv("/mnt/user-data/uploads/a6bccbd9-qcrdh_results.csv")
R = json.load(open("results/results.json")); ab = pd.read_csv("results/ablation.csv")
def f(v, d=4): return f"{v:.{d}f}"
# ---- Table: per-image reversibility, capacity, statistics
rows = []
for _, r in df.iterrows():
    rows.append(f"{r.Image} & {r.EC_bits:,} & {r.payload_bits:,} & \\checkmark & \\checkmark & \\checkmark & {f(r.H_cover,3)} & {f(r.H_stego,4)} & {f(r.chi2_stego,1)} & {r.r_H:+.4f} & {r.r_V:+.4f} & {r.r_D:+.4f} \\\\".replace(",", "\\,"))
open("tables/t_main.tex", "w").write("\n".join(rows))
# ---- summary sensitivity (mean, min, max over 8 images)
def summ(c, d=3): return f"{df[c].mean():.{d}f} & [{df[c].min():.{d}f}, {df[c].max():.{d}f}]"
lines = []
for suf, lab in [("chaos", "chaotic parameter $x_0\\!+\\!10^{-12}$"), ("qkd", "one flipped bit of $K_{\\mathrm{QKD}}$"), ("1px", "one cover pixel (LSB)"), ("1char", "one character of $T$")]:
    lines.append(f"{lab} & {summ('NPCR_'+suf)} & {summ('UACI_'+suf)} \\\\")
open("tables/t_sens.tex", "w").write("\n".join(lines))
# ---- ablation
al = []
for _, r in ab.iterrows():
    al.append(f"{r.variant} & {r.entropy:.4f} & {r.chi2:.0f} & {r.r_H:+.4f} & {r.NPCR_1px:.4f} & {int(r.changed_px):,} \\\\".replace(",", "{,}") if False else f"{r.variant} & {r.entropy:.4f} & {r.chi2:.0f} & {r.r_H:+.4f} & {r.NPCR_1px:.4f} & {int(r.changed_px)} \\\\")
open("tables/t_ablation.tex", "w").write("\n".join(al))
# ---- timing
tm = f"{df.t_embed.mean():.3f} & {df.t_embed.min():.3f}--{df.t_embed.max():.3f} & {df.t_extract.mean():.3f} & {df.t_extract.min():.3f}--{df.t_extract.max():.3f}"
open("tables/t_time.tex", "w").write(tm)
# ---- macros with headline numbers used in text
S = {
 "HstegoMean": df.H_stego.mean(), "HstegoMin": df.H_stego.min(), "HstegoMax": df.H_stego.max(),
 "ChiMin": df.chi2_stego.min(), "ChiMax": df.chi2_stego.max(), "ChiMean": df.chi2_stego.mean(),
 "rAbsMax": float(np.abs(df[["r_H","r_V","r_D"]].values).max()),
 "NPCRlo": min(df[[c for c in df if c.startswith("NPCR")]].min()), "NPCRhi": max(df[[c for c in df if c.startswith("NPCR")]].max()),
 "UACIlo": min(df[[c for c in df if c.startswith("UACI")]].min()), "UACIhi": max(df[[c for c in df if c.startswith("UACI")]].max()),
 "TembMean": df.t_embed.mean(), "TextMean": df.t_extract.mean(),
 "rcoverMin": df.r_cover_H.min(), "rcoverMax": df.r_cover_H.max(),
 "HcoverMin": df.H_cover.min(), "HcoverMax": df.H_cover.max(),
}
open("tables/macros.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v:.4f}}}" for k, v in S.items()))
print({k: round(float(v), 4) for k, v in S.items()}); print(open("tables/t_sens.tex").read()); print(open("tables/t_ablation.tex").read()); print(open("tables/t_time.tex").read())
print(df.tamper_detected.unique(), df.wrong_key_accepted.unique())
