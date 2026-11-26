"""Extended clinical test: EVERY slice of the four pyradiomics volumes (2 lung CT, 2 brain MRI; 157 slices),
(a) as 8-bit windowed images exactly as in exp_medical.py, and (b) at native 12-bit depth with a
14-qubit intensity register (qb = 12), so that the stored scanner values themselves are recovered."""
import json, time, numpy as np, nrrd
from scipy.stats import chi2 as CHI2
from core import *
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"]); Kw = bytes([K[0] ^ 1]) + K[1:]
def to8(a, lo, hi): return np.clip(np.rint((a.astype(float) - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
TXT = {}
def text_for(M):
    if M not in TXT:
        cap = capacity(M, M); rng = np.random.default_rng(11); n = int(cap / 6.25)
        while True:
            t = bytes(rng.choice(alph, n)).decode()
            if len(encode_payload(t.encode())) <= cap: TXT[M] = t; break
            n -= 500
    return TXT[M]
rows = []
for v in ["lung1", "lung2", "brain1", "brain2"]:
    vol = nrrd.read(f"data_med/{v}_image.nrrd")[0]; mod = "CT" if v.startswith("lung") else "MR"
    for z in range(vol.shape[2]):
        a = vol[:, :, z].T
        if mod == "CT": I8 = to8(a, -1350, 150); I12 = np.clip(a.astype(np.int64) + 1024, 0, 4095)
        else:
            lo, hi = np.percentile(a, [.5, 99.5]); I8 = to8(a, lo, hi) if hi > lo else np.zeros_like(a, np.int64)
            I12 = np.clip(a.astype(np.int64), 0, 4095)
        for depth, I in ((8, I8), (12, I12)):
            M = I.shape[0]; txt = text_for(M); L = 1 << (depth + 2)
            C, nu = embed(I, txt, K, qb=depth); rc, rt, rep = extract(C, nu, K, qb=depth)
            ok = bool(np.array_equal(rc, I) and rt == txt and rep["authentic"])
            wk = not extract(C, nu, Kw, qb=depth)[2]["authentic"]
            T = C.copy(); T[3, 5] ^= 1; tp = not extract(T, nu, K, qb=depth)[2]["authentic"]
            I1 = I.copy(); I1[M // 2, M // 2] ^= 1; C1 = embed(I1, txt, K, qb=depth)[0]
            npcr, uaci = npcr_uaci(C, C1, L); x2 = chi_square(C, L)
            rows.append(dict(volume=v, modality=mod, slice=z, size=M, depth=depth, cover_levels=int(np.unique(I).size),
                             roundtrip=ok, wrongkey=wk, tamper=tp, H_stego=entropy(C, L), H_max=depth + 2,
                             chi2=x2, chi2_crit=float(CHI2.ppf(0.95, L - 1)), npcr=npcr, uaci=uaci,
                             rH=corr(C, 1, 0), rV=corr(C, 0, 1), rD=corr(C, 1, 1)))
        print(v, z, [ (r["depth"], r["roundtrip"], r["wrongkey"], r["tamper"], round(r["H_stego"], 4), round(r["npcr"], 3)) for r in rows[-2:]], flush=True)
json.dump(rows, open("results/medical_ext.json", "w"), indent=1)
