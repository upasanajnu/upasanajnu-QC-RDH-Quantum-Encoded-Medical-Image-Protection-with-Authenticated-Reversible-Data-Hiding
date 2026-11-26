"""Medical-image experiment: 4 lung-CT slices (512x512) and 4 brain-MRI slices (256x256)
from the pyradiomics test data (Harvard Medical School, BSD licence)."""
import json, time, numpy as np, nrrd
from core import *
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
def to8(a, lo, hi): return np.clip(np.rint((a.astype(float) - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)
vol = {k: nrrd.read(f"data_med/{k}_image.nrrd")[0] for k in ["lung1", "lung2", "brain1", "brain2"]}
sl = []
for v, z in [("lung1", 20), ("lung1", 30), ("lung2", 25), ("lung2", 40)]:      # CT, lung window W=1500 L=-600
    sl.append((f"CT-{v[-1]}-{z}", to8(vol[v][:, :, z].T, -1350, 150)))
for v, z in [("brain1", 10), ("brain1", 15), ("brain2", 9), ("brain2", 14)]:   # MRI, 0.5-99.5 percentile window
    a = vol[v][:, :, z].T; lo, hi = np.percentile(a, [.5, 99.5]); sl.append((f"MR-{v[-1]}-{z}", to8(a, lo, hi)))
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
def fill_text(cap, seed):
    rng = np.random.default_rng(seed); n = int(cap / 6.25)
    while True:
        t = bytes(rng.choice(alph, n)).decode()
        if len(encode_payload(t.encode())) <= cap: return t, len(encode_payload(t.encode()))
        n -= 500
rows = []; keep = {}
for name, I in sl:
    M = I.shape[0]; cap = capacity(M, M); txt, used = fill_text(cap, 11)
    t0 = time.perf_counter(); C, nu = embed(I, txt, K); te = time.perf_counter() - t0
    t0 = time.perf_counter(); rc, rt, rep = extract(C, nu, K); tx = time.perf_counter() - t0
    ok = bool(np.array_equal(rc, I) and rt == txt and rep["authentic"])
    wk = not extract(C, nu, bytes([K[0] ^ 1]) + K[1:])[2]["authentic"]
    T = C.copy(); T[3, 5] ^= 1; tp = not extract(T, nu, K)[2]["authentic"]
    I1 = I.copy(); I1[M//2, M//2] ^= 1; C1 = embed(I1, txt, K)[0]
    t1 = ("X" if txt[0] != "X" else "Y") + txt[1:]; C2 = embed(I, t1, K)[0]
    C3 = embed(I, txt, bytes([K[0] ^ 1]) + K[1:])[0]
    s = [npcr_uaci(C, X) for X in (C1, C2, C3)]
    rows.append(dict(image=name, size=M, cap=cap, used=used, load=100*used/cap, roundtrip=ok, wrongkey=wk, tamper=tp,
                     H_cover=entropy(I, 256), H_stego=entropy(C, 1024), chi2=chi_square(C, 1024),
                     rH=corr(C, 1, 0), rV=corr(C, 0, 1), rD=corr(C, 1, 1),
                     npcr_px=s[0][0], uaci_px=s[0][1], npcr_ch=s[1][0], uaci_ch=s[1][1], npcr_key=s[2][0], uaci_key=s[2][1],
                     t_embed=te, t_extract=tx))
    keep[name] = I; keep[name + "_s"] = C; keep[name + "_r"] = rc
    print(name, M, ok, wk, tp, round(rows[-1]["H_stego"], 4), round(rows[-1]["chi2"], 1), [round(v, 3) for v in s[0] + s[1] + s[2]])
json.dump(rows, open("results/medical.json", "w"), indent=1); np.savez_compressed("results/medical_imgs.npz", **keep)
