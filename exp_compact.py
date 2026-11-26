"""Parameterised QC-RDH(s, k): enlargement factor s (1 = none, 2 = baseline) and k embedded bits per stego pixel.
Runs the compact configurations on the 16 main test images with full payloads: exact recovery, wrong-key and
tamper rejection, and stego statistics (entropy against a uniform Monte Carlo reference, chi^2, NPCR, UACI, r)."""
import json, math, numpy as np
from scipy.stats import chi2 as CHI2
from core import *
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"]); Kw = bytes([K[0] ^ 1]) + K[1:]
A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
TXT = {}
def text_for(cap):
    if cap not in TXT:
        rng = np.random.default_rng(7); n = int(cap / 6.25)
        while True:
            t = bytes(rng.choice(alph, n)).decode()
            if len(encode_payload(t.encode())) <= cap: TXT[cap] = t; break
            n -= max(100, n // 300)
    return TXT[cap]
out = {}
for kb, sc in [(8, 1), (4, 1), (2, 1)]:
    rows = []
    for name, I in imgs:
        M = I.shape[0]; side = sc * M; cap = kb * side * side; L = 1 << (8 + kb); txt = text_for(cap)
        C, nu = embed(I, txt, K, kb=kb, scale=sc); rc, rt, rep = extract(C, nu, K, kb=kb, scale=sc)
        ok = bool(np.array_equal(rc, I) and rt == txt and rep["authentic"])
        wk = not extract(C, nu, Kw, kb=kb, scale=sc)[2]["authentic"]
        T = C.copy(); T[3, 5] ^= 1; tp = not extract(T, nu, K, kb=kb, scale=sc)[2]["authentic"]
        I1 = I.copy(); I1[M // 2, M // 2] ^= 1; C1 = embed(I1, txt, K, kb=kb, scale=sc)[0]
        np1, ua1 = npcr_uaci(C, C1, L)
        ref = np.random.default_rng(len(rows)).integers(0, L, C.shape)          # ideal uniform source, same size
        rows.append(dict(image=name, cover=M, stego=side, cap=cap, load=100 * len(encode_payload(txt.encode())) / cap,
                         roundtrip=ok, wrongkey=wk, tamper=tp, H=entropy(C, L), H_uniform_ref=entropy(ref, L),
                         chi2=chi_square(C, L), chi2_crit=float(CHI2.ppf(.95, L - 1)), npcr=np1, uaci=ua1,
                         rmax=max(abs(corr(C, 1, 0)), abs(corr(C, 0, 1)), abs(corr(C, 1, 1)))))
    out[f"s{sc}_k{kb}"] = rows
    r = rows; print(kb, sc, all(x["roundtrip"] and x["wrongkey"] and x["tamper"] for x in r),
                    "H-Href", max(abs(x["H"] - x["H_uniform_ref"]) for x in r), "chi pass", sum(x["chi2"] < x["chi2_crit"] for x in r),
                    "npcr", min(x["npcr"] for x in r), max(x["npcr"] for x in r), "uaci", min(x["uaci"] for x in r), max(x["uaci"] for x in r),
                    "rmax", max(x["rmax"] for x in r), "load", min(x["load"] for x in r), flush=True)
json.dump(out, open("results/compact.json", "w"), indent=1)
