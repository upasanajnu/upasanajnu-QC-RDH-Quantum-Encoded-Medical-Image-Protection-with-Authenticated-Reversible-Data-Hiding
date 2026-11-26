"""Consistency audit: recomputes from scratch the numbers quoted in the paper and compares them
with the stored results (reproduction of the main CSV, ideal values, resource arithmetic)."""
import json, math, numpy as np, pandas as pd
from scipy.stats import chi2 as CHI2
from core import *
out = {}
df = pd.read_csv("results/qcrdh_results.csv").set_index("Image")
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
IM = np.load("results/images.npz"); cap = capacity(512, 512)
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
rng = np.random.default_rng(7); n = 340000
while True:
    t = bytes(rng.choice(alph, n)).decode()
    if len(encode_payload(t.encode())) <= cap: break
    n -= 1000
md = 0.0; ok = True
for k in IM:
    I = IM[k].astype(np.int64); C, nu = embed(I, t, K); rc, rt, rep = extract(C, nu, K)
    ok &= bool(np.array_equal(rc, I) and rt == t and rep["authentic"])
    I1 = I.copy(); I1[256, 256] ^= 1; C1 = embed(I1, t, K)[0]; np1, ua1 = npcr_uaci(C, C1)
    v = dict(H_stego=entropy(C, 1024), chi2_stego=chi_square(C, 1024), r_H=corr(C, 1, 0), r_V=corr(C, 0, 1),
             r_D=corr(C, 1, 1), H_cover=entropy(I, 256), NPCR_1px=np1, UACI_1px=ua1)
    md = max(md, max(abs(v[c] - df.loc[k, c]) for c in v))
out["main_table_reproduced"] = dict(exact_recovery_all=ok, max_abs_diff=md)
L, Np = 1024, 1 << 20
out["ideal"] = dict(NPCR=100 * (1 - 1 / L), UACI=100 * (L + 1) / (3 * L),
                    H_expected_2e20=10 - (L - 1) / (2 * Np * math.log(2)), H_expected_2e18=10 - (L - 1) / (2 * (Np >> 2) * math.log(2)),
                    chi2_crit_95=float(CHI2.ppf(0.95, L - 1)))
r5 = json.load(open("results/resources512.json")); c = r5["counts"]; P = json.load(open("results/mcx_cost.json"))["per_gate"]
el = c["prep"] * P["18"]["one_ancilla"]["total"] + (c["writeD"] + c["key"]) * P["20"]["one_ancilla"]["total"]
open_ctrl = 2 * (18 * c["prep"] + 20 * (c["writeD"] + c["key"]))          # X before and after each open control
H = sum(1 / j for j in range(1, Np + 1))
out["resources"] = dict(mcx=sum(c.values()), logical_ops=sum(c.values()) + 18 + 2 + 18 + r5["shear_ops"],
                        elementary_1anc=el, elementary_1anc_open_bound=el + open_ctrl,
                        mcx_upper_bound_8N_12Np=8 * (Np >> 2) + 12 * Np, mcx_within_bound=sum(c.values()) <= 8 * (Np >> 2) + 12 * Np,
                        copies=Np * H, qubits_sent=30 * Np * H, payload_bits=2 * Np, bits_per_qubit=2 * Np / (30 * Np * H),
                        payload_per_transmitted_bit=2 / 10, bpp_stego=2, bpp_cover=8)
json.dump(out, open("results/recheck.json", "w"), indent=1); print(json.dumps(out, indent=1))
