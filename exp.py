"""Experiments for the paper. Everything is regenerated from code; results go to results/*.json|csv."""
import os, json, math, time, hashlib, hmac, sys
import numpy as np, pandas as pd
from skimage import data, color, transform
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from core import *
from qcirc import *
from novelty import *

os.makedirs("results", exist_ok=True)
SIM = AerSimulator(method="statevector")
R = {}                                                        # collected scalar results

def run_statevector(qc):
    q = qc.copy(); q.save_statevector()
    return Statevector(SIM.run(transpile(q, SIM, optimization_level=0)).result().get_statevector())

# ------------------------------------------------------------------ BB84 (identical to the notebook)
def bb84(n_raw=2048, eve=False, seed=0, batch=16):
    rng = np.random.default_rng(seed)
    a_bits, a_bas, b_bas = rng.integers(0, 2, n_raw), rng.integers(0, 2, n_raw), rng.integers(0, 2, n_raw)
    e_bas = rng.integers(0, 2, n_raw)
    def transmit(bits, prep_bas, meas_bas, sd):
        out = np.empty(n_raw, int); circs = []
        for s in range(0, n_raw, batch):
            q = QuantumCircuit(batch, batch)
            for i in range(batch):
                if bits[s+i]: q.x(i)
                if prep_bas[s+i]: q.h(i)
                if meas_bas[s+i]: q.h(i)
                q.measure(i, i)
            circs.append(q)
        res = SIM.run(transpile(circs, SIM), shots=1, seed_simulator=sd).result()
        for k, q in enumerate(circs):
            m = list(res.get_counts(k))[0][::-1]
            out[k*batch:(k+1)*batch] = [int(ch) for ch in m]
        return out
    if eve:
        e_bits = transmit(a_bits, a_bas, e_bas, seed+1); b_bits = transmit(e_bits, e_bas, b_bas, seed+2)
    else:
        b_bits = transmit(a_bits, a_bas, b_bas, seed+1)
    keep = a_bas == b_bas
    ka, kb = a_bits[keep], b_bits[keep]
    sample = rng.permutation(ka.size)[: ka.size // 4]
    qber = float((ka[sample] != kb[sample]).mean())
    rest = np.setdiff1d(np.arange(ka.size), sample)
    key = hashlib.sha256(np.packbits(kb[rest]).tobytes()).digest() if qber <= 0.11 else None
    return dict(raw=n_raw, sifted=int(keep.sum()), qber=qber, aborted=key is None, key=key)

r1 = bb84(eve=False, seed=11); r2 = bb84(eve=True, seed=11)
K_QKD = r1["key"]
# QBER statistics over 20 independent runs (honest vs. intercept-resend Eve)
qs = {"honest": [], "eve": []}
for s in range(20):
    qs["honest"].append(bb84(seed=100+s)["qber"]); qs["eve"].append(bb84(eve=True, seed=100+s)["qber"])
R["bb84"] = dict(key_hex=K_QKD.hex(), honest=dict(raw=r1["raw"], sifted=r1["sifted"], qber=r1["qber"]),
                 eve=dict(raw=r2["raw"], sifted=r2["sifted"], qber=r2["qber"], aborted=bool(r2["aborted"])),
                 qber_honest_mean=float(np.mean(qs["honest"])), qber_eve_mean=float(np.mean(qs["eve"])),
                 qber_eve_min=float(np.min(qs["eve"])), qber_eve_max=float(np.max(qs["eve"])),
                 qber_honest_max=float(np.max(qs["honest"])))
json.dump(qs, open("results/qber_runs.json", "w"))
print("BB84", R["bb84"])

# ------------------------------------------------------------------ chaos map: Lyapunov / bifurcation
def lle_and_orbit(a, b=1.0, c=-0.1, d=-0.1, x=7.0, y=0.0, T=1000, K=3000, keep=150):
    v = np.array([1.0, 0.0]); s = 0.0; orb = []
    for k in range(T + K):
        if k >= T:
            J = np.array([[1 + a/(1+y*y), -2*a*x*y/(1+y*y)**2], [c, 1.0]])
            v = J @ v; nv = np.linalg.norm(v); s += math.log(nv); v /= nv
        x, y = rational_map(x, y, a, b, c, d)
        if not (math.isfinite(x) and abs(x) < 1e6 and abs(y) < 1e6): return np.nan, []
        if k >= T + K - keep: orb.append(x)
    return s / K, orb
As = np.linspace(-3.14, -2.44, 300); LL = []; ORB = []
for a_ in As:
    l, orb = lle_and_orbit(a_); LL.append(l); ORB.append(orb)
l28, _ = lle_and_orbit(-2.8, K=100000)
x, y = 7.0, 0.0; xs = []; ys = []
for k in range(21000):
    x, y = rational_map(x, y, -2.8, 1, -0.1, -0.1)
    if k >= 1000: xs.append(x); ys.append(y)
np.savez("results/chaos.npz", As=As, LL=np.array(LL), xs=np.array(xs), ys=np.array(ys),
         orb=np.array([o if len(o) == 150 else [np.nan]*150 for o in ORB]))
R["chaos"] = dict(lle_at_working_point=float(l28), n_divergent=int(np.isnan(LL).sum()), n_scan=len(As))
print("chaos", R["chaos"])

# ------------------------------------------------------------------ gate-level verification
ok = True
for al in range(8):
    for be in range(8):
        for tv in range(8):
            for sv_ in range(8):
                t_, s_ = QuantumRegister(3, "t"), QuantumRegister(3, "s"); q = QuantumCircuit(t_, s_)
                for i in range(3):
                    if tv >> i & 1: q.x(t_[i])
                    if sv_ >> i & 1: q.x(s_[i])
                draper_affine(q, t_, s_, al, be)
                sv = Statevector(q); idx = int(np.argmax(np.abs(sv.data)))
                exp_t = (tv + al*sv_ + be) % 8
                ok &= (idx & 7) == exp_t and ((idx >> 3) & 7) == sv_
R["draper_exact_4096"] = bool(ok)
print("Draper adder exact:", ok)

small = transform.resize(data.camera(), (8, 8), preserve_range=True, anti_aliasing=True).round().astype(np.int64)
msg = "HbA1c 8.1%"
emb_c, enc_c, nonce_q = sender_circuits(small, msg, K_QKD, DEFAULT_CHAOS)
t0 = time.time(); sv = run_statevector(emb_c.compose(enc_c)); t_sv = time.time() - t0
img_q, nbr, spread = statevector_to_image(sv, 4)
cls, nonce_c = embed(small, msg, K_QKD)
R["gate_level"] = dict(qubits=emb_c.num_qubits, gates_prep_embed=emb_c.size(), gates_enc=enc_c.size(),
                       counts={k: int(v) for k, v in emb_c.compose(enc_c).count_ops().items()},
                       branches=int(nbr), amp_spread=float(spread), match=bool(np.array_equal(img_q, cls)),
                       nonce_match=bool(nonce_q == nonce_c), sim_seconds=float(t_sv))
recv = emb_c.compose(enc_c).compose(enc_c.inverse()); recv.measure_all()
shots = 4000
counts = SIM.run(transpile(recv, SIM, optimization_level=0), shots=shots, seed_simulator=7).result().get_counts()
side = 16; meas = -np.ones((side, side), np.int64)
for k in counts:
    i = int(k.replace(" ", ""), 2); meas[i & 15, (i >> 4) & 15] = i >> 8
ok_pos = int((meas >= 0).sum())
rc_small, rt_small, rep_small = extract(np.where(meas >= 0, meas, 0) ^ 0, nonce_c, K_QKD) if False else (None, None, None)
# receiver: E^-1 E |emb> = |emb>, i.e. the *embedded* (pre-encryption) state is read out; decode from it
Pq, Dq = meas >> 2, meas & 3
cover_rec = Pq[0::2, 0::2]
bits = np.empty(2 * side * side, np.uint8); bits[0::2] = Dq.ravel() >> 1; bits[1::2] = Dq.ravel() & 1
text_rec = decode_payload((bits + 48).tobytes().decode()).decode()
R["receiver_readout"] = dict(shots=shots, positions_seen=ok_pos, cover_ok=bool(np.array_equal(cover_rec, small)),
                             text_ok=(text_rec == msg), coupon_expected=float(256 * (math.log(256) + 0.5772)))
print("gate-level", R["gate_level"], R["receiver_readout"])

# ------------------------------------------------------------------ Theorem 1: large randomized verification
rows_t = []; rng = np.random.default_rng(2024)
t0 = time.time()
for (n, q_) in [(2, 8), (3, 8)]:
    Nn = 1 << (2*n)
    for trial in range(60 if n == 3 else 60):
        E, (y_, x_, c_, a_) = random_fixed_key_cipher(n, q_, seed=1000*n + trial)
        side_ = 1 << n
        I = rng.integers(0, 256, (side_, side_)); d = int(rng.integers(1, Nn + 1)); J = I.copy()
        for p in rng.choice(Nn, d, replace=False): J.flat[p] = (J.flat[p] + int(rng.integers(1, 256))) % 256
        outs = []
        for img in (I, J):
            qc_ = QuantumCircuit(y_, x_, c_, a_); neqr_prep_full(qc_, y_, x_, c_, img); qc_.compose(E, inplace=True)
            o, clean = output_image(run_statevector(qc_), n, q_); assert clean and o is not None; outs.append(o)
        rows_t.append(dict(n=n, N=Nn, gates=E.size(), changed_in=d, changed_out=int((outs[0] != outs[1]).sum())))
tt = pd.DataFrame(rows_t); tt.to_csv("results/theorem1_trials.csv", index=False)
R["theorem1"] = dict(trials=len(tt), equal=int((tt.changed_in == tt.changed_out).sum()),
                     gates_min=int(tt.gates.min()), gates_max=int(tt.gates.max()), seconds=time.time()-t0)
print("Theorem 1:", R["theorem1"])

# ------------------------------------------------------------------ images, text, main experiments (re-run => matches CSV)
def load_gray(name):
    im = getattr(data, name)()
    if im.ndim == 3: im = color.rgb2gray(im[..., :3]) * 255
    return np.clip(np.rint(transform.resize(im, (512, 512), preserve_range=True, anti_aliasing=True)), 0, 255).astype(np.int64)
names = ["camera", "astronaut", "moon", "coffee", "chelsea", "rocket", "brick", "gravel"]
IM = {n: load_gray(n) for n in names}
np.savez_compressed("results/images.npz", **IM)
cap = capacity(512, 512)
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
rng = np.random.default_rng(7); n_chars = 340000
while True:
    max_text = bytes(rng.choice(alph, n_chars)).decode()
    used = len(encode_payload(max_text.encode()))
    if used <= cap: break
    n_chars -= 1000
R["payload"] = dict(cap=cap, chars=n_chars, used=used)
STEGO = {}
for n in ["camera", "astronaut", "moon"]:
    Cst, nc = embed(IM[n], max_text, K_QKD); rc, rt, rep = extract(Cst, nc, K_QKD); STEGO[n] = (Cst, rc)
np.savez_compressed("results/stego3.npz", **{f"{n}_s": STEGO[n][0] for n in STEGO}, **{f"{n}_r": STEGO[n][1] for n in STEGO})

# ------------------------------------------------------------------ ablation (camera)
def embed_variant(cover, text, qkd_key, scramble_on=True, xor_on=True, siv_on=True, fixed_nonce=b"\x00"*16):
    M = cover.shape[0]; side, n = 2*M, int(math.log2(2*M)); data_ = text.encode()
    K = master_key(qkd_key, DEFAULT_CHAOS)
    nonce = siv_nonce(K, cover, data_) if siv_on else fixed_nonce
    S = session(K, nonce); bits = encode_payload(data_); capv = capacity(M, M)
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (capv-len(bits)+7)//8), np.uint8))[: capv-len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (2*b[0::2] + b[1::2]).reshape(side, side)
    E = np.repeat(np.repeat(cover.astype(np.int64), 2, 0), 2, 1); V = 4*E + D
    if scramble_on:
        Y, X = np.indices((side, side)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
        C = np.empty_like(V); C[Y2, X2] = V[Y, X]
    else: C = V.copy()
    return (C ^ keystream(S, side)) if xor_on else C
cv = IM["camera"]; c1 = cv.copy(); c1[256, 256] ^= 1
abl = []
for name, kw in [("Full QC-RDH", {}), ("no position scrambling", dict(scramble_on=False)),
                 ("no key oracle (scrambling only)", dict(xor_on=False)),
                 ("no synthetic nonce (fixed key)", dict(siv_on=False)),
                 ("no scrambling, no key oracle", dict(scramble_on=False, xor_on=False))]:
    A = embed_variant(cv, max_text, K_QKD, **kw); B = embed_variant(c1, max_text, K_QKD, **kw)
    npcr, uaci = npcr_uaci(A, B)
    abl.append(dict(variant=name, entropy=entropy(A, 1024), chi2=chi_square(A, 1024), r_H=corr(A, 1, 0), r_V=corr(A, 0, 1),
                    NPCR_1px=npcr, UACI_1px=uaci, changed_px=int((A != B).sum())))
pd.DataFrame(abl).to_csv("results/ablation.csv", index=False); print(pd.DataFrame(abl).round(4))

# ------------------------------------------------------------------ chosen-cover attack: fixed key vs SIV (also NEQRX-style)
victim = IM["camera"]; c1 = victim.copy(); c1[100, 100] ^= 1
fk = lambda cvr, txt="fixed": fixed_key_embed(cvr, txt, K_QKD)
R["fixed_key_changed"] = int((fk(victim) != fk(c1)).sum())
siv0, siv1 = embed(victim, "fixed", K_QKD)[0], embed(c1, "fixed", K_QKD)[0]
R["siv_changed"] = [int((siv0 != siv1).sum()), int(siv0.size)]
secret_text = "Patient: R. Sharma, 61F | Dx: stage II breast carcinoma | Rx: letrozole 2.5 mg od"
dec_fixed, nq = chosen_cover_attack(lambda cvr: fixed_key_embed(cvr, "attacker text", K_QKD), 512)
rec_fixed = dec_fixed(fixed_key_embed(victim, secret_text, K_QKD))
dec_siv, _ = chosen_cover_attack(lambda cvr: embed(cvr, "attacker text", K_QKD)[0], 512)
rec_siv = dec_siv(embed(victim, secret_text, K_QKD)[0])
R["attack_fixed"] = dict(queries=int(nq), exact=bool(np.array_equal(rec_fixed, victim)))
R["attack_siv"] = dict(correct_pct=float(100*(rec_siv == victim).mean()), chance_pct=100/256)

def logistic_key(M, L0=0.5557924316949603, mu=3.99):
    x = L0; ks = np.empty(M*M, np.int64)
    for k in range(M*M):
        x = mu*x*(1-x); ks[k] = int(x*1e14) % 256
    return ks.reshape(M, M)
KS_NX = logistic_key(512)
def neqrx_encrypt(I, s=5, t=3, P=77, Q=141):
    M = I.shape[0]; Y, X = np.indices((M, M)); E = np.empty_like(I)
    E[(t*Y+Q) % M, (s*X+P) % M] = I
    return E ^ KS_NX
def neqrx_embed(E, bits, a=7, b=19):
    M = E.shape[0]; Mx = np.zeros(M*M, np.int64); Mx[:len(bits)] = bits; Mx = Mx.reshape(M, M)
    Y, X = np.indices((M, M)); Ms = np.empty_like(Mx); Ms[(a*Y+b) % M, (a*X+b) % M] = Mx
    return E ^ Ms
p1 = victim.copy(); p1[100, 100] ^= 1
d_ = int((neqrx_encrypt(victim) != neqrx_encrypt(p1)).sum())
def perm_xor_attack(enc, M):
    Z = enc(np.zeros((M, M), np.int64)); idx = np.arange(M*M).reshape(M, M); owner = np.zeros_like(Z)
    nq_ = math.ceil(math.log2(M*M) / 8)
    for j in range(nq_): owner |= (enc((idx >> (8*j)) & 255) ^ Z) << (8*j)
    def dec(C):
        rec = np.zeros(M*M, np.int64); rec[owner.ravel()] = (C ^ Z).ravel(); return rec.reshape(M, M)
    return dec, 1 + nq_
dec_nx, nq_nx = perm_xor_attack(neqrx_encrypt, 512)
hidden_bits = np.unpackbits(np.frombuffer(secret_text.encode(), np.uint8)).astype(np.int64)
marked = neqrx_embed(neqrx_encrypt(victim), hidden_bits); rec_nx = dec_nx(marked)
mse = ((rec_nx - victim)**2).mean(); psnr_nx = float("inf") if mse == 0 else 10*np.log10(255**2/mse)
R["neqrx"] = dict(changed=d_, npcr=100*d_/victim.size, ceiling=100/victim.size, queries=int(nq_nx),
                  msb7_exact=bool(np.array_equal(rec_nx >> 1, victim >> 1)), psnr=float(psnr_nx))
np.savez_compressed("results/attack_imgs.npz", victim=victim, rec_fixed=rec_fixed, rec_siv=rec_siv, marked=marked, rec_nx=rec_nx)
print({k: R[k] for k in ["fixed_key_changed", "siv_changed", "attack_fixed", "attack_siv", "neqrx"]})

# ------------------------------------------------------------------ quantum resources (512x512, n = 10)
n = 10; Cst, nc = embed(IM["camera"], max_text, K_QKD)
S_ = session(master_key(K_QKD), nc); ks = keystream(S_, 1024)
ones_cover = int(np.unpackbits(IM["camera"].astype(np.uint8)).sum())
ones_key = int(np.unpackbits(ks.astype(">u2").view(np.uint8)).sum())
qft = n*(n-1)//2 + n
R["resources"] = dict(qubits=2*n+NBITS_C, ones_cover=ones_cover, ones_key=ones_key,
                      writeD_est=int(used*.5 + (cap-used)*.5), shear_gates=int(2*ROUNDS*(2*qft + n*n + n)),
                      shots_all_positions=float(2**20 * (math.log(2**20) + 0.5772)))
json.dump(R, open("results/results.json", "w"), indent=1, default=str)
print("saved")
