"""Numerical audit of every formula / lemma stated in the paper. Output: results/checks.json"""
import json, math, itertools, numpy as np
from collections import Counter
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, Operator
from core import *
from qcirc import *
C = {}
rng = np.random.default_rng(5)

# E1 shears: bijection on Z_{2^n}^2, inverse formula, composition (random constants, all n in 2..10)
ok = True
for n in range(2, 11):
    L = 1 << n; Y, X = np.indices((L, L))
    cs = [tuple(int(v) for v in rng.integers(0, L, 4)) for _ in range(ROUNDS)]
    Y2, X2 = scramble(Y, X, cs, n)
    ok &= len(np.unique(Y2 * L + X2)) == L * L                                  # bijection
    Y3, X3 = unscramble(Y2, X2, cs, n); ok &= bool((Y3 == Y).all() and (X3 == X).all())
C["E1_shears_bijective_and_invertible_n2_10"] = bool(ok)

# E2 shear determinant = 1 (unimodular) => bijection also for odd moduli
C["E2_shear_det"] = [int(round(np.linalg.det(np.array([[1, a], [0, 1]])))) for a in (3, 7)]

# E3 U_emb as a unitary: |c> -> |4c> is a permutation of 10-bit basis states restricted to c<256 and extends to a
#     unitary (cyclic left shift by 2 of 10 qubits). Check by building the circuit and the full 2^10 permutation.
q = QuantumCircuit(NBITS_C); shift_by_4(q, range(NBITS_C))
perm = np.array([int(np.argmax(np.abs(Statevector.from_int(c, 1 << NBITS_C).evolve(q).data))) for c in range(1 << NBITS_C)])
C["E3_shift_is_permutation"] = bool(len(set(perm)) == 1024)
C["E3_shift_equals_4c_for_c_lt_256"] = bool(all(perm[c] == 4 * c for c in range(256)))
C["E3_shift_is_cyclic_rotation_all"] = bool(all(perm[c] == ((c << 2) | (c >> 8)) & 1023 for c in range(1024)))

# E4 write-D and key oracle are permutations of the computational basis and involutions (n=1: 12 qubits, ALL 4096 basis states)
y, x, c = registers(1); D = rng.integers(0, 4, (2, 2)); ks = rng.integers(0, 1024, (2, 2))
for name, fn, arg in [("writeD", write_D, D), ("keyoracle", key_oracle, ks)]:
    qq = QuantumCircuit(y, x, c); fn(qq, y, x, c, arg)
    img = []; 
    for b_ in range(1 << 12):
        out = Statevector.from_int(b_, 1 << 12).evolve(qq).data; nz = np.flatnonzero(np.abs(out) > 1e-9)
        assert len(nz) == 1 and abs(abs(out[nz[0]]) - 1) < 1e-9; img.append(int(nz[0]))
    img = np.array(img)
    C[f"E4_{name}_unitary_permutation"] = bool(len(set(img)) == 4096)
    C[f"E4_{name}_involution"] = bool((img[img] == np.arange(4096)).all())
# E5 Draper adder is unitary and inverse is negated phases (n=3 operator check)
t, s = QuantumRegister(3, "t"), QuantumRegister(3, "s"); qq = QuantumCircuit(t, s); draper_affine(qq, t, s, 5, 3)
Uo = Operator(qq).data; C["E5_draper_unitary"] = bool(np.allclose(Uo.conj().T @ Uo, np.eye(64)))
qi = QuantumCircuit(t, s); draper_affine(qi, t, s, (-5) % 8, (-3) % 8)
C["E5_draper_inverse_is_negated_constants"] = bool(np.allclose(Operator(qi).data @ Uo, np.eye(64)))

# E6 NEQR inner product identity <I|J> = (N - d_H)/N, via statevectors (n=2, random images)
def neqr_state(img, n):
    N = img.size; sv = np.zeros(N * 256); side = 1 << n
    for Y in range(side):
        for X in range(side): sv[img[Y, X] * N + (Y + (X << n))] = 1 / math.sqrt(N)
    return sv
ok6 = True
for _ in range(200):
    I = rng.integers(0, 256, (4, 4)); J = I.copy(); d = int(rng.integers(0, 17))
    for p in rng.choice(16, d, replace=False): J.flat[p] = (J.flat[p] + int(rng.integers(1, 256))) % 256
    ok6 &= abs(neqr_state(I, 2) @ neqr_state(J, 2) - (16 - int((I != J).sum())) / 16) < 1e-12
C["E6_neqr_overlap_identity_200"] = bool(ok6)

# E7 canonical Huffman: Kraft equality, prefix-free, header length formula, round trip (200 random texts)
ok7 = True; hdr_ok = True
for _ in range(200):
    nsym = int(rng.integers(1, 60)); alph = rng.choice(256, nsym, replace=False)
    p = rng.dirichlet(np.ones(nsym) * rng.uniform(.2, 3)); length = int(rng.integers(50, 3000))
    data = bytes(rng.choice(alph, length, p=p).astype(np.uint8)); ln = {k: v for k, v in _huff_lengths(data).items()} if False else None
    from core import _huff_lengths, _canonical
    ln = _huff_lengths(data); codes = _canonical(ln)
    if len(ln) > 1: ok7 &= abs(sum(2.0 ** -l for l in ln.values()) - 1) < 1e-12              # complete code (Kraft equality)
    cw = list(codes.values()); ok7 &= not any(a != b and b.startswith(a) for a in cw for b in cw)  # prefix free
    bits = encode_payload(data); ns = len(ln)
    body = sum(ln[b] for b in data); hdr_ok &= len(bits) == H_NSYM + ns * (H_SYM + H_LEN) + H_PLEN + body
    ok7 &= decode_payload(bits) == data
C["E7_huffman_kraft_prefix_roundtrip_200"] = bool(ok7); C["E7_header_length_formula_200"] = bool(hdr_ok)

# E8 end-to-end correctness (Theorem 2) on random small covers / texts / keys, incl. wrong-key and tamper rejection
ok8 = True; rej_key = 0; rej_tam = 0; trials = 150; ran = 0; abort_ok = 0
for t_ in range(trials):
    M = int(rng.choice([8, 16, 32])); cover = rng.integers(0, 256, (M, M))
    alph = bytes(range(32, 127)); qk = bytes(rng.integers(0, 256, 32).astype(np.uint8))
    L_ = int(rng.integers(1, M * M // 2))
    while True:                                   # shrink text until the Huffman bitstream fits the capacity 8N
        txt = bytes(rng.choice(list(alph), L_)).decode()
        try: Cst, nc = embed(cover, txt, qk); break
        except ValueError: L_ = max(1, L_ // 2)
    ran += 1; rc, rt, rep = extract(Cst, nc, qk); ok8 &= bool(np.array_equal(rc, cover) and rt == txt and rep["authentic"])
    qk2 = bytes([qk[0] ^ 1]) + qk[1:]; rep2 = extract(Cst, nc, qk2)[2]; rej_key += not rep2["authentic"]
    T = Cst.copy(); i, j = rng.integers(0, Cst.shape[0], 2); T[i, j] ^= int(rng.integers(1, 1024)); rej_tam += not extract(T, nc, qk)[2]["authentic"]
    big = bytes(rng.choice(list(alph), 8 * M * M)).decode()      # payload certainly above capacity -> must abort
    try: embed(cover, big, qk)
    except ValueError: abort_ok += 1
C["E8_capacity_abort"] = abort_ok
C["E8_roundtrip_ok"] = bool(ok8); C["E8_wrong_key_rejected"] = rej_key; C["E8_tamper_rejected"] = rej_tam; C["E8_trials_run"] = ran; C["E8_trials_attempted"] = trials

# E9 no overflow: max of 4c + D is 1023
C["E9_max_value"] = 4 * 255 + 3

# E10 expectations for random 10-bit data: NPCR, UACI, entropy bias, chi-square, correlation sigma
q_ = 1024; Np = 1 << 20
C["E10_npcr_ideal"] = 100 * (1 - 1 / q_)
X = rng.integers(0, q_, 2_000_000); Y = rng.integers(0, q_, 2_000_000)
C["E10_uaci_ideal_formula"] = 100 * (q_ + 1) / (3 * q_)
C["E10_uaci_monte_carlo"] = float(100 * np.abs(X - Y).mean() / (q_ - 1))
C["E10_entropy_expected_finite_sample"] = 10 - (q_ - 1) / (2 * Np * math.log(2))
H = [entropy(rng.integers(0, q_, Np), q_) for _ in range(5)]; C["E10_entropy_monte_carlo_mean"] = float(np.mean(H))
C["E10_chi2_mean_sd"] = [q_ - 1, math.sqrt(2 * (q_ - 1))]; from scipy.stats import chi2 as _c2; C["E10_chi2_crit_5pct"] = float(_c2.ppf(.95, q_ - 1))
C["E10_corr_sigma"] = 1 / math.sqrt(Np)

# E11 coupon collector
C["E11_expected_shots_256"] = 256 * (math.log(256) + 0.5772156649)
sim = [];
for _ in range(2000):
    seen = set(); k = 0
    while len(seen) < 256: seen.add(int(rng.integers(0, 256))); k += 1
    sim.append(k)
C["E11_simulated_mean_256"] = float(np.mean(sim)); C["E11_exact_H_formula_256"] = float(256 * sum(1 / k for k in range(1, 257)))

# E12 attack query count q + ceil(log_q N)
C["E12_queries"] = {"xor_512": 1 + math.ceil(math.log2(512 * 512) / 8), "general_512": 256 + math.ceil(math.log2(512 * 512) / 8)}

# E13 intercept-resend QBER = 1/4 (analytic: 1/2 * 1/2)
C["E13_qber_intercept_resend"] = .25

# E14 capacity and rates
M = 512; N = M * M
C["E14_capacity_bits"] = capacity(M, M); C["E14_bits_per_cover_pixel"] = capacity(M, M) / N
C["E14_bits_per_stego_pixel"] = capacity(M, M) / (4 * N); C["E14_per_transmitted_bit"] = capacity(M, M) / (4 * N * 10)
json.dump(C, open("results/checks.json", "w"), indent=1); print(json.dumps(C, indent=1))
