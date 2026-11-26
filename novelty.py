# =====================================================================================
#  Novelty module: (1) gate-level verification of the Unitary Diffusion Bound,
#                  (2) chosen-cover attack on fixed-key quantum RDH and its failure under SIV.
# =====================================================================================
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qcirc import draper_affine
from core import (embed, capacity, master_key, session, round_constants, keystream,
                  scramble, NBITS_C)

# ---------------------------------------------------------------- (1) Theorem verification
def neqr_prep_full(qc, y, x, c, img):
    n = len(y); qc.h(list(y) + list(x))
    for Y in range(img.shape[0]):
        for X in range(img.shape[1]):
            st = Y + (X << n)
            for k in range(len(c)):
                if (int(img[Y, X]) >> k) & 1: qc.mcx(list(y) + list(x), c[k], ctrl_state=st)

def random_fixed_key_cipher(n, q, seed, rounds=2, n_xor=12):
    """A random 'quantum image cipher' built from every gate family used in the literature:
    Draper/affine position scrambling, key-image XOR (position-controlled X), per-pixel
    nonlinear substitution with an ancilla (Toffoli compute/uncompute), intensity-qubit CNOT
    'diffusion', position-controlled SWAPs inside the intensity register. Key = seed (fixed)."""
    r = np.random.default_rng(seed)
    y, x, c, a = QuantumRegister(n, "y"), QuantumRegister(n, "x"), QuantumRegister(q, "c"), QuantumRegister(1, "anc")
    qc = QuantumCircuit(y, x, c, a, name="E_K")
    m = 1 << n
    for _ in range(rounds):
        draper_affine(qc, x, y, int(r.integers(1, m)), int(r.integers(0, m)))
        draper_affine(qc, y, x, int(r.integers(1, m)), int(r.integers(0, m)))
        for _ in range(n_xor):                                    # key image XOR
            qc.mcx(list(y) + list(x), c[int(r.integers(q))], ctrl_state=int(r.integers(0, m * m)))
        i, j, k = r.choice(q, 3, replace=False)                   # nonlinear S-box step via ancilla
        qc.ccx(c[i], c[j], a[0]); qc.cx(a[0], c[k]); qc.ccx(c[i], c[j], a[0])
        i, j = r.choice(q, 2, replace=False); qc.cx(c[i], c[j])   # intra-pixel 'diffusion'
        i, j = r.choice(q, 2, replace=False); qc.cswap(y[int(r.integers(n))], c[i], c[j])
    return qc, (y, x, c, a)

def output_image(sv, n, q):
    """Decode an output statevector. Returns (image or None if not a valid NEQR image, ancilla_clean)."""
    side = 1 << n; img = -np.ones((side, side), np.int64); ok = True
    nz = np.flatnonzero(np.abs(sv.data) > 1e-9); anc_clean = True
    for idx in nz:
        Y = idx & (side - 1); X = (idx >> n) & (side - 1); C = (idx >> (2 * n)) & ((1 << q) - 1)
        if idx >> (2 * n + q): anc_clean = False
        if img[Y, X] >= 0: ok = False                              # two branches on one position
        img[Y, X] = C
    if (img < 0).any(): ok = False
    return (img if ok else None), anc_clean

# ---------------------------------------------------------------- (2) chosen-cover attack
def chosen_cover_attack(oracle, M):
    """Attacker submits covers of its choice to a fixed-key embedding oracle (same text each time).
    Uses 1 + ceil(log2(M*M)/8) chosen covers. Returns a decryptor for any stego made with that key."""
    nbits = int(math.log2(M * M)); nq = math.ceil(nbits / 8)
    C0 = oracle(np.zeros((M, M), np.int64))
    idx = np.arange(M * M).reshape(M, M)
    owner = np.zeros_like(C0)                                       # which cover pixel lands at each output position
    for j in range(nq):
        cov = (idx >> (8 * j)) & 255
        Cj = oracle(cov)
        owner |= (((Cj ^ C0) >> 2) & 255) << (8 * j)                # (4E|D) xor (0|D) = 4E  -> reveals E = index byte
    def decrypt_cover(C_victim):
        diff = C_victim ^ C0                                        # = scramble(4E_v | D_v) xor scramble(D_0)
        rec = np.zeros(M * M, np.int64)
        rec[owner.ravel() % (M * M)] = (diff.ravel() >> 2) & 255      # high 8 bits = victim cover pixel
        return rec.reshape(M, M)
    return decrypt_cover, 1 + nq

def fixed_key_embed(cover, text, qkd_key, nonce=b"\x00" * 16):
    """The SAME scheme but with a fixed (plaintext-independent) nonce, i.e. a fixed-key unitary,
    which is the design used by fixed-key quantum image ciphers / quantum RDH schemes."""
    import core
    M = cover.shape[0]; side = 2 * M; n = int(math.log2(side))
    K = master_key(qkd_key); S = session(K, nonce)
    data = text.encode(); bits = core.encode_payload(data); cap = capacity(M, M)
    pad = np.unpackbits(np.frombuffer(core.xof(S, b"pad", (cap - len(bits) + 7) // 8), np.uint8))[: cap - len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (2 * b[0::2] + b[1::2]).reshape(side, side)
    V = 4 * np.repeat(np.repeat(cover.astype(np.int64), 2, 0), 2, 1) + D
    Y, X = np.indices((side, side)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
    C = np.empty_like(V); C[Y2, X2] = V[Y, X]
    return C ^ keystream(S, side)
