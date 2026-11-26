# =====================================================================================
#  Gate-level quantum circuits of QC-RDH (Qiskit). Register layout (little-endian):
#    y[0..n-1]  row index   (y[0] = LSB, added by the scaling step)
#    x[0..n-1]  column index
#    c[0..9]    intensity  (extended NEQR: 8 cover bits + 2 embedding qubits, initially |0>)
# =====================================================================================
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from core import (NBITS_C, master_key, siv_nonce, session, round_constants, keystream,
                  encode_payload, xof, capacity)

def registers(n):
    return QuantumRegister(n, "y"), QuantumRegister(n, "x"), QuantumRegister(NBITS_C, "c")

def _pos_ctrl(y, x):           # control list and ctrl_state for an explicit position |y>|x>
    return list(y) + list(x)

# ---- (1) NEQR preparation of the M x M cover on the upper position qubits ----------------
def neqr_prep(qc, y, x, c, cover):
    up = list(y[1:]) + list(x[1:]); m = len(y) - 1
    qc.h(up)
    for Y in range(cover.shape[0]):
        for X in range(cover.shape[1]):
            st = Y + (X << m)
            for k in range(8):
                if (int(cover[Y, X]) >> k) & 1: qc.mcx(up, c[k], ctrl_state=st)
    return qc

# ---- (2) quantum nearest-neighbour scaling 2^(n-1) -> 2^n : one H per axis ----------------
def scale_up(qc, y, x):
    qc.h(y[0]); qc.h(x[0]); return qc

# ---- (3) embedding: |c> -> |4c> (two cyclic qubit rotations) then |4c> -> |4c + D(p)> ------
def shift_by_4(qc, c):
    for _ in range(2):
        for k in range(NBITS_C - 1, 0, -1): qc.swap(c[k], c[k - 1])
    return qc

def write_D(qc, y, x, c, D):
    ctrl = _pos_ctrl(y, x); n = len(y)
    for Y in range(D.shape[0]):
        for X in range(D.shape[1]):
            st = Y + (X << n)
            if D[Y, X] & 1: qc.mcx(ctrl, c[0], ctrl_state=st)
            if D[Y, X] & 2: qc.mcx(ctrl, c[1], ctrl_state=st)
    return qc

# ---- (4) position scrambling: Draper QFT adders  |t> -> |t + alpha*s + beta  mod 2^n> ------
def draper_affine(qc, tgt, src, alpha, beta):
    n = len(tgt); N = 1 << n
    qc.append(QFTGate(n), tgt)
    for i in range(n):
        if beta: qc.p(2 * math.pi * beta * (1 << i) / N, tgt[i])
        for j in range(len(src)):
            ang = 2 * math.pi * alpha * (1 << j) * (1 << i) / N
            if (alpha * (1 << j) * (1 << i)) % N: qc.cp(ang, src[j], tgt[i])
    qc.append(QFTGate(n).inverse(), tgt)
    return qc

def scramble_circuit(qc, y, x, consts):
    for al, be, ga, de in consts:
        draper_affine(qc, x, y, al, be)      # x += alpha*y + beta
        draper_affine(qc, y, x, ga, de)      # y += gamma*x + delta
    return qc

# ---- (5) key oracle  |p>|c> -> |p>|c xor k(p)> ----------------------------------------------
def key_oracle(qc, y, x, c, ks):
    ctrl = _pos_ctrl(y, x); n = len(y)
    for Y in range(ks.shape[0]):
        for X in range(ks.shape[1]):
            st = Y + (X << n)
            for k in range(NBITS_C):
                if (int(ks[Y, X]) >> k) & 1: qc.mcx(ctrl, c[k], ctrl_state=st)
    return qc

# ---- full sender circuit for a small cover ---------------------------------------------------
def sender_circuits(cover, text, qkd_key, chaos):
    """Returns (prep_and_embed circuit, encryption circuit, nonce). Same key derivation as core.embed."""
    M = cover.shape[0]; side = 2 * M; n = int(math.log2(side))
    data = text.encode("utf-8")
    K = master_key(qkd_key, chaos); nonce = siv_nonce(K, cover, data); S = session(K, nonce)
    bits = encode_payload(data); cap = capacity(M, M)
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (cap - len(bits) + 7) // 8), np.uint8))[: cap - len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (2 * b[0::2] + b[1::2]).reshape(side, side)
    y, x, c = registers(n)
    emb = QuantumCircuit(y, x, c, name="PREP+EMBED")
    neqr_prep(emb, y, x, c, cover); emb.barrier(); scale_up(emb, y, x); emb.barrier()
    shift_by_4(emb, c); emb.barrier(); write_D(emb, y, x, c, D)
    enc = QuantumCircuit(y, x, c, name="ENC")
    scramble_circuit(enc, y, x, round_constants(S, n)); enc.barrier()
    key_oracle(enc, y, x, c, keystream(S, side))
    return emb, enc, nonce

def statevector_to_image(sv, n):
    """Read an NEQR statevector: every non-zero amplitude is one (position, intensity) branch."""
    side = 1 << n; img = -np.ones((side, side), np.int64); amps = sv.data
    nz = np.flatnonzero(np.abs(amps) > 1e-9)
    for idx in nz:
        Y = idx & (side - 1); X = (idx >> n) & (side - 1); C = idx >> (2 * n)
        img[Y, X] = C
    return img, nz.size, float(np.abs(amps[nz]).std())

def measure_image(sv, n, shots, seed=1):
    """Shot-based read-out: each shot returns one (position, intensity) pair exactly."""
    side = 1 << n; img = -np.ones((side, side), np.int64)
    sv = Statevector(sv.data); sv.seed(seed)
    counts = sv.sample_counts(shots)
    for key in counts:
        idx = int(key, 2); img[idx & (side - 1), (idx >> n) & (side - 1)] = idx >> (2 * n)
    return img, int((img >= 0).sum())
