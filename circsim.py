"""Exact logical-gate simulation of the FULL-SIZE QC-RDH circuits (30 qubits for a 512x512 cover).

The same qcirc.py functions that build the Qiskit circuits for the 18-qubit statevector runs are
executed here gate by gate on a branch simulator. The NEQR state sum_p |p>|c(p)> / sqrt(N') is
stored exactly as the list of its basis branches (one 30-bit integer per position), with the common
amplitude tracked separately. This is exact, not an approximation, because every gate keeps the
state in this form:
  * H is applied only to qubits that are |0> in every branch (it doubles the branches),
  * X-type gates (MCX with any ctrl_state, SWAP) permute basis states,
  * each Draper adder block (QFT, phase gates, inverse QFT) is simulated as a unitary on the
    2n position qubits with Qiskit's Statevector, checked to be an exact permutation of the
    2^(2n) basis states with unit phases, and that permutation is then applied to the branches.
Any gate that would leave this form raises an error. The output image is compared with the
classical emulation (core.embed / core.extract) pixel by pixel."""
import math, json, time, sys, numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from core import *
from qcirc import registers, neqr_prep, scale_up, shift_by_4, write_D, draper_affine, key_oracle

class BranchSim:
    def __init__(self, regs):
        self.qidx = {q: i for i, q in enumerate(q for r in regs for q in r)}
        self.nq = len(self.qidx); self.B = np.zeros(1, np.int64); self.nH = 0
        self.cache = {}; self.block = None; self.counts = dict(h=0, mcx=0, swap=0, draper_blocks=0, block_gates=0)
        self.npos = None; self.max_phase_err = 0.0
    # ---- gates used by qcirc.py
    def barrier(self, *a): pass
    def h(self, qs):
        for q in (qs if isinstance(qs, (list, tuple)) else [qs]):
            b = self.qidx[q]; assert not ((self.B >> b) & 1).any(), "H on a non-|0> qubit leaves NEQR form"
            self.B = np.concatenate([self.B, self.B | (1 << b)]); self.nH += 1; self.counts["h"] += 1
        self.cache.clear()
    def swap(self, a, b):
        i, j = self.qidx[a], self.qidx[b]; bi, bj = (self.B >> i) & 1, (self.B >> j) & 1
        self.B ^= ((bi ^ bj) << i) | ((bi ^ bj) << j); self.counts["swap"] += 1
        self.cache = {k: v for k, v in self.cache.items() if i not in v[1] and j not in v[1]}
    def mcx(self, ctrl, tgt, ctrl_state):
        ent = self.cache.get(id(ctrl))
        if ent is None:
            idx = tuple(self.qidx[q] for q in ctrl); key = np.zeros(self.B.size, np.int64)
            for j, b in enumerate(idx): key |= ((self.B >> b) & 1) << j
            inv = -np.ones(1 << len(idx), np.int64); inv[key] = np.arange(self.B.size)
            assert np.unique(key).size == key.size, "control pattern shared by several branches"
            ent = self.cache[id(ctrl)] = (ctrl, idx, inv)
        i = ent[2][ctrl_state]
        if i >= 0: self.B[i] ^= 1 << self.qidx[tgt]
        self.counts["mcx"] += 1
    # ---- Draper adder blocks
    def append(self, gate, qubits):
        if gate.name == "qft":
            assert self.block is None; self.block = [("qft", gate, qubits)]
        elif gate.name == "qft_dg":
            self.block.append(("qft_dg", gate, qubits)); self._run_block(); self.block = None
        else: raise ValueError(gate.name)
    def p(self, ang, q): self.block.append(("p", ang, q))
    def cp(self, ang, a, b): self.block.append(("cp", ang, a, b))
    def _run_block(self):
        npos = self.npos; qc = QuantumCircuit(npos)
        for op in self.block:
            if op[0] in ("qft", "qft_dg"): qc.append(op[1], [self.qidx[q] for q in op[2]])
            elif op[0] == "p": qc.p(op[1], self.qidx[op[2]])
            else: qc.cp(op[1], self.qidx[op[2]], self.qidx[op[3]])
        Np = 1 << npos; ph = np.exp(2j * np.pi * np.arange(Np) / Np)      # distinct phases label the inputs
        out = Statevector(ph).evolve(qc).data
        assert np.allclose(np.abs(out), 1, atol=1e-9), "block is not a permutation"
        src = np.rint(np.angle(out) / (2 * np.pi) * Np).astype(np.int64) % Np
        err = np.abs(np.angle(out * np.conj(ph[src]))).max(); self.max_phase_err = max(self.max_phase_err, float(err))
        assert err < 1e-6 and np.unique(src).size == Np, "block is not a phase-free permutation"
        perm = np.empty(Np, np.int64); perm[src] = np.arange(Np)              # input p -> output perm[p]
        m = Np - 1; self.B = (self.B & ~m) | perm[self.B & m]
        self.counts["draper_blocks"] += 1; self.counts["block_gates"] += len(self.block); self.cache.clear()
    # ---- read-out
    def image(self, n):
        side = 1 << n; img = -np.ones((side, side), np.int64)
        img[self.B & (side - 1), (self.B >> n) & (side - 1)] = self.B >> (2 * n)
        return img, self.B.size

def run(cover, text, K):
    M = cover.shape[0]; side = 2 * M; n = int(math.log2(side)); N = 1 << n
    C_ref, nonce = embed(cover, text, K)                                     # classical emulation
    data = text.encode(); Km = master_key(K); S = session(Km, nonce)
    bits = encode_payload(data); cap = capacity(M, M)
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (cap - len(bits) + 7) // 8), np.uint8))[: cap - len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (2 * b[0::2] + b[1::2]).reshape(side, side); consts = round_constants(S, n); ks = keystream(S, side)
    y, x, c = registers(n); sim = BranchSim([y, x, c]); sim.npos = 2 * n
    t0 = time.time()
    neqr_prep(sim, y, x, c, cover); scale_up(sim, y, x); shift_by_4(sim, c); write_D(sim, y, x, c, D)
    V_q, nb = sim.image(n)
    for al, be, ga, de in consts:                                            # = qcirc.scramble_circuit
        draper_affine(sim, x, y, al, be); draper_affine(sim, y, x, ga, de)
    key_oracle(sim, y, x, c, ks)
    C_q, nb2 = sim.image(n); t_enc = time.time() - t0; enc_counts = dict(sim.counts)
    # receiver: U_pi^dagger U_key^dagger (key oracle is self-inverse; inverse shears in reverse order)
    key_oracle(sim, y, x, c, ks)
    for al, be, ga, de in reversed(consts):
        draper_affine(sim, y, x, (-ga) % N, (-de) % N); draper_affine(sim, x, y, (-al) % N, (-be) % N)
    V_back, _ = sim.image(n)
    rec_cover = (V_back >> 2)[0::2, 0::2]; Dq = V_back & 3
    rbits = np.empty(2 * side * side, np.uint8); rbits[0::2] = Dq.ravel() >> 1; rbits[1::2] = Dq.ravel() & 1
    rtext = decode_payload((rbits + 48).tobytes().decode()).decode()
    return dict(qubits=2 * n + NBITS_C, branches=int(nb2), amplitude_norm=float(nb2 * 2.0 ** (-sim.nH)),
                stego_equals_emulation=bool(np.array_equal(C_q, C_ref)),
                embedded_equals_emulation=bool(np.array_equal(V_q, 4 * np.repeat(np.repeat(cover, 2, 0), 2, 1) + D)),
                receiver_recovers_cover=bool(np.array_equal(rec_cover, cover)), receiver_recovers_text=(rtext == text),
                logical_ops_sender=enc_counts["mcx"] + enc_counts["h"] + enc_counts["swap"] + enc_counts["block_gates"],
                mcx_sender=enc_counts["mcx"], max_phase_err=sim.max_phase_err, seconds=time.time() - t0)

if __name__ == "__main__":
    R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
    A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
    imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
    if len(sys.argv) > 1: imgs = [i for i in imgs if i[0] in sys.argv[1:]]
    alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
    out = {}
    try: out = json.load(open("results/circsim.json"))
    except Exception: pass
    for name, I in imgs:
        cap = capacity(*I.shape); rng = np.random.default_rng(7); n_chars = 340000 if I.shape[0] == 512 else 85000
        while True:
            t = bytes(rng.choice(alph, n_chars)).decode()
            if len(encode_payload(t.encode())) <= cap: break
            n_chars -= 1000
        out[name] = run(I, t, K); print(name, out[name], flush=True)
        json.dump(out, open("results/circsim.json", "w"), indent=1)
