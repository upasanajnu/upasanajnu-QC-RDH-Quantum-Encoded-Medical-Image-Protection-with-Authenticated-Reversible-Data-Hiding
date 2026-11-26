"""BB84 with finite-key post-processing over an authenticated classical channel (two parties, TCP loopback).

* Quantum channel: SIMULATED. For each signal Alice picks a bit and a basis; Bob measures in a random basis.
  Matching bases give Alice's bit flipped with probability p (depolarising noise); an intercept-resend
  eavesdropper can be switched on. Measurement statistics only, as in any classical BB84 simulation.
* Classical channel: REAL TCP connection between two threads (Alice server, Bob client), every message
  authenticated with a Wegman-Carter MAC: polynomial hash modulo the prime 2^127-1 keyed by a one-time pair
  (r, s) taken from a pre-shared key pool (forgery probability <= #blocks / (2^127-1) per message).
  An optional man-in-the-middle relay modifies one message to test that tampering is rejected.
* Post-processing: sifting, parameter estimation on the X-basis bits, error correction with a Cascade
  implementation (leakage counted exactly), error verification by a 2^61-1 polynomial hash, privacy
  amplification by Toeplitz hashing, key length from the finite-key bound of Tomamichel et al. (2012):
  l <= n(1 - h(Q_tol + mu)) - leak_EC - log2(2/(eps_sec^2 eps_cor)),  mu = sqrt((n+k)/(nk) (k+1)/k ln(4/eps_sec))."""
import socket, threading, json, os, math, hashlib, struct, sys, numpy as np

P127 = (1 << 127) - 1; P61 = (1 << 61) - 1
def h2(p): return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)

class AuthChannel:
    """Length-prefixed JSON messages. The whole transcript (every byte sent and received, in order) is
    authenticated once at the end with a Wegman-Carter tag in each direction, before privacy amplification;
    each tag consumes 256 bits of the pre-shared pool (127-bit hash key r and 128-bit one-time pad s)."""
    def __init__(self, sock, pool, role): self.s, self.pool, self.used, self.role, self.tr = sock, pool, 0, role, bytearray()
    def _key(self):
        k = self.pool[self.used:self.used + 32]; self.used += 32
        assert len(k) == 32, "authentication key pool exhausted"
        return int.from_bytes(k[:16], "little") % P127, int.from_bytes(k[16:], "little")
    @staticmethod
    def _poly(r, data):
        acc = 0
        for i in range(0, len(data), 15):
            acc = (acc + int.from_bytes(data[i:i + 15] + b"\x01", "little")) * r % P127
        return acc
    def send(self, obj):
        data = json.dumps(obj).encode(); self.tr += b"A>" if self.role == "A" else b"B>"; self.tr += struct.pack(">I", len(data)) + data
        self.s.sendall(struct.pack(">I", len(data)) + data)
    def recv(self):
        n = struct.unpack(">I", self._read(4))[0]; data = self._read(n)
        self.tr += b"B>" if self.role == "A" else b"A>"; self.tr += struct.pack(">I", n) + data
        return json.loads(data)
    def tag(self):                                    # tag of the transcript, keys: first slot = Alice, second = Bob
        return [(self._poly(r, bytes(self.tr)) + s) % (1 << 128) for r, s in (self._key(), self._key())]
    def _read(self, n):
        b = b""
        while len(b) < n:
            c = self.s.recv(n - len(b))
            if not c: raise ConnectionError("closed")
            b += c
        return b

def parity(bits, idx): return int(bits[idx].sum() & 1)

def cascade_bob(ch, bob, seeds, Qest):
    """Bob's side of Cascade with backtracking; Alice answers parity requests (all parities counted as leakage).
    Blocks of one pass are disjoint, so the odd blocks of one pass are searched in parallel."""
    n = bob.size; leak = 0; k1 = max(4, int(0.73 / max(Qest, 1e-3)))
    perms = [np.arange(n)] + [np.random.default_rng(sd).permutation(n) for sd in seeds[1:]]
    def ask(blocks):
        nonlocal leak
        ch.send({"parities": [b.tolist() for b in blocks]}); leak += len(blocks); return ch.recv()["p"]
    passes = []                                            # (blocks, alice parities)
    for pas, perm in enumerate(perms):
        ksz = k1 << pas; blocks = [perm[i:i + ksz] for i in range(0, n, ksz)]
        passes.append((blocks, ask(blocks)))
        while True:
            odd = None
            for bl, ap in passes:
                o = [b for b, a in zip(bl, ap) if a != parity(bob, b)]
                if o: odd = o; break
            if odd is None: break
            todo = odd
            while todo:
                halves = [b[: len(b) // 2] for b in todo]; ap2 = ask(halves); nxt = []
                for b, hb, a in zip(todo, halves, ap2):
                    sub = hb if a != parity(bob, hb) else b[len(b) // 2:]
                    if len(sub) == 1: bob[sub[0]] ^= 1
                    else: nxt.append(sub)
                todo = nxt
    ch.send({"parities": None}); return bob, leak

def alice(srv, pool, cfg, out):
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["N"]; bits = rng.integers(0, 2, N); bases = (rng.random(N) < cfg["pz"]).astype(int)   # 1 = Z
    # ---- simulated quantum channel (sent to Bob out of band) ----
    bb = (rng.random(N) < cfg["pz"]).astype(int)
    if cfg["eve"]:
        eb = rng.integers(0, 2, N); eres = np.where(eb == bases, bits, rng.integers(0, 2, N))
        res = np.where(bb == eb, eres, rng.integers(0, 2, N))
    else: res = np.where(bb == bases, bits, rng.integers(0, 2, N))
    res = res ^ ((rng.random(N) < cfg["p"]) & (bb == bases))
    out["quantum"] = (bb, res)
    # ---- authenticated classical post-processing ----
    conn = srv.accept()[0]; conn.settimeout(30); ch = AuthChannel(conn, pool, "A")
    try: msg = ch.recv()
    except (ValueError, ConnectionError) as e: out["result"] = dict(abort=True, reason=str(e)); conn_close(ch); return
    bob_bases = np.array(msg["bases"])
    keep = bob_bases == bases; ch.send({"bases": bases.tolist()})
    zi = np.flatnonzero(keep & (bases == 1)); xi = np.flatnonzero(keep & (bases == 0))
    ch.send({"test": bits[xi].tolist()})                      # all X-basis bits are used for estimation
    v = ch.recv()
    if v.get("abort"): out["result"] = dict(abort=True, lam=v["lam"]); return
    akey = bits[zi].copy(); seeds = [int(x) for x in rng.integers(1, 2**31, 4)]; ch.send({"seeds": seeds})
    while True:
        m = ch.recv()
        if m["parities"] is None: break
        ch.send({"p": [parity(akey, np.array(b, dtype=np.int64)) for b in m["parities"]]})
    leak = ch.recv()["leak"]
    # error verification (eps_cor) with a polynomial hash mod 2^61-1
    r = int(rng.integers(1, P61)); hv = 0
    for b in np.packbits(akey).tolist(): hv = (hv * r + b + 1) % P61
    ch.send({"r": r, "hash": hv}); ok = ch.recv()["ok"]
    if not ok: out["result"] = dict(abort=True, reason="verification"); return
    n, k, Qt = zi.size, xi.size, cfg["Qtol"]; es, ec = cfg["eps_sec"], cfg["eps_cor"]
    mu = math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(4 / es))
    ell = int(math.floor(n * (1 - h2(Qt + mu)) - leak - math.log2(2 / (es ** 2 * ec))))
    if ell < 256: ch.send({"ell": ell}); out["result"] = dict(abort=True, reason="too short", ell=ell); return
    seed = rng.integers(0, 2, n + ell - 1).tolist(); ch.send({"ell": ell, "toeplitz": seed})
    ta, tb = ch.tag(); ch.s.sendall(ta.to_bytes(16, "little")); got = int.from_bytes(ch._read(16), "little")
    if got != tb: out["result"] = dict(abort=True, reason="transcript authentication failed"); return
    key = toeplitz(akey, np.array(seed), ell)
    out["result"] = dict(abort=False, n=int(n), k=int(k), leak=int(leak), mu=mu, ell=ell, key=np.packbits(key).tobytes().hex(),
                         auth_bits_used=8 * ch.used, net_bits=ell - 8 * ch.used)

def conn_close(ch):
    try: ch.s.close()
    except Exception: pass

def toeplitz(x, seed, ell):
    n = x.size; xi = np.flatnonzero(x); out = np.zeros(ell, np.int64)
    for i in range(ell): out[i] = seed[i + (n - 1) - xi].sum() & 1          # T[i, j] = seed[i - j + n - 1]
    return out

def bob(sock, pool, cfg, quantum, out):
    ch = AuthChannel(sock, pool, "B"); bb, res = quantum
    try:
        ch.send({"bases": bb.tolist()}); ab = np.array(ch.recv()["bases"]); keep = ab == bb
        zi = np.flatnonzero(keep & (bb == 1)); xi = np.flatnonzero(keep & (bb == 0))
        test = np.array(ch.recv()["test"]); lam = float((test != res[xi]).mean())
        if lam > cfg["Qtol"]: ch.send({"abort": True, "lam": lam}); out["result"] = dict(abort=True, lam=lam); return
        ch.send({"abort": False}); seeds = ch.recv()["seeds"]
        bkey, leak = cascade_bob(ch, res[zi].copy(), seeds, max(lam, 0.005)); ch.send({"leak": leak})
        m = ch.recv(); hv = 0
        for b in np.packbits(bkey).tolist(): hv = (hv * m["r"] + b + 1) % P61
        ch.send({"ok": hv == m["hash"]})
        m = ch.recv()
        if "toeplitz" not in m: out["result"] = dict(abort=True); return
        ta, tb = ch.tag(); got = int.from_bytes(ch._read(16), "little"); ch.s.sendall(tb.to_bytes(16, "little"))
        if got != ta: out["result"] = dict(abort=True, reason="transcript authentication failed"); return
        key = toeplitz(bkey, np.array(m["toeplitz"]), m["ell"])
        out["result"] = dict(abort=False, lam=lam, key=np.packbits(key).tobytes().hex())
    except (ValueError, ConnectionError) as e: out["result"] = dict(abort=True, reason=str(e))

def mitm_relay(lsock, target_port, flip_at):
    """Active attacker on the classical channel: relays all traffic and changes one value 0 -> 1 in Alice's
    announcement of her test bits, keeping the message well-formed."""
    c, _ = lsock.accept(); a = socket.create_connection(("127.0.0.1", target_port)); done = [False]
    def pump(src, dst, tamper):
        while True:
            try: d = src.recv(1 << 20)
            except OSError: break
            if not d: break
            if tamper and not done[0]:
                j = d.find(b'"test": [')
                i = d.find(b"0,", j) if j >= 0 else -1
                if i > 0: d = d[:i] + b"1" + d[i + 1:]; done[0] = True
            try: dst.sendall(d)
            except OSError: break
    threading.Thread(target=pump, args=(c, a, False), daemon=True).start(); pump(a, c, True)

def run(cfg, mitm=False):
    pool = os.urandom(32 * 20000)                                   # pre-shared authentication key pool
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); port = srv.getsockname()[1]
    A, B = {}, {}
    ta = threading.Thread(target=alice, args=(srv, pool, cfg, A), daemon=True); ta.start()
    if mitm:
        rl = socket.socket(); rl.bind(("127.0.0.1", 0)); rl.listen(1)
        threading.Thread(target=mitm_relay, args=(rl, port, 3), daemon=True).start(); port = rl.getsockname()[1]
    import time
    while "quantum" not in A: time.sleep(0.01)
    s = socket.create_connection(("127.0.0.1", port)); s.settimeout(30)
    tb = threading.Thread(target=bob, args=(s, pool, cfg, A["quantum"], B)); tb.start(); tb.join(60)
    s.close(); ta.join(5)
    return A.get("result"), B.get("result")

if __name__ == "__main__":
    base = dict(N=60000, pz=0.5, p=0.01, Qtol=0.03, eps_sec=1e-10, eps_cor=1e-15, eve=False, seed=1)
    res = {}
    for name, cfg, mitm in [("honest p=1%", base, False), ("honest p=2%", dict(base, p=0.02, seed=2), False),
                            ("intercept-resend", dict(base, eve=True, seed=3), False), ("man-in-the-middle", dict(base, seed=4), True)]:
        a, b = run(cfg, mitm)
        ok = (a and b and not a.get("abort") and not b.get("abort") and a["key"] == b["key"])
        res[name] = dict(alice={k: v for k, v in (a or {}).items() if k != "key"}, bob={k: v for k, v in (b or {}).items() if k != "key"},
                         keys_equal=bool(ok), key_bits=(a or {}).get("ell"))
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "results", "bb84_finite_key.json"), "w"), indent=1)
