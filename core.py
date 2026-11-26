# =====================================================================================
#  QC-RDH core — classical twin of the quantum circuits.
#  Every step below is the classical reversible function whose permutation-unitary is
#  implemented gate-by-gate in Section 6 of the notebook (and verified to agree exactly).
# =====================================================================================
import hashlib, hmac, struct, math, heapq
import numpy as np

NBITS_C = 10                                # intensity register: 8-bit cover + 2 embedded bits
DEFAULT_CHAOS = dict(a=-2.8, b=1.0, c=-0.1, d=-0.1, x0=7.0, y0=0.0, T=1000)
ROUNDS = 4                                  # cat-map shear rounds for position scrambling

# ---------------------------------------------------------------- 1. chaotic key schedule
def rational_map(x, y, a, b, c, d):
    return x + a * x / (1 + y * y) + b, y + c * x + d

def chaos_digest(chaos, n_states=256):
    """Iterate the 2-D rational map (paper Eq. 1), discard transient, hash the trajectory."""
    x, y = float(chaos["x0"]), float(chaos["y0"])
    a, b, c, d = (float(chaos[k]) for k in "abcd")
    for _ in range(int(chaos["T"])):
        x, y = rational_map(x, y, a, b, c, d)
    h = hashlib.sha256()
    for _ in range(n_states):
        x, y = rational_map(x, y, a, b, c, d)
        if not (math.isfinite(x) and math.isfinite(y) and abs(x) < 1e6 and abs(y) < 1e6):
            raise ValueError("invalid chaotic key: trajectory diverges for these parameters")
        h.update(struct.pack("<dd", x, y))
    return h.digest()

def master_key(qkd_key: bytes, chaos=DEFAULT_CHAOS):
    """K = HMAC(K_QKD, chaos digest): needs BOTH the 256-bit QKD key and the chaotic parameters."""
    return hmac.new(qkd_key, b"QC-RDH|" + chaos_digest(chaos), hashlib.sha256).digest()

def siv_nonce(K, cover, payload: bytes, ctr=None):
    """Synthetic IV: deterministic, depends on every cover pixel and payload byte (SIV construction).
    ctr (optional, Prop. counter-based freshness): 64-bit message counter inserted after the label."""
    m = hmac.new(K, b"nonce|" + (b"" if ctr is None else struct.pack("<Q", ctr)), hashlib.sha256)
    m.update(struct.pack("<II", *cover.shape)); m.update(cover.astype("<u2").tobytes()); m.update(payload)
    return m.digest()[:16]

def session(K, nonce):  return hmac.new(K, b"session|" + nonce, hashlib.sha256).digest()
def xof(seed, label, nbytes): return hashlib.shake_256(seed + label).digest(nbytes)

def round_constants(S, n):
    raw = np.frombuffer(xof(S, b"rounds", 8 * 4 * ROUNDS), dtype="<u8").reshape(ROUNDS, 4)
    return [tuple(int(v) & ((1 << n) - 1) for v in r) for r in raw]   # (alpha, beta, gamma, delta)

def keystream(S, side, nb=NBITS_C):         # nb-bit key value for every position (nb = 10 by default)
    if nb > 16:                             # wide registers (covers of more than 14 bits): 32-bit words
        raw = np.frombuffer(xof(S, b"ks", 4 * side * side), dtype="<u4").astype(np.int64)
        return (raw & ((1 << nb) - 1)).reshape(side, side)
    raw = np.frombuffer(xof(S, b"ks", 2 * side * side), dtype="<u2").astype(np.int64)
    return (raw & ((1 << nb) - 1)).reshape(side, side)

# ---------------------------------------------------------------- 2. position scrambling (quantum shears)
def scramble(Y, X, consts, n):
    m = (1 << n) - 1
    for al, be, ga, de in consts:
        X = (X + al * Y + be) & m            # Draper adder:  |x> -> |x + alpha*y + beta>
        Y = (Y + ga * X + de) & m            # Draper adder:  |y> -> |y + gamma*x + delta>
    return Y, X

def unscramble(Y, X, consts, n):
    m = (1 << n) - 1
    for al, be, ga, de in reversed(consts):
        Y = (Y - ga * X - de) & m
        X = (X - al * Y - be) & m
    return Y, X

# ---------------------------------------------------------------- 3. canonical Huffman over UTF-8 bytes + header
H_NSYM, H_SYM, H_LEN, H_PLEN = 9, 8, 5, 32

def _huff_lengths(data: bytes):
    freq = np.bincount(np.frombuffer(data, np.uint8), minlength=256)
    syms = [s for s in range(256) if freq[s]]
    if not syms: return {}
    if len(syms) == 1: return {syms[0]: 1}
    heap = [(int(freq[s]), s, [s]) for s in syms]; heapq.heapify(heap)
    depth = {s: 0 for s in syms}; tie = 256
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap); f2, _, b = heapq.heappop(heap)
        for s in a + b: depth[s] += 1
        heapq.heappush(heap, (f1 + f2, tie, a + b)); tie += 1
    assert max(depth.values()) < (1 << H_LEN), "code length exceeds header field"
    return depth

def _canonical(lengths):
    codes, code, prev = {}, 0, 0
    for s, l in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= (l - prev); prev = l; codes[s] = format(code, "0%db" % l); code += 1
    return codes

def encode_payload(data: bytes) -> str:
    ln = _huff_lengths(data); codes = _canonical(ln)
    body = "".join(codes[b] for b in data)
    hdr = format(len(ln), "0%db" % H_NSYM)
    for s, l in sorted(ln.items(), key=lambda kv: (kv[1], kv[0])):
        hdr += format(s, "08b") + format(l, "0%db" % H_LEN)
    return hdr + format(len(body), "0%db" % H_PLEN) + body

def decode_payload(bits: str) -> bytes:
    p = 0; n = int(bits[p:p + H_NSYM], 2); p += H_NSYM; ln = {}
    for _ in range(n):
        s = int(bits[p:p + 8], 2); p += 8; ln[s] = int(bits[p:p + H_LEN], 2); p += H_LEN
    plen = int(bits[p:p + H_PLEN], 2); p += H_PLEN
    if p + plen > len(bits) or 0 in ln.values(): raise ValueError("corrupt header")
    inv = {c: s for s, c in _canonical(ln).items()}; out = bytearray(); cur = ""
    for b in bits[p:p + plen]:
        cur += b
        if cur in inv: out.append(inv[cur]); cur = ""
        elif len(cur) > 31: raise ValueError("corrupt code stream")
    return bytes(out)

# ---------------------------------------------------------------- 4. sender / receiver
def capacity(M, N):  return 8 * M * N        # 2 bits in every pixel of the 2M x 2N enlarged image

def embed(cover, text, qkd_key, chaos=DEFAULT_CHAOS, ctr=None, qb=8, kb=2, scale=2):
    """Returns (stego ciphertext 2M x 2N with (qb+2)-bit values, public nonce).
    ctr: optional message counter (freshness extension); qb: cover bit depth (8 in all main experiments)."""
    M, N = cover.shape
    assert M == N and M & (M - 1) == 0, "NEQR needs a 2^n x 2^n cover (pad/resize first)"
    side = scale * M; n = int(math.log2(side)); assert qb + kb <= 32 and qb <= 16 and scale in (1, 2)
    data = text.encode("utf-8")
    K = master_key(qkd_key, chaos); nonce = siv_nonce(K, cover, data, ctr); S = session(K, nonce)
    bits = encode_payload(data); cap = kb * side * side          # = capacity(M, N) for kb = 2, scale = 2
    if len(bits) > cap: raise ValueError(f"payload {len(bits)} bits > capacity {cap}")
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (cap - len(bits) + 7) // 8), np.uint8))[: cap - len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (b.reshape(-1, kb) << np.arange(kb - 1, -1, -1)).sum(1).reshape(side, side)   # kb bits per pixel, MSB first
    E = np.repeat(np.repeat(cover.astype(np.int64), scale, 0), scale, 1)  # quantum NN scaling (H on new LSB qubits)
    V = (E << kb) + D                                                     # |c> -> |2^kb c> (qubit rotation) then |2^kb c + D>
    Y, X = np.indices((side, side)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
    C = np.empty_like(V); C[Y2, X2] = V[Y, X]                             # position scrambling unitary
    return C ^ keystream(S, side, qb + kb), nonce                          # key-oracle |p>|c> -> |p>|c xor k(p)>

def extract(C, nonce, qkd_key, chaos=DEFAULT_CHAOS, ctr=None, qb=8, kb=2, scale=2):
    """Returns (cover, text, report). Raises nothing: failures are reported, never silently accepted."""
    side = C.shape[0]; n = int(math.log2(side)); M = side // scale
    K = master_key(qkd_key, chaos); S = session(K, nonce)
    U = C ^ keystream(S, side, qb + kb)
    Y, X = np.indices((side, side)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
    V = U[Y2, X2]                                                         # inverse scrambling
    P, D = V >> kb, V & ((1 << kb) - 1)
    cover = P[0::scale, 0::scale]
    consistent = bool((P == np.repeat(np.repeat(cover, scale, 0), scale, 1)).all())   # copies must agree
    bits = ((D.ravel()[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    try:
        data = decode_payload((bits + 48).tobytes().decode()); text = data.decode("utf-8")
        used = len(encode_payload(data)); cap = bits.size
        pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (cap - used + 7) // 8), np.uint8))[: cap - used]
        pad_ok = bool(np.array_equal(bits[used:], pad))                     # padding is key-derived -> checkable
    except Exception:
        data, text, pad_ok = b"", None, False
    authentic = (consistent and pad_ok and cover.max() < (1 << qb)
                 and hmac.compare_digest(siv_nonce(K, cover, data, ctr), nonce))
    return cover, text, dict(consistent=consistent, authentic=bool(authentic))

# ---------------------------------------------------------------- 5. metrics
def entropy(a, levels):
    h = np.bincount(a.ravel(), minlength=levels) / a.size; h = h[h > 0]
    return float(-(h * np.log2(h)).sum())

def chi_square(a, levels):
    obs = np.bincount(a.ravel(), minlength=levels); exp = a.size / levels
    return float(((obs - exp) ** 2 / exp).sum())

def npcr_uaci(a, b, levels=1 << NBITS_C):
    return 100 * float((a != b).mean()), 100 * float((np.abs(a - b) / (levels - 1)).mean())

def corr(a, dx, dy):
    x = a[: a.shape[0] - dy, : a.shape[1] - dx].ravel().astype(float); y = a[dy:, dx:].ravel().astype(float)
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- 6. counter-based freshness (Prop. fresh)
class Sender:
    """Stateful sender: one 64-bit counter, incremented per message; (ctr, nonce) sent in the clear."""
    def __init__(self, qkd_key, chaos=DEFAULT_CHAOS): self.k, self.ch, self.ctr = qkd_key, chaos, 0
    def send(self, cover, text):
        self.ctr += 1; C, nu = embed(cover, text, self.k, self.ch, ctr=self.ctr); return C, self.ctr, nu

class Receiver:
    """Stateful receiver: keeps the largest accepted counter and rejects any counter not larger."""
    def __init__(self, qkd_key, chaos=DEFAULT_CHAOS): self.k, self.ch, self.last = qkd_key, chaos, 0
    def receive(self, C, ctr, nu):
        if ctr <= self.last: return None, None, dict(consistent=False, authentic=False, fresh=False)
        cov, txt, rep = extract(C, nu, self.k, self.ch, ctr=ctr); rep["fresh"] = True
        if rep["authentic"]: self.last = ctr
        return cov, txt, rep

# ---------------------------------------------------------------- 7. integer-only chaos digest (interoperable profile)
FX = 32                                                    # Q32.32 fixed point, int64 states, int128 products

def _fx(v): return int(round(v * (1 << FX)))
def _fdiv(a, b): return a // b                            # floor division (defined for negative operands)

def chaos_digest_fixed(chaos, n_states=256):
    """Same rational map in Q32.32 integer arithmetic with floor rounding: bit-identical on every platform."""
    x, y = _fx(chaos["x0"]), _fx(chaos["y0"]); a, b, c, d = (_fx(chaos[k]) for k in "abcd"); one = 1 << FX
    def step(x, y):
        den = one + _fdiv(y * y, one)
        return x + _fdiv(_fdiv(a * x, one) * one, den) + b, y + _fdiv(c * x, one) + d
    for _ in range(int(chaos["T"])): x, y = step(x, y)
    h = hashlib.sha256()
    for _ in range(n_states):
        x, y = step(x, y)
        if abs(x) >= (10 ** 6) << FX or abs(y) >= (10 ** 6) << FX: raise ValueError("trajectory diverges")
        h.update(struct.pack("<qq", x, y))
    return h.digest()

# ---------------------------------------------------------------- 8. room-reserving configuration (RR)
# The sender compresses the cover losslessly (JPEG-LS, ISO/IEC 14495-1) before quantum encoding and fills the
# room this frees with payload. The container has the size of the cover and w-bit pixels (w = qb + extra);
# it is prepared as an NEQR state and encrypted with the same scrambling and key oracle as QC-RDH.
RR_HDR = 32                                   # bits of the header that stores the length of the compressed cover

def _jls_encode(cover, qb):
    import imagecodecs
    return bytes(imagecodecs.jpegls_encode(cover.astype(np.uint8 if qb <= 8 else np.uint16)))

def _jls_decode(z):
    import imagecodecs
    return imagecodecs.jpegls_decode(z).astype(np.int64)

def rr_capacity(cover, qb=8, extra=0):
    """Payload bits available: container bits minus header minus compressed cover."""
    return cover.size * (qb + extra) - RR_HDR - 8 * len(_jls_encode(cover, qb))

def _bits_to_words(b, w): return (b.reshape(-1, w) << np.arange(w - 1, -1, -1)).sum(1)
def _words_to_bits(v, w): return ((v.ravel()[:, None] >> np.arange(w - 1, -1, -1)) & 1).astype(np.uint8).ravel()

def embed_rr(cover, text, qkd_key, chaos=DEFAULT_CHAOS, ctr=None, qb=8, extra=0):
    """Returns (stego M x N with (qb+extra)-bit values, public nonce)."""
    M, N = cover.shape
    assert M == N and M & (M - 1) == 0 and cover.max() < (1 << qb)
    n = int(math.log2(M)); w = qb + extra; data = text.encode("utf-8")
    K = master_key(qkd_key, chaos); nonce = siv_nonce(K, cover, data, ctr); S = session(K, nonce)
    z = _jls_encode(cover, qb); zb = np.unpackbits(np.frombuffer(z, np.uint8)).astype(np.int64)
    hdr = np.array([int(c) for c in format(len(z), "032b")], np.int64)
    pb = np.frombuffer(encode_payload(data).encode(), np.uint8).astype(np.int64) - 48
    tot = M * N * w; used = RR_HDR + zb.size + pb.size
    if used > tot: raise ValueError(f"payload {pb.size} bits > capacity {tot - RR_HDR - zb.size}")
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (tot - used + 7) // 8), np.uint8))[: tot - used].astype(np.int64)
    V = _bits_to_words(np.concatenate([hdr, zb, pb, pad]), w).reshape(M, N)      # NEQR intensities of the container
    Y, X = np.indices((M, M)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
    C = np.empty_like(V); C[Y2, X2] = V[Y, X]
    return C ^ keystream(S, M, w), nonce

def extract_rr(C, nonce, qkd_key, chaos=DEFAULT_CHAOS, ctr=None, qb=8, extra=0):
    M = C.shape[0]; n = int(math.log2(M)); w = qb + extra
    K = master_key(qkd_key, chaos); S = session(K, nonce)
    U = C ^ keystream(S, M, w)
    Y, X = np.indices((M, M)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
    bits = _words_to_bits(U[Y2, X2], w)
    try:
        lz = int("".join(map(str, bits[:RR_HDR])), 2); end = RR_HDR + 8 * lz
        if end > bits.size: raise ValueError("bad length")
        cover = _jls_decode(np.packbits(bits[RR_HDR:end]).tobytes())
        rest = bits[end:]; data = decode_payload((rest + 48).tobytes().decode()); text = data.decode("utf-8")
        used = len(encode_payload(data)); pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (rest.size - used + 7) // 8), np.uint8))[: rest.size - used]
        ok = (bool(np.array_equal(rest[used:], pad)) and cover.shape == (M, M) and cover.max() < (1 << qb)
              and hmac.compare_digest(siv_nonce(K, cover, data, ctr), nonce))
    except Exception:
        cover, text, ok = np.zeros((M, M), np.int64), None, False
    return cover, text, dict(consistent=ok, authentic=bool(ok))
