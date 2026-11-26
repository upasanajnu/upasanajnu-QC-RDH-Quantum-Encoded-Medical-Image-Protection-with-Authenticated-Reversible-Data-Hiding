"""Like-for-like benchmark on the 16 main images (identical covers and payloads, one CPU core):
standard authenticated encryption of cover and payload (AES-256-GCM, AES-256-SIV), compress-then-encrypt
(JPEG-LS + AES-SIV), a generic separable reserve-room RDHEI built from standard parts (content owner:
JPEG-LS + AES-CTR, data hider: payload + AES-CTR in the vacated room, no authentication), and the classical
twins of QC-RDH (2,2), (1,8) and RR. Payload: the text that fills RR for each image (Huffman-coded bits)."""
import json, os, time, numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESSIV
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from core import *
from core import _jls_encode, _jls_decode
from exp_rr_fill import fill
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
kg, ks = os.urandom(32), os.urandom(64); kc, kd = os.urandom(32), os.urandom(32)
def ctr(key, data, nonce): return Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor().update(data)
def timeit(f, rep=3):
    t = []
    for _ in range(rep): t0 = time.perf_counter(); out = f(); t.append(time.perf_counter() - t0)
    return out, 1000 * float(np.median(t))
rows = []
for name, I in imgs:
    N = I.size; cov = I.astype(np.uint8).tobytes(); txt = fill(rr_capacity(I)); pb = encode_payload(txt.encode())
    P = len(pb); pbytes = np.packbits(np.frombuffer(pb.encode(), np.uint8) - 48).tobytes()
    res = {}
    # AES-GCM over cover || payload (random 96-bit nonce, 128-bit tag)
    n96 = os.urandom(12); (c, te) = timeit(lambda: AESGCM(kg).encrypt(n96, cov + pbytes, None))
    (_, td) = timeit(lambda: AESGCM(kg).decrypt(n96, c, None)); res["AES-256-GCM"] = (8 * (len(c) + 12), te, td)
    (c, te) = timeit(lambda: AESSIV(ks).encrypt(cov + pbytes, None)); (_, td) = timeit(lambda: AESSIV(ks).decrypt(c, None))
    res["AES-256-SIV"] = (8 * len(c), te, td)
    def cte(): z = _jls_encode(I, 8); return AESSIV(ks).encrypt(len(z).to_bytes(4, "big") + z + pbytes, None)
    def ctd(c):
        m = AESSIV(ks).decrypt(c, None); lz = int.from_bytes(m[:4], "big"); return _jls_decode(m[4:4 + lz])
    (c, te) = timeit(cte); (rec, td) = timeit(lambda: ctd(c)); assert np.array_equal(rec, I)
    res["JPEG-LS + AES-256-SIV"] = (8 * len(c), te, td)
    def rrbe():                                   # content owner then data hider, same container size 8N
        z = _jls_encode(I, 8); n1, n2 = os.urandom(16), os.urandom(16)
        enc = ctr(kc, len(z).to_bytes(4, "big") + z, n1); room = N - len(enc)
        assert len(pbytes) <= room
        return enc + ctr(kd, pbytes, n2) + bytes(room - len(pbytes))
    (c, te) = timeit(rrbe); res["Generic RRBE RDHEI (JPEG-LS + AES-CTR)"] = (8 * N + 256, te, te)
    for lab, emb, ext, kw in [("QC-RDH (2,2), classical twin", embed, extract, {}), ("QC-RDH (1,8), classical twin", embed, extract, dict(kb=8, scale=1)),
                              ("QC-RDH RR, classical twin", embed_rr, extract_rr, {})]:
        (out, te) = timeit(lambda: emb(I, txt, K, **kw)); C, nu = out
        (r2, td) = timeit(lambda: ext(C, nu, K, **kw)); assert np.array_equal(r2[0], I) and r2[1] == txt
        w = int(C.max()).bit_length() if lab.startswith("QC-RDH (2") else (16 if "(1,8)" in lab else 8)
        bits = C.size * (10 if "(2,2)" in lab else 16 if "(1,8)" in lab else 8) + 128
        res[lab] = (bits, te, td)
    rows.append(dict(image=name, payload=P, cover_bits=8 * N, **{k: dict(bits=v[0], rate=P / v[0], ms_embed=v[1], ms_extract=v[2]) for k, v in res.items()}))
    print(name, P, {k: (round(P / v[0], 3), round(v[1], 1)) for k, v in res.items()}, flush=True)
json.dump(rows, open("results/benchmark.json", "w"), indent=1)
