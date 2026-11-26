import numpy as np
from core import encode_payload
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
def fill(cap, seed=7):
    """Pseudo-random text whose Huffman code fills a capacity of cap bits as far as possible (seed 7)."""
    rng = np.random.default_rng(seed); n = max(10, int(cap / 6.25))
    while True:
        t = bytes(rng.choice(alph, n)).decode()
        if len(encode_payload(t.encode())) <= cap: return t
        n -= max(20, n // 400)
