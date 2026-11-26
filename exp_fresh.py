"""Counter-based freshness extension (Prop. fresh): implemented in core.Sender/Receiver and tested on
the 16 test images. Checks: round trip, equal inputs -> unrelated stego images, replay and reordering
rejected, counter tampering rejected, and the stateless scheme's replay/equality behaviour for contrast."""
import json, numpy as np
from core import *
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
txt = "Patient: R. Sharma, 61F | Dx: stage II breast carcinoma | Rx: letrozole 2.5 mg od"
rows = []
for name, I in imgs:
    snd, rcv = Sender(K), Receiver(K)
    C1, c1, n1 = snd.send(I, txt); C2, c2, n2 = snd.send(I, txt); C3, c3, n3 = snd.send(I, txt)   # same input 3 times
    ok1 = rcv.receive(C1, c1, n1); ok3 = rcv.receive(C3, c3, n3)                 # 1 then 3 (2 is skipped)
    rt = bool(np.array_equal(ok1[0], I) and ok1[1] == txt and ok1[2]["authentic"] and ok3[2]["authentic"])
    replay = not rcv.receive(C1, c1, n1)[2]["authentic"]                        # old pair again
    reorder = not rcv.receive(C2, c2, n2)[2]["authentic"]                       # older counter after newer
    forged_ctr = not Receiver(K).receive(C3, c3 + 1, n3)[2]["authentic"]       # counter changed in transit
    npcr, uaci = npcr_uaci(C1, C2)
    # stateless scheme, for contrast
    S1, m1 = embed(I, txt, K); S2, m2 = embed(I, txt, K)
    rows.append(dict(image=name, roundtrip=rt, replay_rejected=replay, reorder_rejected=reorder,
                     ctr_tamper_rejected=forged_ctr, equal_inputs_npcr=npcr, equal_inputs_uaci=uaci,
                     nonces_distinct=len({n1, n2, n3}) == 3,
                     stateless_equal_outputs=bool(np.array_equal(S1, S2) and m1 == m2),
                     stateless_replay_accepted=bool(extract(S1, m1, K)[2]["authentic"])))
    print(rows[-1])
summ = {k: (all(r[k] for r in rows) if isinstance(rows[0][k], bool) else
            [min(r[k] for r in rows), float(np.mean([r[k] for r in rows])), max(r[k] for r in rows)])
        for k in rows[0] if k != "image"}
json.dump(dict(rows=rows, summary=summ), open("results/fresh.json", "w"), indent=1); print(summ)
