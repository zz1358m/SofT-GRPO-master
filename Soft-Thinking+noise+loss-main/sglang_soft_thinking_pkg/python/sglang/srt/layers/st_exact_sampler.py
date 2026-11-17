"""[ST-OPT] Bit-identical fast path for the soft-thinking top-k / top-p step.

Current path (per row of a batch x vocab probability matrix):
    r = top_p_renorm_prob(top_k_renorm_prob(probs, top_k), top_p)       # flashinfer kernels, full vocabulary
    v, i = torch.topk(r, K); v = v / v.sum(-1)
Both renorm kernels scan the row in tiles of BLOCK_THREADS x VEC_SIZE elements (1024 x 4 or 512 x 4 floats),
and every sum is a per-tile block reduction accumulated tile by tile in tile order. Elements that do not pass the
pivot contribute exact zeros. Hence the kernels return the same bits on a compact row that keeps the candidate
values at the same position inside their 4096-element block, with the occupied blocks in the same order and zeros
elsewhere, provided the compact row still contains every value >= the k-th value (all ties) and at least one
smaller value (so that the top-k pivot search ends between them, as on the full row).
torch.topk(r, K) is reproduced exactly: it gathers the elements > the K-th value in index order, then the elements
equal to it in index order, and sorts these K values with the same small-sort routine as torch.sort.
Rows that need more candidates (ties beyond the margin, top_k > K) are reported in `bad`; the caller then uses the
current path for the whole batch.
"""
import torch
from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

BLOCK = 4096  # multiple of the renorm tile (1024 x 4 or 512 x 4 float32)


class Cand:
    __slots__ = ("cv", "ci", "bad", "bad_rows")


CHUNKMAX = __import__("os").environ.get("SOFT_EXACT_CHUNKMAX", "1") == "1"


def topk_rows(probs, C, G=64):
    """Largest C values per row (sorted) and their indices, via chunk maxima: the C largest elements lie in the C
    chunks (of G elements) with the largest maxima, otherwise C chunk maxima would exceed one of them. With ties at
    the C-th place an equal element of another chunk may be kept instead; this does not matter here: if an element
    >= the k-th value is missed, the pool already holds C elements >= the k-th value and the row is `bad`."""
    B, V = probs.shape
    if not CHUNKMAX or V % G or not probs.is_contiguous():
        return torch.topk(probs, k=C, dim=-1)
    cm = probs.view(B, V // G, G).amax(dim=2)  # chunk maxima
    ch = torch.topk(cm, k=C, dim=1).indices
    pos = (ch[:, :, None] * G + torch.arange(G, device=probs.device)).view(B, C * G)
    cv, j = torch.topk(probs.gather(1, pos), k=C, dim=1)
    return cv, pos.gather(1, j)


def candidates(probs, top_ks, K, extra=3):
    C = K + extra
    cv, ci = topk_rows(probs, C)  # sorted, descending
    k = top_ks.long()
    vk = cv.gather(1, (k.clamp(min=1, max=C - 1) - 1)[:, None])  # k-th largest value
    bad = ((k > K) | (k < 1))[:, None] | (cv[:, C - 1:C] >= vk)  # need a candidate strictly below the k-th
    c = Cand()
    c.cv, c.ci, c.bad_rows = cv, ci, bad.squeeze(1)
    c.bad = c.bad_rows.any()
    return c


def finish(c, top_ks, top_ps, K):
    cv, ci = c.cv, c.ci
    B, C = ci.shape
    # compact rows: same offset inside each 4096 block, occupied blocks kept in order
    blk = torch.div(ci, BLOCK, rounding_mode="floor")
    off = ci - blk * BLOCK
    sb, order = torch.sort(blk, dim=1)
    new = torch.ones_like(sb, dtype=torch.bool)
    new[:, 1:] = sb[:, 1:] != sb[:, :-1]
    rank = torch.empty_like(sb).scatter_(1, order, new.long().cumsum(1) - 1)
    pos = rank * BLOCK + off
    comp = torch.zeros(B, C * BLOCK, dtype=cv.dtype, device=cv.device)
    comp.scatter_(1, pos, cv)
    r = top_p_renorm_prob(top_k_renorm_prob(comp, top_ks), top_ps)
    rv = r.gather(1, pos)  # renormalized candidate values (0 if removed)
    # torch.topk(full row, K) on the full row = rv at ci, zeros elsewhere
    nz = rv > 0
    fill = torch.arange(K + C, device=ci.device).expand(B, K + C)  # smallest zero positions
    in_nz = ((fill[:, :, None] == ci[:, None, :]) & nz[:, None, :]).any(-1)
    neg = torch.full_like(rv, -1.0)
    ev = torch.cat([torch.where(nz, rv, neg), torch.where(in_nz, -1.0, 0.0).to(rv.dtype)], dim=1)
    ei = torch.cat([ci, fill], dim=1)
    kth = torch.topk(ev, K, dim=1).values[:, K - 1:K]
    grp = torch.where(ev > kth, 0, torch.where(ev == kth, 1, 2))
    o1 = torch.sort(ei, dim=1, stable=True).indices
    o2 = torch.sort(grp.gather(1, o1), dim=1, stable=True).indices
    sel = o1.gather(1, o2)[:, :K]  # gather order of torch.topk
    sv, si = ev.gather(1, sel), ei.gather(1, sel)
    vals, perm = torch.sort(sv, dim=1, descending=True)  # same small sort as inside torch.topk
    return vals, si.gather(1, perm)


def reference(probs, top_ks, top_ps, K):
    r = top_p_renorm_prob(top_k_renorm_prob(probs, top_ks), top_ps)
    return torch.topk(r, k=K, dim=-1)
