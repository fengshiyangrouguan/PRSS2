"""Frozen utterance embedding for the chi features of future observations.

chi(u) = CountSketch( mean over tokens of embed_tokens[t] (+ optional tag) )

Everything is frozen: the input embedding is used in no_grad (the P side
of the Ky Fan measurement is a constant by contract) and the sketch
tables come from a fixed seed.  The optional ``tag`` appends a fixed
signature to the mean BEFORE the sketch, which is how the second
horizon's "one more memory update happened" marker enters chi_2 (plan
v2 L4: chi_2 carries the one-update identifier).
"""

import torch
from torch import nn


class UtteranceEmbed(nn.Module):
    """chi(u) = CountSketch(mean embed + tag) -> [d_chi]; fully frozen."""

    def __init__(self, hidden_dim: int, d_chi: int = 64, seed: int = 0,
                 tag_dim: int = 8, n_tags: int = 2, repeats: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_chi = int(d_chi)
        self.tag_dim = int(tag_dim)
        self.n_tags = int(n_tags)
        full_dim = hidden_dim + tag_dim
        rows = torch.arange(full_dim)
        cols = rows % d_chi
        signs = torch.ones(full_dim)
        g = torch.Generator()
        g.manual_seed(int(seed))
        extra_n = int(repeats) * full_dim
        extra_rows = torch.randint(0, full_dim, (extra_n,), generator=g)
        extra_cols = torch.randint(0, d_chi, (extra_n,), generator=g)
        extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
        self.register_buffer("sketch_rows", torch.cat([rows, extra_rows]),
                             persistent=True)
        self.register_buffer("sketch_cols", torch.cat([cols, extra_cols]),
                             persistent=True)
        self.register_buffer("sketch_signs", torch.cat([signs, extra_signs]),
                             persistent=True)
        self.register_buffer(
            "scale", torch.tensor((self.d_chi / full_dim) ** 0.5),
            persistent=True)
        # Fixed per-tag signatures (tag 0 = plain, tag 1 = one-update).
        tg = torch.Generator()
        tg.manual_seed(int(seed) + 17)
        self.register_buffer("tag_table",
                             torch.randint(0, 2, (n_tags, tag_dim),
                                           generator=tg) * 2 - 1,
                             persistent=True)

    def forward(self, embed_tokens: nn.Embedding, token_ids: torch.Tensor,
                tag: int = 0) -> torch.Tensor:
        """Frozen chi for one utterance.

        Args:
            embed_tokens: the (frozen) input embedding module.
            token_ids: [B, L_u] utterance token ids (padded spans get
                masked by the caller via ``valid``).
            tag: 0 = plain (horizon 1), 1 = one-update marker (horizon 2).

        Returns:
            chi [B, d_chi], detached (the P side is a constant).
        """
        if tag < 0 or tag >= self.n_tags:
            raise ValueError("tag must be in [0, {}), got {}"
                             .format(self.n_tags, tag))
        with torch.no_grad():
            emb = embed_tokens(token_ids.to(embed_tokens.weight.device))
            mean = emb.float().mean(dim=1)  # [B, hidden_dim]
            if tag != 0:
                tag_vec = self.tag_table[tag].to(dtype=mean.dtype,
                                                 device=mean.device)
                mean = torch.cat([mean, tag_vec.expand(mean.shape[0], -1)],
                                 dim=-1)
            else:
                zero_tag = torch.zeros(mean.shape[0], self.tag_dim,
                                       dtype=mean.dtype, device=mean.device)
                mean = torch.cat([mean, zero_tag], dim=-1)
            cols = self.sketch_cols.to(mean.device)
            rows = self.sketch_rows.to(mean.device)
            signs = self.sketch_signs.to(dtype=mean.dtype, device=mean.device)
            out = torch.zeros(mean.shape[0], self.d_chi, dtype=mean.dtype,
                              device=mean.device)
            out.index_add_(1, cols, mean[:, rows] * signs)
            return (out * self.scale).detach()
