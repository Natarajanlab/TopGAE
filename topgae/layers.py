"""The dual-score message-passing layer of TopGAE (Methods, "Dual-score message passing")."""
import torch
from torch import nn
from torch.nn import functional as F

from .structure import structural_features
from .utils import scatter_softmax, scatter_sum


class DualScoreLayer(nn.Module):
    """Role-specific gated embeddings + S1/S2-weighted message passing.

    For every training edge (i -> j) the layer builds a pair representation
    ``[u_i || u_j || u_i * u_j]`` from the gated source and target embeddings
    and scores it twice:

    * **S1** - softmax over the incoming sources (TFs) of target ``j``
      (relative regulatory contribution);
    * **S2** - ``2 * sigmoid(.)`` in (0, 2), an independent interaction
      modulation factor.

    Messages ``S1 * S2 * W_msg(u_i)`` are summed at the target and added to a
    residual + LayerNorm. In the optional reverse pass the same S1 logits are
    normalised over each source's outgoing targets so that source embeddings
    also receive information from their targets (no reciprocal edges added).

    Parameters
    ----------
    in_features : int
        Number of cells (expression features per gene).
    out_features : int
        Hidden width.
    use_importance : bool
        Enable the role-specific structural gates (degree + HITS).
    reverse_messages, use_s1, use_s2 : bool
        Switches for the reverse pass and the two edge scores.
    """

    def __init__(self, in_features, out_features, dropout=0.2,
                 use_importance=True, reverse_messages=True, use_s1=True, use_s2=True):
        super().__init__()
        self.out_features = out_features
        self.use_importance, self.reverse_messages = use_importance, reverse_messages
        self.use_s1, self.use_s2 = use_s1, use_s2
        self.W_core_tf = nn.Linear(in_features, out_features)
        self.W_core_gene = nn.Linear(in_features, out_features)
        self.W_msg = nn.Linear(out_features, out_features, bias=False)
        self.W_reverse = nn.Linear(out_features, out_features, bias=False)
        self.W_row_res = nn.Linear(out_features, out_features)
        self.W_col_res = nn.Linear(out_features, out_features)
        self.norm_row, self.norm_col = nn.LayerNorm(out_features), nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        def score_head(width):
            return nn.Sequential(nn.Linear(width, out_features), nn.SiLU(), nn.Linear(out_features, 1))

        self.W_attn_s1, self.W_attn_s2 = score_head(3 * out_features), score_head(3 * out_features)
        self.reg_importance, self.tgt_importance = score_head(out_features + 4), score_head(out_features + 4)
        # Small modulation initially, with nonzero gradients for gate heads.
        self.importance_strength = nn.Parameter(torch.tensor(-2.0))
        self.last_importance = None

    def forward(self, x_row, x_col, train_edge_index, row_genes, col_genes,
                skip_core=False, structure=None):
        if skip_core:
            if x_row.shape[1] != self.out_features or x_col.shape[1] != self.out_features:
                raise ValueError("skip_core requires hidden-width inputs.")
            hr, hc = x_row, x_col
        else:
            hr = F.leaky_relu(self.W_core_tf(self.dropout(x_row)), 0.1)
            hc = F.leaky_relu(self.W_core_gene(self.dropout(x_col)), 0.1)
        if self.use_importance:
            if structure is None:
                structure = structural_features(train_edge_index, row_genes, col_genes, x_row.device)
            sr, sc = structure
            gr = torch.sigmoid(self.reg_importance(torch.cat([hr, sr], -1)))
            gc = torch.sigmoid(self.tgt_importance(torch.cat([hc, sc], -1)))
            strength = 0.5 * torch.sigmoid(self.importance_strength)
            hr, hc = hr * (1 + strength * (2 * gr - 1)), hc * (1 + strength * (2 * gc - 1))
            self.last_importance = (gr.detach(), gc.detach())
        else:
            self.last_importance = None
        r, c = train_edge_index
        pair = torch.cat([hr[r], hc[c], hr[r] * hc[c]], dim=-1)
        scores = self.W_attn_s1(pair).squeeze(-1)
        if not self.use_s1:
            scores = torch.zeros_like(scores)
        s1 = scatter_softmax(scores, c, hc.shape[0])
        s2 = 2 * torch.sigmoid(self.W_attn_s2(pair).squeeze(-1)) if self.use_s2 else torch.ones_like(scores)
        values = F.leaky_relu(self.W_msg(hr), 0.1)[r]
        col_msg = scatter_sum((s1 * s2)[:, None] * values, c, hc.shape[0])
        hcol = self.norm_col(F.leaky_relu(self.W_col_res(hc), 0.1) + col_msg)
        hrow = F.leaky_relu(self.W_row_res(hr), 0.1)
        if self.reverse_messages:
            # Reuse S1 logits, normalised across each source's targets for this direction.
            reverse_s1 = scatter_softmax(scores, r, hr.shape[0])
            reverse_values = F.leaky_relu(self.W_reverse(hc), 0.1)[c]
            hrow = hrow + scatter_sum((reverse_s1 * s2)[:, None] * reverse_values, r, hr.shape[0])
        return s1, s2, self.norm_row(hrow), hcol
