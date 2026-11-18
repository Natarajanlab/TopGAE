"""TopGAE: topology-informed, regulatory influence-weighted graph autoencoder."""
import torch
from torch import nn

from .layers import DualScoreLayer


class TopGAE(nn.Module):
    """Graph autoencoder for directed GRN reconstruction.

    Encoder: one :class:`~topgae.layers.DualScoreLayer` followed by separate
    linear projections into source (TF) and target latent spaces.
    Decoders: an asymmetric bilinear link decoder (``z_i^T W z_j + b``, so
    i -> j and j -> i score differently) and a shared feature decoder that
    reconstructs masked expression profiles.

    Parameters
    ----------
    in_features : int
        Number of cells (columns of the expression matrix).
    out_feature : int
        Hidden width of the dual-score layer.
    latent_dim : int
        Latent dimension of the source / target embeddings.
    dropout : float
        Input dropout rate.
    use_importance : bool
        Use structural (degree + HITS) role-specific gates.
    reverse_messages : bool
        Send target -> source messages in the reverse pass.
    use_s1, use_s2 : bool
        Enable the S1 (allocation) and S2 (modulation) edge scores.
    """

    def __init__(self, in_features, out_feature=128, latent_dim=64, dropout=0.2,
                 use_importance=True, reverse_messages=True, use_s1=True, use_s2=True):
        super().__init__()
        self.in_features = in_features
        # Attribute names are kept stable so saved state_dicts remain loadable.
        self.ds_layer = DualScoreLayer(in_features, out_feature, dropout, use_importance,
                                       reverse_messages, use_s1, use_s2)
        self.latent_row, self.latent_col = nn.Linear(out_feature, latent_dim), nn.Linear(out_feature, latent_dim)
        self.decoder = nn.Bilinear(latent_dim, latent_dim, 1)
        self.decoder_feat = nn.Sequential(nn.Linear(latent_dim, out_feature), nn.SiLU(),
                                          nn.Linear(out_feature, in_features))
        self.S1_output = self.S2_output = None

    def encode(self, x_row, x_col, train_edge_index, row_genes, col_genes,
               skip_core=False, structure=None, store_scores=True):
        """Return source and target latent embeddings ``(Z_row, Z_col)``.

        With ``store_scores=True`` the dense S1 / S2 matrices are kept in
        ``self.S1_output`` / ``self.S2_output``. Zero outside message edges
        means "not evaluated", not a zero predicted link.
        """
        s1, s2, hr, hc = self.ds_layer(x_row, x_col, train_edge_index, row_genes,
                                       col_genes, skip_core, structure)
        if store_scores:
            self.S1_output = x_row.new_zeros(x_row.shape[0], x_col.shape[0])
            self.S2_output = torch.zeros_like(self.S1_output)
            r, c = train_edge_index
            self.S1_output[r, c] = s1.detach()
            self.S2_output[r, c] = s2.detach()
        return self.latent_row(hr), self.latent_col(hc)

    def decode_link(self, Z_row, Z_col, edge_index):
        """Link logits for the (source, target) pairs in ``edge_index``."""
        return self.decoder(Z_row[edge_index[0]], Z_col[edge_index[1]]).squeeze(-1)

    def decode_all(self, Z_row, Z_col):
        """Dense (n_sources x n_targets) link-logit matrix.

        Exact bilinear decoder without an N_source x N_target x latent_dim expansion.
        """
        return Z_row @ self.decoder.weight[0] @ Z_col.T + self.decoder.bias[0]

    def decode_feat(self, Z_row, Z_col):
        """Reconstruct expression profiles for both roles."""
        return self.decoder_feat(Z_row), self.decoder_feat(Z_col)
