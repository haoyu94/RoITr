# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
from model.transformer.positional_encoding import PPFStructualEmbedding, GeometricStructureEmbedding
from model.transformer.attention import AttentionLayer, AttentionOutput, RPEAttentionLayer, LocalRPEAttentionLayer
import torch


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))
########################################################################################################################
# PPFTransformer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', with_cross_pos_embed=False):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout, with_cross_pos_embed=with_cross_pos_embed)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)
        self.with_cross_pos_embed = with_cross_pos_embed

    def forward(
        self,
        input_states,
        memory_states,
        inputs_pos_embed,
        memory_pos_embed,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        if inputs_pos_embed is not None:
            inputs_pos_embed = torch.max(inputs_pos_embed, dim=-2)[0]
        if memory_pos_embed is not None:
            memory_pos_embed = torch.max(memory_pos_embed, dim=-2)[0]

        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            inputs_pos_embed,
            memory_pos_embed,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        with_cross_pos_embed=False
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        self.with_cross_pos_embed = with_cross_pos_embed

        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn, with_cross_pos_embed=with_cross_pos_embed))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.with_cross_pos_embed:
                    feats0, scores0 = self.layers[i](feats0, feats1, embeddings0, embeddings1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, embeddings1, embeddings0, memory_masks=masks0)
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, None, None, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, None, None, memory_masks=masks0)

            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PPFTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        dropout=None,
        activation_fn='ReLU',
        with_cross_pos_embed=False
    ):
        r"""Geometric Transformer (GeoTransformer).
        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            activation_fn: activation function
        """
        super(PPFTransformer, self).__init__()

        self.embedding = PPFStructualEmbedding(hidden_dim, mode='global')

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, with_cross_pos_embed=with_cross_pos_embed
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_feats,
        src_feats,
        ref_ppfs,
        src_ppfs,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer
        Args:
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_ppfs (Tensor): (B, N, K, 4)
            src_ppfs (Tensor): (B, M, K, 4)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)
        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_feats = ref_feats.unsqueeze(0)
        src_feats = src_feats.unsqueeze(0)
        ref_ppfs = ref_ppfs.unsqueeze(0)
        src_ppfs = src_ppfs.unsqueeze(0)

        ref_embeddings = self.embedding(ref_ppfs)
        src_embeddings = self.embedding(src_ppfs)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )
        ref_feats = self.out_proj(ref_feats).squeeze(0)
        src_feats = self.out_proj(src_feats).squeeze(0)

        return ref_feats, src_feats


class LocalPPFTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        dropout=None
    ):
        r"""Geometric Transformer (GeoTransformer).
        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            activation_fn: activation function
        """
        super(LocalPPFTransformer, self).__init__()

        self.embedding = PPFStructualEmbedding(hidden_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = LocalRPEAttentionLayer(d_model=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        feats,
        node_idx,
        group_idx,
        ppfs
    ):
        r"""Geometric Transformer
        Args:
            feats (Tensor): (N, in_dim)
            node_idx: (M,)
            group_idx: (M, K)
            ppfs (Tensor): (M, K, 4)
        Returns:
            new_feats: torch.Tensor (M, C2)
        """
        pos_embeddings = self.embedding(ppfs) #[M, K, hidden_dims]
        feats = self.in_proj(feats) #[N, in_dim] -> [N, hidden_dim]
        new_feats, _ = self.transformer(
            feats,
            pos_embeddings,
            node_idx,
            group_idx
        )
        new_feats = self.out_proj(new_feats)

        return new_feats
