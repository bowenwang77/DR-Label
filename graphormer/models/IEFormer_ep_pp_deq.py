# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch import Tensor
from ..modules.optimizations import weight_norm
from ..modules.visualize import visualize_trace
import math


from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        shrinked = False,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.shrinked = shrinked

    def wnorm(self):
        self.in_proj_net, self.in_proj_fn = weight_norm(module = self.in_proj, names=['weight'], dim = 0)
        self.out_proj_net, self.out_proj_fn = weight_norm(module = self.out_proj, names=['weight'], dim = 0)

    
    def reset(self):
        if 'in_proj_fn' in self.__dict__:
            self.in_proj_fn.reset(self.in_proj_net)
        if 'out_proj_fn' in self.__dict__:
            self.out_proj_fn.reset(self.out_proj_net)

    def forward(
        self,
        query: Tensor,
        # query_init: Tensor = None,
        attn_bias: Tensor = None,
        # attn_bias_init: Tensor=None, 
        # use_input_inj: bool = False,
        node_non_padding_mask = None,
        # update_pos = False,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        output = self.out_proj(attn)
        return output.masked_fill_(~node_non_padding_mask.permute(1,0).unsqueeze(-1),0)


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        gbf_kernels: int = 128,
        explicit_pos: bool = False,
        update_pos: bool = False,
        sphere_pass_origin: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        post_ln: bool = False,
        wnorm: bool = False,
        shrinked: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        # self.sphere_pass_origin = sphere_pass_origin
        self.shrinked= shrinked
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        # self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            shrinked = shrinked,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.post_ln = post_ln



    def wnorm(self, ):
        self.self_attn.wnorm()
        self.fc1_net, self.fc1_fn = weight_norm(module = self.fc1, names=['weight'], dim = 0)
        self.fc2_net, self.fc2_fn = weight_norm(module = self.fc2, names=['weight'], dim = 0)

    def reset(self, ):
        self.self_attn.reset()
        if 'fc1_fn' in self.__dict__:
            self.fc1_fn.reset(self.fc1_net)
        if 'fc2_fn' in self.__dict__:
            self.fc2_fn.reset(self.fc2_net)

    def forward(
        self,
        x: Tensor,
        x_init: Tensor = None,
        delta_pos: Tensor = None, 
        edge_features: Tensor = None,
        attn_bias: Tensor = None,
        # attn_bias_init: Tensor=None,
        use_input_inj: bool = False,
        explicit_pos: bool = False,
        gbf=None,
        bias_proj = None,
        edge_type = None,
        node_non_padding_mask = None,
        edge_non_padding_mask = None,
        non_fix_atom_mask = None,
        final_ln = None,
        # update_pos = False,
        drop_edge_mask = None,
        drop_edge_training = False,
        drop_or_add = False,
    ):  
        if explicit_pos:
            
            if use_input_inj: 
                x = x+x_init
                residual = x
            else:
                residual = x
            if not self.post_ln:
                x = self.self_attn_layer_norm(x)
            x = self.self_attn(
                query= x,
                # query_init = x_init,
                attn_bias=attn_bias,
                # attn_bias_init = attn_bias_init,
                # use_input_inj = use_input_inj,
                node_non_padding_mask =node_non_padding_mask,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            # x = x * self.inv_sqrt_2
            if self.post_ln:
                x = self.self_attn_layer_norm(x)
            residual = x
            if not self.post_ln:
                x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            # x = x * self.inv_sqrt_2
            if self.post_ln:
                x = self.final_layer_norm(x)
            if use_input_inj: 
                x = x-x_init
            return x
        else:
            residual = x
            if not self.post_ln:
                x = self.self_attn_layer_norm(x)
            x = self.self_attn(
                query=x,
                query_init = x_init,
                attn_bias=attn_bias,
                use_input_inj = use_input_inj,
                explicit_pos = explicit_pos,
                gbf = gbf,
                node_non_padding_mask = node_non_padding_mask,
                edge_non_padding_mask = edge_non_padding_mask,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            # x = x * self.inv_sqrt_2
            if self.post_ln:
                x = self.self_attn_layer_norm(x)

            residual = x
            if not self.post_ln:
                x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            # x = x * self.inv_sqrt_2
            if self.post_ln:
                x = self.final_layer_norm(x)
            return x, attn_bias

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def wnorm(self, ):
        self.layer1_net, self.layer1_fn = weight_norm(module = self.layer1, names=['weight'], dim = 0)
        self.layer2_net, self.layer2_fn = weight_norm(module = self.layer2, names=['weight'], dim = 0)

    def reset(self, ):
        if 'layer1_fn' in self.__dict__:
            self.layer1_fn.reset(self.layer1_net)
        if 'layer2_fn' in self.__dict__:
            self.layer2_fn.reset(self.layer2_net)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
        drop_edge_mask: Tensor,
        drop_or_add: bool,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn = attn.view(-1, n_node, n_node) + attn_bias
        # if drop_or_add:
        #     attn.masked_fill_(drop_edge_mask.unsqueeze(0),-float("inf"))
        # else:
        #     attn_add = attn.masked_fill(~drop_edge_mask.unsqueeze(0),float(0))
        #     attn=attn+attn_add
        attn_probs = softmax_dropout(
            attn, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        # attn_probs.masked_fill_(
        #         drop_edge_mask.unsqueeze(0).unsqueeze(0), float(0)
        #     )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        # Sum-based edge operation
        if drop_or_add:
            rot_attn_probs.masked_fill_(
                    drop_edge_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float(0)
                )
            x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        else:
            rot_attn_probs_add = rot_attn_probs.masked_fill(
                    ~drop_edge_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float(0)
                )
            x = (rot_attn_probs+rot_attn_probs_add) @ v.unsqueeze(2)  # [bsz, head , 3, n, d]

        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()

        return cur_force


class NodeTaskHead_FitSphere(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        gbf_kernels: int,
        use_shift_proj: bool,
        sphere_pass_origin: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = NonLinear(embed_dim, gbf_kernels//2)
        self.k_proj: Callable[[Tensor], Tensor] = NonLinear(embed_dim, gbf_kernels//2)
        self.e_proj: Callable[[Tensor], Tensor] = NonLinear(2*gbf_kernels, 1, hidden = gbf_kernels)
        self.gbf_kernels = gbf_kernels
        self.scaling = (embed_dim // gbf_kernels) ** -0.5
        # self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        # self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        # self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        # self.fit_scale: Callable[[Tensor], Tensor] = NonLinear(num_heads,1)
        self.use_shift_proj = use_shift_proj
        self.sphere_pass_origin = sphere_pass_origin

    def scale_calc(self, attn_probs): 
        fit_scale_pred = self.fit_scale(attn_probs.permute(0,2,3,1))
        return fit_scale_pred.squeeze(-1)

    def sphere_fit(self, fit_scale_pred, delta_pos, edge_non_padding_mask, drop_edge_mask, drop_or_add, non_fix_atom_mask):
        ###Get sphere for each node
        non_padding_mask = edge_non_padding_mask
        
        if drop_or_add:
            fit_scale_pred.masked_fill_(
                    drop_edge_mask.unsqueeze(0), float(0)
                )
            point_clouds = delta_pos*fit_scale_pred.unsqueeze(dim=-1)
            effective_edge_count = (non_padding_mask*~drop_edge_mask).sum(dim=2)+1e-5
        else:
            point_clouds = delta_pos*fit_scale_pred.unsqueeze(dim=-1)
            fit_scale_pred_add = fit_scale_pred.masked_fill(
                    ~drop_edge_mask.unsqueeze(0), float(0)
                )
            point_clouds_adding = delta_pos*fit_scale_pred_add.unsqueeze(dim=-1)
            effective_edge_count = (non_padding_mask.long()+drop_edge_mask.long()).masked_fill(~non_padding_mask, 0).sum(dim=2)+1e-5

        untrust_node = effective_edge_count/(non_padding_mask.sum(dim=2)+1e-5)<0.3

        ##Sphere passing origin
        if drop_or_add:
            v_3 = ((point_clouds**2).sum(dim=3).unsqueeze(dim=3)*point_clouds).sum(dim=2)/effective_edge_count.unsqueeze(dim=-1)
            A_1 = (point_clouds.permute(0,1,3,2)@point_clouds)/effective_edge_count.unsqueeze(dim=-1).unsqueeze(dim=-1)
        else:
            v_3 = ((point_clouds**2).sum(dim=3).unsqueeze(dim=3)*point_clouds).sum(dim=2)
            v_3_adding = ((point_clouds_adding**2).sum(dim=3).unsqueeze(dim=3)*point_clouds_adding).sum(dim=2)
            v_3 = (v_3+v_3_adding)/effective_edge_count.unsqueeze(dim=-1)
            A_1 = (point_clouds.permute(0,1,3,2)@point_clouds)
            A_1_adding = (point_clouds_adding.permute(0,1,3,2)@point_clouds_adding)
            A_1 = (A_1+A_1_adding)/effective_edge_count.unsqueeze(dim=-1).unsqueeze(dim=-1)

        A_1[~non_fix_atom_mask]=torch.eye(3, device=A_1.device).unsqueeze(0)
        # A_1 = A_1 + torch.randn_like(A_1)*1e-9 #Avoid singular matrix
        try:
            sphere_center = (2*A_1).float().inverse()@v_3.unsqueeze(-1).float()
            pos_shift = 2* sphere_center.squeeze(-1)

        except RuntimeError:
            is_singular = torch.linalg.det(A_1).abs() < 1e-15
            A_1[is_singular] = torch.eye(3, device=A_1.device).unsqueeze(0)

            sphere_center = (2*A_1).float().inverse()@v_3.unsqueeze(-1).float()
            pos_shift = 2* sphere_center.squeeze(-1)
            # Handle cases where A_1 is not invertible (i.e., singular)
            # Find the indices of the singular matrices
            # singular_indices = torch.nonzero(is_singular, as_tuple=True)[0]
            print("is_singular_shape", is_singular.shape,"\nis_sigular_sum",is_singular.sum(dim=-1), \
                  "\nsingular_indices\n", torch.nonzero(is_singular, as_tuple=True)[0],"\n",torch.nonzero(is_singular, as_tuple=True)[0])
            # Set the corresponding pos_shift to zero
            pos_shift[is_singular] = 0.

        pos_shift[untrust_node] = 0
        pos_shift[~non_fix_atom_mask] = 0
        return pos_shift

    def forward(
        self,
        query: Tensor,
        edge_features: Tensor,
        delta_pos: Tensor,
        # edge_coefficient: Tensor,
        edge_non_padding_mask: Tensor,
        drop_edge_mask: Tensor,
        non_fix_atom_mask: Tensor,
        drop_or_add: bool,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(query)
        qs = q.unsqueeze(-2).expand(-1,-1,n_node,-1)        
        ks = k.unsqueeze(-2).expand(-1,-1,n_node,-1)     
        e_emb = F.gelu(torch.cat([qs,ks.permute(0,2,1,3)], dim=-1))
        

        fit_scale_pred = self.e_proj(torch.cat([e_emb,edge_features],dim = -1)).squeeze(-1)
        fit_scale_pred = fit_scale_pred.masked_fill(~edge_non_padding_mask,0)

        node_pos_pred = self.sphere_fit(fit_scale_pred,delta_pos, edge_non_padding_mask, drop_edge_mask, drop_or_add, non_fix_atom_mask) # [bsz, n, n], [bsz,n,n,3] -> [bsz, n, 3].  The final prediction of atomic shift.




        """
        to supervise the contact map: output fit_scale_pred
        """
        if self.use_shift_proj:
            return node_pos_pred, fit_scale_pred
        else:
            return node_pos_pred

@register_model("IEFormer_ep_pp_deq")
class Graphormer3D(BaseFairseqModel):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument("--blocks", type=int, metavar="L", help="num blocks")
        parser.add_argument(
            "--embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--min-node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--num-kernel",
            type=int,
        )
        parser.add_argument(
            "--deq-mode",
            default = False,
            action = 'store_true',
        )
 
        parser.add_argument(
            "--pretrain-step",
            type=int,
            default = 0,
        )
        parser.add_argument(
            "--compute-jac-loss",
            default = False,
            action = 'store_true',
        )
        parser.add_argument(
            "--jac-loss-weight",
            type = float,
            metavar="D",
            help="loss weight for jacobian loss",
        )
        parser.add_argument(
            "--SAA-idx",
            type = int,
            default = 1,
        )
        parser.add_argument(
            "--post-ln",
            default = False,
            action = 'store_true',
        )
        parser.add_argument(
            "--wnorm",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--use-fit-sphere",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--use-shift-proj",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--edge-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for edge fitting",
        )
        parser.add_argument(
            "--min-edge-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for edge fitting",
        )
        parser.add_argument(
            "--no-node-mask",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--use-unnormed-node-label",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--get-solver-trace",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--sphere-pass-origin",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--get-inter-pos-trace",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--visualize",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--noisy-nodes",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--noisy-node-weight",
            type=float,
            metavar="D",
            help="weight of noisy node loss",
        )
        parser.add_argument(
            "--noisy-nodes-rate",
            type=float,
            metavar="D",
            help="fraction of noisy node",
        )
        parser.add_argument(
            "--noise-scale",
            type=float,
            metavar="D",
            help="scale of node noise",
        )
        parser.add_argument(
            "--noise-type",
            type=str,
            help="type of added noise",
        )
        parser.add_argument(
            "--noise-deltapos-normed",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--noise-in-traj",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--noisy-label",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--noisy-label-downscale",
            type=float,
            metavar="D",
            help="scale of node noise",
        )
        parser.add_argument(
            "--use-input-inj",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--full-dataset",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--explicit-pos",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--shrinked",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--pos-update-freq",
            type = int,
            default = 1,
        )
        parser.add_argument(
            "--remove-outliers",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--edge-dropout-rate",
            type=float,
            metavar="D",
            help="edge dropout rate for ablation",
        )
        parser.add_argument(
            "--l2-node-loss",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--drop-edge-training",
            default = False,
            action = 'store_true'
        )
        parser.add_argument(
            "--drop-or-add",
            default = False,
            action = 'store_true',
            help="True: drop edge. False: add edge",
        )
        parser.add_argument(
            "--explicit-inter-pos",
            default = False,
            action = 'store_true',
            help="True: drop edge. False: add edge",
        )
        parser.add_argument(
            "--train_url",
            default= None,
            type = str,
            help="save dir",
        )
        parser.add_argument(
            "--data_url",
            default=None,
            type = str,
            help="dataset dir",
        )
        parser.add_argument(
            "--fix-atoms",
            default = False,
            action = 'store_true',
            help="True: drop edge. False: add edge",
        )
        parser.add_argument(
            "--no-tail",
            default = False,
            action = 'store_true',
            help="True: no tail, only blocks. False: with tail",
        )
        parser.add_argument(
            "--geo-guided-deq",
            default = False,
            action = 'store_true',
            help="True:geo-guided deq break",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.atom_types = 64
        self.edge_types = 64 * 64
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.args.embed_dim, padding_idx=0
        )
        self.tag_encoder = nn.Embedding(3, self.args.embed_dim)
        self.input_dropout = self.args.input_dropout

        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    num_attention_heads=self.args.attention_heads,
                    gbf_kernels = self.args.num_kernel,
                    explicit_pos = self.args.explicit_pos,
                    update_pos = layer_id%self.args.pos_update_freq==(self.args.pos_update_freq-1),
                    sphere_pass_origin = self.args.sphere_pass_origin,
                    dropout=self.args.dropout,
                    attention_dropout=self.args.attention_dropout,
                    activation_dropout=self.args.activation_dropout,
                    post_ln = self.args.post_ln,
                    wnorm = self.args.wnorm,
                    shrinked = self.args.shrinked,
                )
                for layer_id in range(self.args.layers)
            ]
        )

        if self.args.wnorm:
            for layers in self.layers:
                layers.wnorm()

        self.deq_mode = self.args.deq_mode
        self.pretrain_step = self.args.pretrain_step
        self.compute_jac_loss = self.args.compute_jac_loss
        self.jac_loss_weight = self.args.jac_loss_weight
        self.use_fit_sphere = self.args.use_fit_sphere
        self.use_shift_proj = self.args.use_shift_proj
        self.use_input_inj = self.args.use_input_inj
        self.explicit_pos = self.args.explicit_pos
        self.pos_update_freq = self.args.pos_update_freq
        self.fix_atoms = self.args.fix_atoms
        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.args.embed_dim)
        self.drop_edge_training = self.args.drop_edge_training
        self.drop_or_add = self.args.drop_or_add
        gbf_kernels = self.args.num_kernel
        self.embedding_dim = self.args.embed_dim
        self.shrinked = self.args.shrinked
        self.no_tail = self.args.no_tail
        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, 1
        )
        if self.args.wnorm:
            self.engergy_proj.wnorm()
        self.energe_agg_factor: Callable[[Tensor], Tensor] = nn.Embedding(3, 1)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        K = self.args.num_kernel

        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayer(K, self.edge_types)
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.args.attention_heads
        )
        if self.args.wnorm:
            self.bias_proj.wnorm()
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.args.embed_dim)
        if not self.no_tail:
            if self.use_fit_sphere:
                self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead_FitSphere(
                    self.args.embed_dim, self.args.num_kernel, self.args.use_shift_proj, self.args.sphere_pass_origin
                )
                # self.edge_factor: Callable[[Tensor], Tensor] = nn.Embedding(10, 1)
                # nn.init.normal_(self.edge_factor.weight, 0, 0.5)
            else:
                self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead(
                    self.args.embed_dim, self.args.attention_heads
                )
        # if self.use_input_inj:
            # self.input_injection: Callable[[Tensor], Tensor] = nn.Linear(
            #     self.args.embed_dim, self.args.embed_dim * 3, bias=False
            # )

        if self.explicit_pos:
            if not self.args.shrinked:
                self.q_proj: Callable[[Tensor], Tensor] = NonLinear(self.embedding_dim, gbf_kernels//2)
                self.k_proj: Callable[[Tensor], Tensor] = NonLinear(self.embedding_dim, gbf_kernels//2)
            else:
                self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(self.embedding_dim, gbf_kernels//2)
            self.e_proj: Callable[[Tensor], Tensor] = NonLinear(2*gbf_kernels, 1, hidden = gbf_kernels)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        return super().set_num_updates(num_updates)
    
    def sphere_fit(self, fit_scale_pred, delta_pos, edge_non_padding_mask, drop_edge_mask, drop_or_add, non_fix_atom_mask):
        ###Get sphere for each node
        non_padding_mask = edge_non_padding_mask
        
        if drop_or_add:
            fit_scale_pred.masked_fill_(
                    drop_edge_mask.unsqueeze(0), float(0)
                )
            point_clouds = delta_pos*fit_scale_pred.unsqueeze(dim=-1)
            effective_edge_count = (non_padding_mask*~drop_edge_mask).sum(dim=2)+1e-5
        else:
            point_clouds = delta_pos*fit_scale_pred.unsqueeze(dim=-1)
            fit_scale_pred_add = fit_scale_pred.masked_fill(
                    ~drop_edge_mask.unsqueeze(0), float(0)
                )
            point_clouds_adding = delta_pos*fit_scale_pred_add.unsqueeze(dim=-1)
            effective_edge_count = (non_padding_mask.long()+drop_edge_mask.long()).masked_fill(~non_padding_mask, 0).sum(dim=2)+1e-5

        untrust_node = effective_edge_count/(non_padding_mask.sum(dim=2)+1e-5)<0.3

        ##Sphere passing origin
        if drop_or_add:
            v_3 = ((point_clouds**2).sum(dim=3).unsqueeze(dim=3)*point_clouds).sum(dim=2)/effective_edge_count.unsqueeze(dim=-1)
            A_1 = (point_clouds.permute(0,1,3,2)@point_clouds)/effective_edge_count.unsqueeze(dim=-1).unsqueeze(dim=-1)
        else:
            v_3 = ((point_clouds**2).sum(dim=3).unsqueeze(dim=3)*point_clouds).sum(dim=2)
            v_3_adding = ((point_clouds_adding**2).sum(dim=3).unsqueeze(dim=3)*point_clouds_adding).sum(dim=2)
            v_3 = (v_3+v_3_adding)/effective_edge_count.unsqueeze(dim=-1)
            A_1 = (point_clouds.permute(0,1,3,2)@point_clouds)
            A_1_adding = (point_clouds_adding.permute(0,1,3,2)@point_clouds_adding)
            A_1 = (A_1+A_1_adding)/effective_edge_count.unsqueeze(dim=-1).unsqueeze(dim=-1)

        A_1[~non_fix_atom_mask]=torch.eye(3, device=A_1.device).unsqueeze(0)
        # A_1 = A_1 + torch.randn_like(A_1)*1e-9 #Avoid singular matrix
        try:
            sphere_center = (2*A_1).float().inverse()@v_3.unsqueeze(-1).float()
            pos_shift = 2* sphere_center.squeeze(-1)

        except RuntimeError:
            is_singular = torch.linalg.det(A_1).abs() < 1e-15
            A_1[is_singular] = torch.eye(3, device=A_1.device).unsqueeze(0)

            sphere_center = (2*A_1).float().inverse()@v_3.unsqueeze(-1).float()
            pos_shift = 2* sphere_center.squeeze(-1)

            print("is_singular_shape", is_singular.shape,"\nis_sigular_sum",is_singular.sum(dim=-1), \
                  "\nsingular_indices\n", torch.nonzero(is_singular, as_tuple=True)[0],"\n",torch.nonzero(is_singular, as_tuple=True)[0])

            pos_shift[is_singular] = 0.


        pos_shift[untrust_node] = 0
        pos_shift[~non_fix_atom_mask] = 0
        return pos_shift
    
    def deq_func(self,encoding, encoding_init, 
            use_input_inj, explicit_pos, gbf,bias_proj, 
            edge_type,node_non_padding_mask ,
            edge_non_padding_mask,non_fix_atom_mask, final_ln,
            drop_edge_mask, drop_edge_training, drop_or_add):
        layer_id = 0
        if use_input_inj:
            pos_init = encoding_init[:,:,-3:]
        for enc_layer in self.layers:

            if layer_id%self.args.pos_update_freq==0:
                n_node, _, _ = encoding.size()
                if use_input_inj:
                    pos = encoding_init[:,:,-3:]+encoding[:,:,-3:]
                    pos_prev = pos
                else:
                    pos = encoding[:,:,-3:]
                    pos_prev = pos
                delta_pos = pos.permute(1,0,2).unsqueeze(1) - pos.permute(1,0,2).unsqueeze(2)
                dist: Tensor = delta_pos.norm(dim=-1)
                delta_pos =delta_pos/(dist.unsqueeze(-1) + 1e-8) 
                gbf_feature = self.gbf(dist, edge_type)
                edge_features = gbf_feature.masked_fill(
                    ~edge_non_padding_mask.unsqueeze(-1), 0.0
                )        
                attn_bias = self.bias_proj(edge_features).permute(0, 3, 1, 2).contiguous()

                attn_bias.masked_fill_(
                    ~node_non_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_bias = attn_bias.view(-1, n_node, n_node)

            encoding = encoding[:,:,:-3]
            if use_input_inj:
                encoding_init = encoding_init[:,:,:-3]

            encoding = enc_layer(encoding, encoding_init,delta_pos,
                edge_features, attn_bias=attn_bias,
                # attn_bias_init=attn_bias_init,
                use_input_inj=use_input_inj, explicit_pos= explicit_pos, 
                gbf = gbf, bias_proj=bias_proj, edge_type=edge_type,
                node_non_padding_mask=node_non_padding_mask ,
                edge_non_padding_mask=edge_non_padding_mask ,non_fix_atom_mask=non_fix_atom_mask,
                final_ln = final_ln,
                drop_edge_mask = drop_edge_mask,
                drop_edge_training = drop_edge_training,
                drop_or_add = drop_or_add,
                )

            ##Pos update
            if layer_id%self.args.pos_update_freq==self.args.pos_update_freq-1:
                n_node, n_graph, embed_dim = encoding.size()
                if use_input_inj:
                    encoding = encoding+encoding_init
                if not self.shrinked:
                    query_pos = final_ln(encoding)
                    q_pos = self.q_proj(query_pos)
                    k_pos = self.k_proj(query_pos)
                    qs = q_pos.unsqueeze(-2).expand(-1,-1,n_node,-1).permute(1,0,2,3)       
                    ks = k_pos.unsqueeze(-2).expand(-1,-1,n_node,-1).permute(1,0,2,3)     
                    e_emb = F.gelu(torch.cat([qs,ks.permute(0,2,1,3)], dim=-1))
                else:
                    q_pos = self.q_proj(encoding)
                    qs = q_pos.unsqueeze(-2).expand(-1,-1,n_node,-1).permute(1,0,2,3)   
                    e_emb = F.gelu(torch.cat([qs,qs.permute(0,2,1,3)], dim=-1))
                fit_scale_pred = self.e_proj(torch.cat([e_emb,edge_features],dim = -1)).squeeze(-1)
                fit_scale_pred = fit_scale_pred.masked_fill(~non_fix_atom_mask.unsqueeze(-1),0)
                if not drop_edge_training:
                    drop_edge_mask = torch.zeros_like(drop_edge_mask)
                node_pos_pred = self.sphere_fit(fit_scale_pred,delta_pos, edge_non_padding_mask, drop_edge_mask, drop_or_add, non_fix_atom_mask)
                #denormalize?
                # pos_adding = node_pos_pred.masked_fill(~non_fix_atom_mask.unsqueeze(-1), 0)
                small_mol_mask = non_fix_atom_mask.sum(-1) < 7
                if small_mol_mask.sum()>0:
                    print("removed_small_mol",non_fix_atom_mask.sum(-1))
                node_pos_pred[small_mol_mask] = 0
                pos = pos+node_pos_pred.permute(1,0,2)
                # pos = pos.permute(1,0,2)
            if use_input_inj:
                encoding = torch.cat([encoding,pos-pos_init],dim=-1)
                encoding_init = torch.cat([encoding_init,pos_init],dim=-1)
            else:
                encoding = torch.cat([encoding,pos],dim=-1)
            layer_id += 1
        if use_input_inj:
            pos_prev = pos_prev-pos_init
        return encoding, pos_prev, fit_scale_pred

    def forward(self, atoms: Tensor, tags: Tensor, pos: Tensor, real_mask: Tensor, step: int =-1, noisy_pos: Tensor=None, noisy_label_pos: Tensor = None):
        node_non_padding_mask = atoms.ne(0)
        if self.fix_atoms:
            non_fix_atom_mask = tags.ne(0)
        else: 
            non_fix_atom_mask = node_non_padding_mask.clone()
        non_self_loop_mask = 1-torch.eye(atoms.shape[1],device = node_non_padding_mask.device)
        edge_non_padding_mask = (atoms.float().unsqueeze(-1)@atoms.float().unsqueeze(-2))>0
        edge_non_padding_mask = edge_non_padding_mask * non_self_loop_mask.bool()

        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1)

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node
        )

        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(
            ~edge_non_padding_mask.unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        if self.args.wnorm: ##use here since it seems each wnorm corresponds to a reset func.
            for layers in self.layers:
                layers.reset()

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        ) #the initial embedding for nodes. 
        output = output.transpose(0, 1).contiguous()
        jac_loss = torch.tensor(0.0).to(output)

        # When deq-mode or input injection mode, the output represents hidden embedding. 
        # In normal mode, output is equal to output, and output is set to None.
        if self.use_input_inj:
            output_init = output.clone()
            output = torch.zeros_like(output)
            # if set initial state to be 0, write here: output= torch.zeros_like(output)
            # output = self.input_injection(output)
            if self.explicit_pos:
                output_init = torch.concat([output_init,pos.permute(1,0,2).clone()],dim=2)
                output = torch.concat([output,torch.zeros_like(pos.permute(1,0,2))],dim=2)
        else:
            output_init = None
            if self.explicit_pos:
                output = torch.concat([output,pos.permute(1,0,2)],dim=2)

        f_deq_nstep,f_deq_shrink_ratio, f_deq_residual, edge_output, edge_target_mask = -1,-1,-1,-1,-1
        tags1=tags.float()+1
        edge_type_mask = tags1.unsqueeze(-1)@tags1.unsqueeze(-2)
        edge_target_mask = edge_non_padding_mask*edge_type_mask

        ##Dropping edge for ablation:graph_attn_bias, edge_target_mask
        drop_edge_mask = torch.zeros(n_node,n_node).to(edge_target_mask.device)<self.args.edge_dropout_rate
        if self.args.edge_dropout_rate>0:
            drop_edge_mask = torch.rand(n_node,n_node).to(edge_target_mask.device)<self.args.edge_dropout_rate
            edge_target_mask.masked_fill_(drop_edge_mask.unsqueeze(0), float(0))

        self.drop_edge_training = self.drop_edge_training and self.training
        inter_pos_trace = []
        if not (self.args.deq_mode and step > self.args.pretrain_step):
            for _ in range(self.args.blocks):
                output, pos_prev ,fit_scale_pred = self.deq_func(output,output_init, 
                                        self.use_input_inj,self.explicit_pos,
                                        self.gbf,self.bias_proj, edge_type,
                                        node_non_padding_mask ,edge_non_padding_mask, non_fix_atom_mask,
                                        self.final_ln, drop_edge_mask, self.drop_edge_training, self.drop_or_add)
                if self.args.get_inter_pos_trace:
                    if self.use_intput_inj:
                        inter_pos_trace.append(output+output_init)
                    else:
                        inter_pos_trace.append(output)
            if self.use_input_inj:
                new_output = output+output_init
            else:
                new_output = output

        if self.explicit_pos:
            final_output = self.final_ln(new_output[:,:,:-3])
            final_output = final_output.transpose(0, 1)
            final_pos = new_output[:,:,-3:].permute(1,0,2)
            final_pos_base = pos_prev.permute(1,0,2)+pos
            if self.no_tail: 
                delta_pos = final_pos_base.unsqueeze(1) - final_pos_base.unsqueeze(2)
            else:
                delta_pos = final_pos.unsqueeze(1) - final_pos.unsqueeze(2)
            dist: Tensor = delta_pos.norm(dim=-1)
            delta_pos = -delta_pos/(dist.unsqueeze(-1) + 1e-8)
            if not self.no_tail:
                gbf_feature = self.gbf(dist, edge_type)
                edge_features = gbf_feature.masked_fill(
                    ~edge_non_padding_mask.unsqueeze(-1), 0.0
                )        
                attn_bias = self.bias_proj(edge_features).permute(0, 3, 1, 2).contiguous()
                # graph_attn_bias.masked_fill_(
                #     ~edge_non_padding_mask.unsqueeze(1), float("-inf")
                # )
                # graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
                attn_bias.masked_fill_(
                    ~node_non_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_bias = attn_bias.view(-1, n_node, n_node)
        else:
            final_output = self.final_ln(new_output)
            final_output = final_output.transpose(0, 1)

        if self.args.get_inter_pos_trace:
            if self.explicit_pos:
                if self.explicit_inter_pos:
                    deltapos_trace = [inter_pos[:,:,-3:].transpose(0,1) - pos for inter_pos in inter_pos_trace]
                    # deltapos_trace.append(node_output)
                else:
                    deltapos_trace=[]
                    for i in range(len(inter_pos_trace)):
                        inter_pos_trace[i]=self.final_ln(inter_pos_trace[i][:,:,:-3]).transpose(0,1)
                        deltapos_trace.append(self.node_proc(inter_pos_trace[i], edge_features, delta_pos, edge_non_padding_mask)[0])
            else:
                deltapos_trace=[]
                for i in range(len(inter_pos_trace)):
                    inter_pos_trace[i]=self.final_ln(inter_pos_trace[i]).transpose(0,1)
                    deltapos_trace.append(self.node_proc(inter_pos_trace[i], edge_features, delta_pos, edge_non_padding_mask)[0])
            deltapos_trace = torch.cat(deltapos_trace,dim=1).view(deltapos_trace[0].shape[0],-1,deltapos_trace[0].shape[1],deltapos_trace[0].shape[2]) 

        eng_output = F.dropout(final_output, p=0.1, training=self.training)
        eng_output = (
            self.engergy_proj(eng_output) * self.energe_agg_factor(tags)
        ).flatten(-2) 
        output_mask = (
            tags > 0
        ) & real_mask  # no need to consider padding, since padding has tag 0, real_mask False
        eng_output *= output_mask #only consider center cell surface+adsorbate
        eng_output = eng_output.sum(dim=-1)

        if not self.no_tail:
            if self.use_shift_proj:
                # edge_coefficient = self.edge_factor(edge_target_mask.long()).squeeze()
                node_output,fit_scale_pred = self.node_proc(final_output, gbf_feature, delta_pos, edge_non_padding_mask, drop_edge_mask, non_fix_atom_mask, self.drop_or_add)
                # node_output = node_output.masked_fill(~non_fix_atom_mask.unsqueeze(-1), 0)
                fit_scale_pred = fit_scale_pred.masked_fill(~non_fix_atom_mask.unsqueeze(-1),0)
                if self.explicit_pos:
                    node_output = final_pos+node_output-pos
            else:
                node_output = self.node_proc(final_output, attn_bias, delta_pos, drop_edge_mask, self.drop_or_add)
                node_output = node_output.masked_fill(~non_fix_atom_mask.unsqueeze(-1), 0)

        node_target_mask = output_mask.unsqueeze(-1)


        if self.no_tail:
            node_output = final_pos - pos
        output_vals = {'eng_output':eng_output, 'node_output':node_output, 'node_target_mask': node_target_mask,'edge_dirs':delta_pos}
        if self.deq_mode:
            output_vals['f_deq_shrink_ratio'] = f_deq_shrink_ratio
        if self.compute_jac_loss:
            output_vals['jac_loss']=jac_loss.view(-1,1)
            output_vals['f_deq_nstep']=f_deq_nstep
            output_vals['f_deq_residual']=f_deq_residual
        if self.use_shift_proj:
            output_vals['fit_scale_pred']=fit_scale_pred
            output_vals['edge_target_mask']=edge_target_mask

        if self.explicit_pos:
            if self.no_tail:
                output_vals['final_delta_pos']=pos_prev.permute(1,0,2)
            else:
                output_vals['final_delta_pos']=final_pos-pos
        if self.drop_edge_training:
            output_vals['drop_edge_mask']=drop_edge_mask

        return output_vals


@register_model_architecture("IEFormer_ep_pp_deq", "IEFormer_ep_pp_deq")
def base_architecture(args):
    args.blocks = getattr(args, "blocks", 4)
    args.layers = getattr(args, "layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
    args.attention_heads = getattr(args, "attention_heads", 48)
    args.input_dropout = getattr(args, "input_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.jac_loss_weight = getattr(args, "jac_loss_weight", 15)
    args.node_loss_weight = getattr(args, "node_loss_weight", 15)
    args.min_node_loss_weight = getattr(args, "min_node_loss_weight", 1)
    args.edge_loss_weight = getattr(args, "edge_loss_weight", 30)
    args.min_edge_loss_weight = getattr(args, "min_edge_loss_weight", 2)
    args.noisy_node_weight = getattr(args, "noisy_node_weight", 1)
    args.noisy_label_downscale = getattr(args, "noisy_label_downscale", 0.05)
    args.eng_loss_weight = getattr(args, "eng_loss_weight", 1)
    args.num_kernel = getattr(args, "num_kernel", 128)
    args.edge_dropout_rate = getattr(args, "edge_dropout_rate", -1)

