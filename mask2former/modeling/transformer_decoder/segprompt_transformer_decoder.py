# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.comm import get_local_rank,get_world_size,all_gather
from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
import torchshow 
from mask2former.data.lvis_info.lvis_categories_tr_part50 import IN_VOC_ID_SET,OUT_VOC_ID_SET
from mask2former.data.lvis_info.lvis_categories_tr import ID2NAME,NAME2ID
import os
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class SegPrompt_MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        example_query_sigma: float = 0.9,
        example_query_proj: bool = True,
        example_query_loss: float = 0.0,
        example_query_batchwise_loss: float = 0.0,
        use_voc_emb = True,
        save_path: str ,
        use_clip_textemb: bool,
        enforce_clip_proj: bool,
        open_classification: bool,
        clip_logit_weight: tuple,
        open_att_mask: bool,
        query_repeat: int,
        addtional_query: bool,
        open_branch_detached: bool,
        open_self_att: bool,
        no_att_mask: bool,
        self_att_independent: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_self_attention_layers_open = nn.ModuleList()
        #self.transformer_cross_attention_layers_open2img = nn.ModuleList()
        #self.transformer_ffn_layers_open = nn.ModuleList()
        self.counter = 0
        self.segma = example_query_sigma
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            # self.transformer_ffn_layers_open.append(
            #     FFNLayer(
            #         d_model=hidden_dim,
            #         dim_feedforward=dim_feedforward,
            #         dropout=0.0,
            #         normalize_before=pre_norm,
            #     )
            # )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.example_query_loss = example_query_loss
        self.example_query_loss_batchwise = example_query_batchwise_loss
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.use_voc_emb = use_voc_emb
        if self.use_voc_emb:
            if use_clip_textemb:
            # load clip text emb
                textemb = torch.load('weight/lvis1203_clip_text.pth',map_location='cpu') # 1203 * 512
                # padding the text_emb for id 1203
                textemb = torch.cat([textemb, torch.zeros(1204-textemb.shape[0], textemb.shape[1])], dim=0)
                if enforce_clip_proj:
                    # 512 -> hidden_dim
                    self.clip_proj = nn.Linear(512, hidden_dim)
                    self.id2feat = nn.Sequential(nn.Embedding.from_pretrained(textemb, freeze=True), self.clip_proj)
                    self.id2pe =  nn.Sequential(nn.Embedding.from_pretrained(textemb, freeze=True), self.clip_proj)
                else:
                    self.id2feat = nn.Sequential(nn.Embedding.from_pretrained(textemb, freeze=True))
                    self.id2pe =  nn.Sequential(nn.Embedding.from_pretrained(textemb, freeze=True)) # 512
            else:
                self.id2feat = nn.Embedding(1204,hidden_dim)
                self.id2pe = nn.Embedding(1204,hidden_dim)
            
        self.memory_bank = nn.Embedding(1204,hidden_dim)
        self.memory_bank.requires_grad_(False)
        self.memory_bank.weight.zero_()
        self.ex_proj = example_query_proj
        self.ex2pe = nn.Linear(hidden_dim,hidden_dim) if self.ex_proj else nn.Sequential()
        self.ex2feat = nn.Linear(hidden_dim,hidden_dim) if self.ex_proj else nn.Sequential()
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.open_classification = open_classification
        if open_classification:
            self.open_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.save_path = save_path + '/'
        # save query features
        self.extract = False
        self.ex_banks = []
        self.ex_id = 132 - 1
        self.ex_counter = 0
        self.open_att_mask = open_att_mask
        self.no_att_mask = no_att_mask
        
        self.init0 = False
        self.query_repeat = query_repeat
        self.addtional_query = addtional_query
        if self.query_repeat != 1 and self.addtional_query:
            self.query_repeat_feat = nn.Embedding(self.query_repeat, hidden_dim)
            self.query_repeat_pos = nn.Embedding(self.query_repeat, hidden_dim)
        self.detach_open = open_branch_detached
        self.self_att_independent = self_att_independent

        

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["example_query_sigma"] = cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_SIGMA
        ret["example_query_proj"] = cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_PROJ
        ret["save_path"] = cfg.OUTPUT_DIR
        ret["example_query_loss"] = cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_LOSS
        ret["example_query_batchwise_loss"] = cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_BATCHWISE_LOSS
        ret["use_voc_emb"] = cfg.MODEL.MASK_FORMER.USE_VOC_EMB
        ret["use_clip_textemb"] = cfg.MODEL.MASK_FORMER.USE_CLIP_TEXTEMB
        ret["enforce_clip_proj"] = cfg.MODEL.MASK_FORMER.ENFORCE_CLIP_PROJ
        ret["open_classification"] = cfg.MODEL.MASK_FORMER.OPEN_CLASSIFICATION
        ret["clip_logit_weight"] = cfg.MODEL.MASK_FORMER.CLIP_LOGIT_WEIGHT
        ret["open_att_mask"] = cfg.MODEL.MASK_FORMER.OPEN_ATT_MASK
        ret['query_repeat'] = cfg.MODEL.MASK_FORMER.QUERY_REPEAT
        ret['addtional_query'] = cfg.MODEL.MASK_FORMER.ADDTIONAL_QUERY
        ret['open_branch_detached'] = cfg.MODEL.MASK_FORMER.OPEN_BRANCH_DETACHED  
        ret['open_self_att'] = cfg.MODEL.MASK_FORMER.OPEN_SELF_ATT
        ret['no_att_mask'] = cfg.MODEL.MASK_FORMER.NO_ATT_MASK
        ret['self_att_independent'] = cfg.MODEL.MASK_FORMER.SELF_ATT_INDEPENDENT
        
        return ret

    def forward(self, x, mask_features, mask = None,candidate_ids = None,masks_e = None,labels_e = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # check labels_e is a tuple
        if isinstance(labels_e, tuple):
            ann_ids = labels_e[1]
            output_path = labels_e[2]
            img_path = labels_e[3]
            img_id = img_path.split('/')[-1].split('.')[0]
            labels_e = labels_e[0]
        else:
            ann_ids = None
        # disable mask, it does not affect performance
        # del mask
        # zmz here we want to use masks to constraint

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # query_embed.zero_()
        # output.zero_()
        if self.init0:   
            query_embed.zero_()
            output.zero_()
        if self.query_repeat == 1:
            if self.use_voc_emb:
                voc_pe = self.id2pe(candidate_ids) #NXQXC
                voc_output = self.id2feat(candidate_ids) #NXQXC
            else:
                voc_pe = 0
                voc_output = 0
            ex_pe = self.ex2pe(self.memory_bank(candidate_ids)) #NXQXC
            # voc_pe = 0
            # voc_output = 0
            voc_pe = ex_pe + voc_pe #NXQXC
            
            ex_output = self.ex2feat(self.memory_bank(candidate_ids))
            voc_output = ex_output + voc_output
        else:
            # use_voc_emb = True
            rand_ints = []
            for i in range(candidate_ids.shape[1]//self.query_repeat):
                ranges = output.shape[0]
                if not self.addtional_query:
                    rand_int = torch.randint(0, ranges, (self.query_repeat,))
                else: 
                    # 0~9
                    rand_int = torch.arange(0, self.query_repeat)
                while(len(rand_int)!=len(rand_int.unique())):
                    rand_int = torch.randint(0, ranges, (self.query_repeat,))
                rand_ints.append(rand_int)
            rand_ints = torch.cat(rand_ints, dim=0)
            rand_ints = rand_ints.unsqueeze(0).repeat(bs,1).to(candidate_ids.device)
            if self.use_voc_emb:
                voc_pe = self.id2pe(candidate_ids) #NXQXC
                voc_output = self.id2feat(candidate_ids) #NXQXC
            else:
                voc_pe = 0
                voc_output = 0
            ex_pe = self.ex2pe(self.memory_bank(candidate_ids)) #NXQXC
            # voc_pe = 0
            # voc_output = 0
            voc_pe = ex_pe + voc_pe #NXQXC
            
            ex_output = self.ex2feat(self.memory_bank(candidate_ids))
            voc_output = ex_output + voc_output
            if not self.addtional_query:
                if not self.detach_open:
                    voc_pe = voc_pe + self.query_embed(rand_ints)
                    voc_output = voc_output + self.query_feat(rand_ints)
                else:
                    voc_pe = voc_pe + self.query_embed(rand_ints).detach()
                    voc_output = voc_output + self.query_feat(rand_ints).detach()
            else:
                voc_pe = voc_pe + self.query_repeat_pos(rand_ints)
                voc_output = voc_output + self.query_repeat_feat(rand_ints)
        voc_pe = voc_pe.permute(1,0,2) #QxNXC
        voc_output = voc_output.permute(1,0,2)
        
        # output[0,:,:] = 0
        # output[:,:,:] = 0
        # mq = output.mean(0).squeeze()
        # mp = query_embed.mean(0).squeeze()
        #query_embed[:,:,:] = 0
        # output[0,:,:] = mq
        ori_query_num = query_embed.shape[0]
        if labels_e !=  None:
            zero_query_num = labels_e.shape[1] # temporily
        else:
            zero_query_num = 0
        voc_query_num = voc_pe.shape[0]
        if zero_query_num < voc_query_num:
            zero_output = torch.zeros_like(voc_output)[:zero_query_num]
            zero_pe = torch.zeros_like(voc_pe)[:zero_query_num]
        else:
            # zero_query_num x C
            zero_output = torch.zeros((zero_query_num, voc_output.shape[1], voc_output.shape[2])).to(voc_output.device)
            zero_pe = torch.zeros((zero_query_num, voc_pe.shape[1], voc_pe.shape[2])).to(voc_pe.device)

        total_query_num = ori_query_num + voc_query_num + zero_query_num
        self_att_mask = torch.zeros((total_query_num,total_query_num)).to(output.device)
        if not self.no_att_mask:
            # oriquery cannot see open_query or ex_query
            self_att_mask[:ori_query_num,ori_query_num:] = 1.0
            # vocquery cannot see example_query
            self_att_mask[ori_query_num:ori_query_num+voc_query_num,ori_query_num+voc_query_num:] = 1.0
            if self.self_att_independent:
            # vocquery cannot see ori_query
                self_att_mask[ori_query_num:ori_query_num+voc_query_num,:ori_query_num] = 1.0
            # vocquery cannot see other vocquery
            
            # example_query  canonly see themselves , TODO,probaly we should consider pading
            
            self_att_mask[ori_query_num+voc_query_num:,:ori_query_num+voc_query_num] = 1.0
            # only itself
            #self_att_mask[ori_query_num+voc_query_num:,:] = 1.0
            # all self
            #self_att_mask[:,:] = 1.0
            if self.open_att_mask:
                self_att_mask[ori_query_num:ori_query_num+voc_query_num,ori_query_num:ori_query_num+voc_query_num] = 1.0
                if self.query_repeat == 1:
                    self_att_mask[torch.arange(0,self_att_mask.shape[0]),torch.arange(0,self_att_mask.shape[0])] = 0
                else: 
                    # 保证类内可见，类间不可见  
                    for i in range(ori_query_num, self_att_mask.shape[0], self.query_repeat):
                        self_att_mask[i:i+self.query_repeat,i:i+self.query_repeat] = 0
            # visualize self_att_mask as  a heatmap and save it
        if self.counter == 0:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set(font_scale=1.2)
            plt.figure(figsize=(10,10))
            sns.heatmap(self_att_mask.cpu().numpy(),cmap='Blues',annot=False)
            plt.savefig(self.save_path+'self_att_mask_exp3.png')
        self_att_mask = self_att_mask > 0
        predictions_class = []
        predictions_mask = []
        openvoc_class = []
        openvoc_mask = []
        example_mask = []
        example_class =     []
        output = torch.cat((output,voc_output),0)
        output = torch.cat((output,zero_output),0)
        query_embed = torch.cat((query_embed,voc_pe),0)
        query_embed = torch.cat((query_embed,zero_pe),0)
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class[:,:ori_query_num,:])
        predictions_mask.append(outputs_mask[:,:ori_query_num,:])
        if self.open_classification:
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_open(output, mask_features, attn_mask_target_size=size_list[0])

        openvoc_class.append(outputs_class[:,ori_query_num:total_query_num-zero_query_num,:]) 
        openvoc_mask.append(outputs_mask[:,ori_query_num:total_query_num-zero_query_num,:])
        example_mask.append(outputs_mask[:,total_query_num-zero_query_num:,:])
        example_class.append(outputs_class[:,total_query_num-zero_query_num:,:])
        # self.memory_bank +=  get_local_rank()
        # memory_bank = all_gather(self.memory_bank)
        # print("rank{}:{}".format(get_local_rank(),memory_bank))
        # memory_bank = torch.stack(memory_bank) # node * 1204*hid_dim
        # memory_bank = torch.sum(memory_bank,0)
        self.counter += 1
        assert output.shape[0] == self_att_mask.shape[0]
        #torchshow.save(masks_e[-1]>0,'example_itself_debug/'+str(self.counter)+'mask.jpg')
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            temp_size = size_list[(i) % self.num_feature_levels]
            if mask:
                # this is to use img-t as mask
                attn_mask_r = F.interpolate(masks_e.float(), size=temp_size, mode="bilinear", align_corners=False)#BS=1 X N_GT X H X W
                attn_mask_r = (attn_mask_r.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) <= 0).bool()
                attn_mask_r = attn_mask_r.detach()  # (8xbs)xnum_qx256
                # here we only focus on first query
                attn_mask[:,-zero_query_num:,:] = attn_mask_r  # (8xbs)xnum_qx(16*16) 0~7 all same
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # avoid empty attention,but would broke gt
            # attention: cross-attention first

            
            
            #output = torch.cat((output,voc_output),0)
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_att_mask, # zmz use att_mask
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class[:,:ori_query_num,:])
            predictions_mask.append(outputs_mask[:,:ori_query_num,:])
            if self.open_classification:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_open(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            # if outputs_class[:,ori_query_num:total_query_num-zero_query_num,:].shape[1] > 30:
            #     print(2)
            openvoc_class.append(outputs_class[:,ori_query_num:total_query_num-zero_query_num,:]) 
            openvoc_mask.append(outputs_mask[:,ori_query_num:total_query_num-zero_query_num,:])
            example_class.append(outputs_class[:,total_query_num-zero_query_num:,:])
            example_mask.append(outputs_mask[:,total_query_num-zero_query_num:,:])
        if labels_e !=  None and self.counter <= 9999999:
            example_query = output[-zero_query_num:]
            #example_query[torch.where(labels_e == id_set[0])[1],torch.where(labels_e == id_set[0])[0],:].shape
            if ann_ids != None:
                # save example query for each anno
                assert len(ann_ids) == zero_query_num
                # 判断 ann_ids 是严格递增的
                ori_ann_ids = ann_ids
                ann_ids = sorted(ann_ids)
                assert ori_ann_ids == ann_ids
                
                output_dir = os.path.join(output_path,'query')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # for i in range(zero_query_num):
                #     torch.save(example_query[i], os.path.join(output_dir, str(ann_ids[i])+'.pt'))
                # save a total example query for each image
                torch.save(example_query, os.path.join(output_dir, str(img_id)+'.pt'))
            example_queries =  all_gather(example_query)
            example_queries = [x.detach().cpu() for x in example_queries]
            example_queries = torch.cat(example_queries,0)
            labels_es = all_gather(labels_e)
            labels_es = [x.detach().cpu() for x in labels_es]
            labels_es = torch.cat(labels_es,1)
            id_set , id_fre = torch.unique(labels_es, return_counts = True)
            pos_class_num = (id_set<1203).sum().item()
            class_emb_bank = torch.zeros(1204, example_query.shape[-1]) # 1204 * hid_dim
            flag = True
            for i in range(len(id_set)):
                id=id_set[i]
                temp = example_queries[torch.where(labels_es == id)[1],torch.where(labels_es == id)[0],:]
                assert temp.shape[0] == id_fre[i]
                class_emb_bank[id] = torch.mean(temp,0).to(output.device)
                if self.extract:
                    if id == self.ex_id:
                        # save example query to self.ex_banks
                        self.ex_banks.append(class_emb_bank[id])
                        flag = False
                        self.ex_counter += 1
                        assert class_emb_bank[id].max()!= 0, "example query is zero"
                    if self.ex_counter == 500:
                        # save self.ex_banks to file
                        torch.save(self.ex_banks,self.save_path + str(self.init0) + '_ex_banks.pt')
                        exit()
            if self.example_query_loss:
                # compute similarity between example query and self.memory_bank
                example_query_emb = class_emb_bank[id_set].to(output.device)   # PN*256 ,should we consider the padding class?
                example_query_emb = example_query_emb/torch.norm(example_query_emb,dim=1,keepdim=True)
                # get similarity between example query and self.memory_bank
                # select the self.memory_bank which is IN_VOC
                memory_bank_emb = self.memory_bank.weight/(self.memory_bank.weight.norm(dim=1, keepdim=True)+1e-8)
                seen_class_sum = (self.memory_bank.weight.norm(dim=1, keepdim=True)>0).sum()
                if seen_class_sum > 80:
                    # compute cosine similarity between example query and self.memory_bank
                    similarity = torch.matmul(example_query_emb,memory_bank_emb.transpose(0,1)) #PN*1204
                    # compute cosine similarity between example query and corresponding class in self.memory_bank
                    similarity_class = torch.matmul(example_query_emb,memory_bank_emb[id_set].transpose(0,1)) #PN*1204
                    # compute binary cross entropy loss
                    # prepare labels
                    labels = torch.zeros(similarity.shape[0],1204)
                    
                    labels[range(len(id_set)),id_set] = 1
                                                

                    labels = labels.to(output.device)
                    loss = F.binary_cross_entropy_with_logits(similarity,labels)/pos_class_num
                    loss *= self.example_query_loss
                    if self.example_query_loss_batchwise:
                        inbatch_labels = torch.zeros(similarity_class.shape)
                        inbatch_labels[range(len(id_set)),range(len(id_set))] = 1
                        inbatch_labels = inbatch_labels.to(output.device)
                        inbatch_loss = F.binary_cross_entropy_with_logits(similarity_class,inbatch_labels)/pos_class_num + \
                            F.binary_cross_entropy_with_logits(similarity_class.T,inbatch_labels)/pos_class_num
                        inbatch_loss *= self.example_query_loss_batchwise
                    else:
                        inbatch_loss = 0

                else:
                    loss = 0
                    inbatch_loss = 0
            else: 
                loss = 0
                inbatch_loss = 0


            # update memory_bank
            for id in id_set:
                if self.memory_bank.weight[id].max() == 0:
                    self.memory_bank.weight[id] = class_emb_bank[id]
                self.memory_bank.weight[id] = self.segma*self.memory_bank.weight[id] + (1-self.segma)*class_emb_bank[id].to(output.device)
            # optimize for to matrix operation
            
            if flag and self.extract:
                print("id not found")
            #class_emb_bank = class_emb_bank.detach()
            assert len(predictions_class) == self.num_layers + 1
            #memorybank = all_gather(self.memory_bank.weight)
        # assert memorybank[0].cpu().sum()==memorybank[1].cpu().sum()
        # assert memorybank[0].cpu().min()==memorybank[1].cpu().min()
        else :
            loss = 0
            inbatch_loss = 0

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'aux_outputs_open': self._set_aux_loss(
                openvoc_class if self.mask_classification else None, openvoc_mask
            ),
             'output' : output,
             #'pred_masks_all':predictions_mask,
             'openvoc_class':openvoc_class,
             'openvoc_mask': openvoc_mask,
            'example_mask' :example_mask[-1],
            'example_class' : example_class[-1],
            'aux_outputs_example': self._set_aux_loss(
                example_class if self.mask_classification else None, example_mask
            ),
            "example_query_loss":loss,
            "example_query_loss_inbatch":inbatch_loss,

        }
        # out = {
        #     'pred_logits': predictions_class[8],
        #     'pred_masks': predictions_mask[8],
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class if self.mask_classification else None, predictions_mask
        #     ),
        #       'output' : output
        # }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
         # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def forward_prediction_heads_open(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.open_class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)


        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
