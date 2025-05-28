# Copyright (c) Facebook, Inc. and its affiliates.
from cProfile import label
from html.entities import name2codepoint
from tempfile import tempdir
from typing import Tuple
import time

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
import torchshow
import copy
from mask2former.data.lvis_info import lvis_categories_tr_part50
from mask2former.modeling import meta_arch
from .modeling.criterion import SetCriterion,SetCriterion_Open,SetCriterion_Example
from .modeling.matcher import HungarianMatcher , HungarianMatcherClasswise
#from torch_scatter import scatter
from detectron2.modeling import build_model
import logging
from mask2former.data.lvis_info.lvis_categories_tr_part50 import IN_VOC_ID_SET,OUT_VOC_ID_SET,COCO_ID_SET,OUT_COCO_ID_SET
from mask2former.data.lvis_info.lvis_categories_005 import ID_005_SET
from mask2former.data.lvis_info.lvis_categories_010 import ID_010_SET
from mask2former.data.lvis_info.lvis_categories_020 import ID_020_SET
from mask2former.data.lvis_info.lvis_categories_r import ID_R_SET
from mask2former.data.lvis_info.lvis_categories_tr import ID2NAME,NAME2ID
import random
import os,csv
def get_coords(H, W, device):
        """H, W are the resolution of the first level"""

        x_range = torch.linspace(0.5, W - 0.5, W, device=device)
        y_range = torch.linspace(0.5, H - 0.5, H, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        locations = torch.stack((x, y), dim=-1).view(1, -1, 2)
        return locations

@META_ARCH_REGISTRY.register()
class SegPrompt(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        criterion2: nn.Module,
        criterion3: nn.Module,
        num_queries: int,
        num_open_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        meta_architecture: str,
        binary_thres : float,
        save_dir: str,
        freeze_backbone: bool,
        query_repeat: int,
        use_clip_imgfeat: bool,
        use_lvis: bool,
        only_coco: bool,
        train_dataset: str,
        delete_mask: bool,
        binary_class: bool,
        min_mask_area: int,
        new_mask_idset: bool,
        example_sup : bool,
        few_shot : bool,
        max_example_num : int,
        open_class: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False  
        # if few_shot==2 :
        #     # freeze the model
        #     for p in self.parameters():
        #         p.requires_grad = False   
        self.criterion = criterion
        self.criterion2 = criterion2
        self.criterion3 = criterion3
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        # add for glip training
        self.meta_architecture = meta_architecture
        self.counter = 0
        self.binary_thres =  binary_thres
        if self.binary_thres > 0:
            # when binary_thres > 0, which means we use binary train.
             self.binary = True
        else:
             self.binary = False
        self.save_path = save_dir + '/'
        #  create a new folder to save the results
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # save the eval result
        self.num_open_queries = num_open_queries
        self.per_class_true_positive = torch.zeros(1203, dtype=torch.int64)
        self.per_class_false_positive = torch.zeros(1203, dtype=torch.int64)
        self.per_class_false_negative = torch.zeros(1203, dtype=torch.int64)
        self.per_class_true_negative = torch.zeros(1203, dtype=torch.int64)
        self.per_class_total = torch.zeros(1203, dtype=torch.int64)
        self.id2visualize = IN_VOC_ID_SET  | OUT_VOC_ID_SET #[968,995,137,138,132,1162,411,547] #no need to -1
        self.img_id = 0
        self.query_repeat = query_repeat
        self.use_clip_imgfeat = use_clip_imgfeat
        if only_coco: 
            self.in_voc_id_set = COCO_ID_SET
            self.out_voc_id_set = set()
        else:
            if use_lvis:
                if '_80' in train_dataset:
                    self.in_voc_id_set = COCO_ID_SET
                    self.out_voc_id_set = set()
                elif 'fre' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET  | OUT_VOC_ID_SET
                    self.out_voc_id_set = set() 
                    print(train_dataset)
                elif 'shot' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET  | OUT_VOC_ID_SET
                    self.out_voc_id_set = set() 
                elif 'only' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = set()
                elif 'coco_2017_train' in train_dataset:
                    # 1~80
                    self.in_voc_id_set = set(range(1,82))
                    self.out_voc_id_set = set()
                elif 'rm005' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = OUT_VOC_ID_SET - ID_005_SET
                elif 'rm010' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = OUT_VOC_ID_SET - ID_010_SET
                elif 'rm020' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = OUT_VOC_ID_SET - ID_020_SET
                elif 'rm_r' in train_dataset:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = OUT_VOC_ID_SET - ID_R_SET
                elif train_dataset == 'lvis_v1_train':
                    self.in_voc_id_set = IN_VOC_ID_SET  | OUT_VOC_ID_SET
                    self.out_voc_id_set = set()  
                elif 'ade' in train_dataset:
                    self.in_voc_id_set = set(range(1,101))
                    self.out_voc_id_set = set()
                else:
                    self.in_voc_id_set = IN_VOC_ID_SET
                    self.out_voc_id_set = OUT_VOC_ID_SET
                
            else:
                self.in_voc_id_set = COCO_ID_SET
                self.out_voc_id_set = OUT_COCO_ID_SET
           

            print ('in_voc_id_set', len(self.in_voc_id_set))
            print ('out_voc_id_set', len(self.out_voc_id_set))
        self.train_dataset = train_dataset

        self.delete_mask = delete_mask
        self.binary_class = binary_class
        self.min_mask_area = min_mask_area
        self.new_mask_idset = new_mask_idset
        self.example_sup = example_sup
        self.few_shot = few_shot
        self.max_example_num = max_example_num
      
        self.open_class = open_class


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        open_no_object_weight = cfg.MODEL.MASK_FORMER.OPEN_NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        c,d,m = cfg.MODEL.MASK_FORMER.OPEN_BRANCH_WEIGHT
        class_weight2 = c*class_weight
        dice_weight2 = d*dice_weight
        mask_weight2 = m*mask_weight
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        matcher2 = HungarianMatcherClasswise(
            cost_class=class_weight2,
            cost_mask=mask_weight2,
            cost_dice=dice_weight2,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict2 = {"loss_ce2": class_weight2, "loss_mask2": mask_weight2, "loss_dice2": dice_weight2}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            aux_weight_dict2 = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            for i in range(dec_layers - 1):
                aux_weight_dict2.update({k + f"_{i}": v for k, v in weight_dict2.items()})
            weight_dict2.update(aux_weight_dict2)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        criterion2 = SetCriterion_Open(
            sem_seg_head.num_classes,
            matcher=matcher2,
            weight_dict=weight_dict2,
            eos_coef=open_no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            query_repeat=cfg.MODEL.MASK_FORMER.QUERY_REPEAT,
            only_match = cfg.MODEL.MASK_FORMER.QUERY_REPEAT_ONLY_MATCH,
            binary_class = cfg.MODEL.MASK_FORMER.BINARY_CLASSIFICATION
        )
        if cfg.MODEL.MASK_FORMER.EXAMPLE_SUP:
            criterion3 =  SetCriterion_Example(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses= losses ,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )
        use_lvis = False if 'remove' in cfg.DATASETS.TRAIN[0] else True
        only_coco = True if 'lvisfied' in cfg.DATASETS.TRAIN[0] else False
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "criterion2": criterion2,
            "criterion3": criterion3 if cfg.MODEL.MASK_FORMER.EXAMPLE_SUP else None,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "num_open_queries": cfg.MODEL.MASK_FORMER.NUM_OPEN_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # for glip training
            "meta_architecture" : cfg.MODEL.META_ARCHITECTURE,
            # for binary training
            "binary_thres" : cfg.TEST.BINARY_THRES,
            # output path
            "save_dir": cfg.OUTPUT_DIR,
            "freeze_backbone": cfg.MODEL.MASK_FORMER.FREEZE_BACKBONE,
            'query_repeat': cfg.MODEL.MASK_FORMER.QUERY_REPEAT,
            'use_clip_imgfeat': cfg.MODEL.MASK_FORMER.USE_CLIP_IMGFEAT,
            "use_lvis": use_lvis,
            "only_coco": only_coco,
            "train_dataset": cfg.DATASETS.TRAIN[0],
            "delete_mask"   : cfg.MODEL.MASK_FORMER.DELETE_MASK,
            "binary_class"  : cfg.MODEL.MASK_FORMER.BINARY_CLASSIFICATION,
            'min_mask_area' : cfg.MODEL.MASK_FORMER.MIN_MASK_AREA,
            'new_mask_idset': cfg.MODEL.MASK_FORMER.NEW_MASK_IDSET,
            # cfg.MODEL.MASK_FORMER.EXAMPLE_SUP
            'example_sup'   : cfg.MODEL.MASK_FORMER.EXAMPLE_SUP,
            'few_shot'      : cfg.MODEL.MASK_FORMER.TEST.FEW_SHOT,
            'max_example_num': cfg.MODEL.MASK_FORMER.MAX_EXAMPLE_NUM,
            'open_class'    : cfg.MODEL.MASK_FORMER.OPEN_CLASSIFICATION,

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """


        
        # origin path for mask2former
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.pixel_mean.sum() < 3:
                images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images] 
        else:   
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        ori_img_size = images[0].shape[-2:]
        images = ImageList.from_tensors(images, self.size_divisibility)
        if self.training:
            assert ori_img_size == images.tensor.shape[-2:]
        features = self.backbone(images.tensor)
        self.counter += 1  
        max_candi_num = self.num_open_queries
        min_mask_area = self.min_mask_area
        max_example_num = self.max_example_num
        bs = features['res2'].shape[0]
        if "instances" in batched_inputs[0] and self.training:
            if 'logit' in batched_inputs[0].keys():
                # for binary training
                logit = batched_inputs[0]['logit']
                logit = logit.to(self.device)
                # torchshow.show(logit)
                # # show img and logit together
                # torchshow.show(images.tensor+logit)
  
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            if len(self.out_voc_id_set):
                not_exhaustive_category_ids = [x["not_exhaustive_category_ids"]for x in batched_inputs]
                neg_category_ids = [x["neg_category_ids"]for x in batched_inputs]
            else:
                not_exhaustive_category_ids = [[] for x in batched_inputs]
                neg_category_ids = [[] for x in batched_inputs]
            img_id = [x.get("image_id", -1) for x in batched_inputs]
            self.img_id = img_id
            masks = [(target['masks']) for target in targets]
            labels = [(target['labels']) for target in targets]
            masks_inv = []
            labels_inv = []
            targets_inv = []
            masks_ouv = []
            labels_ouv = []
            targets_ouv = []
            targets_example = []
            candidate_ids = []
            candidate_labels = []
            mask_loss_masks = []
            skip_first_loss = False
            logits = []
            for i in range(len(labels)):
                label_list = labels[i].tolist()
                #keep = [ True if id+1 in IN_VOC_ID_SET else False for id in label_list]
                keep = [ True if label_list[k]+1 in self.in_voc_id_set and masks[i][k].sum()> min_mask_area else False for k in range(len(label_list))]
                #keep = [ True if  masks[i][k].sum()> min_mask_area else False for k in range(len(label_list))]
                # filter the mask not in VOC OR TOO small
                mask_inv = masks[i][keep]
                label_inv = labels[i][keep]  # for example query extraction
                targets_inv.append({'masks':masks[i][keep],'labels':labels[i][keep]}) # in vocabulary, use in compute loss
                if 'logit' in batched_inputs[0].keys():
                    # add pseudo mask to mask_inv
                    not_keep = [ True if label_list[k]+1 not in self.in_voc_id_set and pseudo_mask[k].sum() > min_mask_area else False for k in range(len(label_list))]
                    mask_inv = torch.cat([mask_inv,pseudo_mask[not_keep]],dim=0)
                    label_inv = torch.cat([label_inv,labels[i][not_keep]],dim=0)

                label_unique,indexs = torch.unique(label_inv,return_inverse=True)
                if len(label_inv): # ensure we at least has one gt
                   
                    id2index  = torch.tensor([torch.where(label_inv == id)[0][0] for id in label_unique]) # first consider unique id
                    mask_unique = mask_inv[id2index,:,:]
                if len(label_inv)>=max_example_num:
                    if len(label_unique) >= max_example_num :
                        mask_inv = mask_unique
                        label_inv = label_unique
                        # random sample
                        index = random.sample(range(len(label_unique)),max_example_num)
                        mask_inv = mask_inv[index,:,:]   
                        label_inv = label_inv[index]  
                    else: 
                        #randomly choose to pad
                        idpool = set(range(len(label_inv))) - set(id2index.tolist())
                        index = random.sample(idpool,max_example_num-len(label_unique))
                        mask_inv = torch.cat((mask_unique,mask_inv[index,:,:]),0)
                        label_inv = torch.cat((label_unique,label_inv[index]),0)
                    targets_example.append({'masks':mask_inv,'labels':label_inv})
                else:
                    targets_example.append({'masks':mask_inv,'labels':label_inv})
                    label_pad_zero = torch.ones(max_example_num-len(label_inv),dtype=torch.long).to(self.device)*1203 # pad with 1203 
                    mask_pad_zero = torch.zeros((max_example_num-len(label_inv),mask_inv.shape[1],mask_inv.shape[2]),dtype=torch.float).to(self.device)
                    mask_inv = torch.cat((mask_inv,mask_pad_zero),dim=0)
                    label_inv = torch.cat((label_inv,label_pad_zero),dim=0)
                
                masks_inv.append(mask_inv)
                labels_inv.append(label_inv)   # use to extract example query
                
                if True not in  keep:
                    #print('no mask')
                    skip_first_loss = True
                keep = [ True if id+1 in self.out_voc_id_set else False for id in label_list]
                masks_ouv.append(masks[i][keep])
                labels_ouv.append(labels[i][keep])
                targets_ouv.append({'masks':masks[i][keep],'labels':labels[i][keep]}) # out vocbulary
                candidate_id_pos = torch.cat((label_unique,labels_ouv[i].unique()))
               
                if not_exhaustive_category_ids[i]:
                    candidate_id_pos = torch.cat((candidate_id_pos,(torch.tensor(not_exhaustive_category_ids[i])-1).to(self.device))).unique()
                if len(candidate_id_pos) > max_candi_num:
                    candidate_id_pos = candidate_id_pos[:max_candi_num]
                candidate_id_neg = torch.tensor(neg_category_ids[i]).unique().to(self.device) - 1
                # candidate_id = torch.cat((candidate_id_pos,candidate_id_neg-1)) # not ensure no overlap
                # # add random negative
                # candidate_id_neg = torch.tensor(neg_category_ids[i]).unique().to(self.device)
                
                if self.query_repeat > 1:
                    # repeat self.query_repeat times
                    candidate_id_pos = candidate_id_pos.repeat(self.query_repeat)
                    # continious repeat
                    candidate_id_pos = candidate_id_pos[torch.argsort(candidate_id_pos)]
                    candidate_id_neg = candidate_id_neg.repeat(self.query_repeat)
                    candidate_id_neg = candidate_id_neg[torch.argsort(candidate_id_neg)]
                    
                candidate_id = torch.cat((candidate_id_pos,candidate_id_neg))
                neg_id_set = (self.in_voc_id_set | self.out_voc_id_set) - set(not_exhaustive_category_ids[i])-\
                    set(neg_category_ids[i]) - set((candidate_id_pos+1).tolist())
                if len(candidate_id) < max_candi_num:
                    neg_pad = torch.tensor(random.sample(list(neg_id_set),(max_candi_num-len(candidate_id))//self.query_repeat+1)).to(self.device)-1
                    if self.query_repeat > 1:
                        neg_pad = neg_pad.repeat(self.query_repeat)
                        neg_pad = neg_pad[torch.argsort(neg_pad)]
                    candidate_id = torch.cat((candidate_id,neg_pad))
                #assert len(candidate_id) == len(candidate_id_neg) + len(candidate_id_pos) # bottle conflict here
                assert candidate_id.max() < 1203 and candidate_id.min()>=0 , 'label id out of bound'
                if len(candidate_id) >= max_candi_num:
                    candidate_id = candidate_id[:max_candi_num]
                else:
                    pad_zero = torch.ones(max_candi_num - len(candidate_id)).to(self.device)*1203
                    candidate_id = torch.cat((candidate_id,pad_zero))
                # mask_id_set = IN_VOC_ID_SET - set((torch.tensor(not_exhaustive_category_ids[i])).tolist()) -\
                # set((candidate_id_neg).tolist())    # record which id has mask gt
                mask_id_set = set((labels_inv[i]+1).unique().tolist()) - set((torch.tensor(not_exhaustive_category_ids[i])).tolist()) -\
                set(neg_category_ids[i])    # record which id has mask gt
                # change to new version, 11.14
                if self.new_mask_idset:
                    mask_id_set = set((label_unique+1).tolist()) - set((torch.tensor(not_exhaustive_category_ids[i])).tolist()) -\
                    set(neg_category_ids[i])    # record which id has mask gt
                mask_loss_mask = torch.tensor([ True if int(id)+1 in mask_id_set else False for id in candidate_id])
                mask_loss_masks.append(mask_loss_mask)
                # if mask_loss_mask.sum() == 0:
                #     losses = {} 
                #     print('no mask gt to be train')
                #     for k in self.criterion.weight_dict:
                #         losses[k] = 0*features['res2'].sum()
                #     return losses
                candidate_ids.append(candidate_id)
                candidate_id2label = torch.zeros(len(candidate_id)).to(self.device)
                candidate_id2label[len(candidate_id_pos):] = 1 # 0 means pos ,1 means neg
                candidate_labels.append(candidate_id2label)
            candidate_ids = torch.stack(candidate_ids,0).long()
            candidate_labels = torch.stack(candidate_labels,0)
            mask_loss_masks = torch.stack(mask_loss_masks,0)
            masks_inv = torch.stack(masks_inv,0)
            labels_inv = torch.stack(labels_inv,0)
            
        else:
            max_candi_num = 10
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets_test(gt_instances, images)
                ann_ids = batched_inputs[0]['ann_ids'] if 'ann_ids' in batched_inputs[0] else None
                labels = [(target['labels']) for target in targets]
                if len(labels):
                    assert len(labels[0]) == len(ann_ids)
                masks = [(target['masks']) for target in targets]
                assert len(labels) == len(masks)
                if len(labels) == 0:
                    candidate_ids = torch.zeros((bs,max_candi_num)).long().to(self.device)
                    labels_inv = candidate_ids
                    # if self.id2visualize: # use specific ids to visualize
                    #     candidate_ids[:,:len(self.id2visualize)] = torch.tensor(self.id2visualize).to(self.device)
                    candidate_labels = torch.ones((bs,max_candi_num)).float().to(self.device)
                    masks  = None
                else:
                    candidate_ids = torch.stack([labels[i].unique() for i in range(len(labels))],0).long()
                    pos_num  = candidate_ids.shape[1]
                    neg_id_set = (self.in_voc_id_set | self.out_voc_id_set) - set((torch.unique(candidate_ids)+1).tolist())
                    if pos_num > max_candi_num:
                        candidate_ids = candidate_ids[:,:max_candi_num]
                        pos_num = max_candi_num
                    else:
                        neg_pad = torch.tensor(random.sample(list(neg_id_set),(max_candi_num-pos_num)//self.query_repeat+1)).to(self.device)
                        candidate_ids = torch.cat((candidate_ids,neg_pad.unsqueeze(0)-1),1)
                    if not candidate_ids.shape[-1]:
                        print("empty candidate_ids")
                        # padding to avoid empty candidate_ids
                        candidate_ids = torch.ones(1,1).long().to(self.device)*1203
                    candidate_labels = torch.zeros(candidate_ids.shape).to(self.device)
                    candidate_labels[:,pos_num:] = 1
                    only_pos = False
                    if only_pos:
                        candidate_ids = candidate_ids[:,:pos_num]
                        candidate_labels = candidate_labels[:,:pos_num]
                        if not candidate_ids.shape[-1]:
                            print("empty candidate_ids")
                            # padding to avoid empty candidate_ids
                            candidate_ids = torch.ones(1,1).long().to(self.device)*1203
                            candidate_labels = torch.zeros(candidate_ids.shape).to(self.device)

                    if self.query_repeat > 1:
                        # repeat self.query_repeat times
                        candidate_ids = candidate_ids.repeat(1,self.query_repeat)
                        # continious repeat
                        candidate_ids = candidate_ids[:,torch.argsort(candidate_ids)]
                        # squeeze
                        candidate_ids = candidate_ids.squeeze(0).long()
                        candidate_labels = candidate_labels.repeat(1,self.query_repeat)
                        candidate_labels = candidate_labels[:,torch.argsort(candidate_labels)]
                        candidate_labels = candidate_labels.squeeze(0)

                    labels_inv = torch.stack(labels,0)
                    if ann_ids:
                        images_path = batched_inputs[0]['file_name']
                        assert len(ann_ids) == labels_inv.shape[1]
                        labels_inv = (labels_inv,ann_ids,self.save_path,images_path)
                    #candidate_labels = torch.zeros(candidate_ids.shape).to(self.device)
            else:
                candidate_ids = torch.zeros((bs,max_candi_num)).long().to(self.device)
                labels_inv = candidate_ids

                candidate_labels = torch.ones((bs,max_candi_num)).float().to(self.device)
                masks  = None
            mask_loss_masks = None
            if masks is not None:
                masks_inv = torch.stack(masks,0)
            else:
                masks_inv = None


        outputs = self.sem_seg_head(features,masks,candidate_ids,masks_inv,labels_inv) 

       
        output_query = outputs['output']
        
        if self.training: 
            # mask classification target
            # if "instances" in batched_inputs[0]:
            #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #     targets = self.prepare_targets(gt_instances, images)
            # else:
            #     targets = None
            
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            N,Q,H,W = mask_pred_results.shape
            mask_result = mask_pred_results.permute(0,2,3,1)
            mask_result = mask_result.view(N,H*W,Q)
            mask_in_one = mask_result.argmax(dim=2)
            self.counter += 1

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            final_query = outputs["output"]
          
            if self.delete_mask:
                targets_inv2 = copy.deepcopy(targets_inv)
            
            if self.binary_class:
                for target in targets_inv:
                        target['labels'].zero_()
            #print(self.counter)
            # if self.counter == 7:
            #     print('stop for debug')
            if self.example_sup :
                fakeoutput = {}
                fakeoutput['pred_logits'] = outputs['example_class']
                fakeoutput['pred_masks'] = outputs['example_mask']
                fakeoutput['aux_outputs'] = outputs['aux_outputs_example']
                if self.binary_class:
                    for target in targets_example:
                            target['labels'].zero_()
                losse3 = self.criterion3(fakeoutput, targets_example) 

            if self.delete_mask:
                losses2 = self.criterion2(outputs,targets_inv2,candidate_ids,candidate_labels,mask_loss_masks)
            else:
                if self.open_class:
                    for target in targets:
                        target['labels'].zero_()
                if "coco_2017_train" in self.train_dataset:
                    # remove id > 79
                    for target in targets:
                        index = target['labels'] < 80
                        target['labels'] = target['labels'][index]
                        target['masks'] = target['masks'][index]
                losses2 = self.criterion2(outputs,targets,candidate_ids,candidate_labels,mask_loss_masks)
            

            losses = self.criterion(outputs, targets_inv)
            
            fake_loss = 0*outputs['pred_logits'].sum()+0*outputs['pred_masks'].sum()

            # bipartite matching-based loss
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict or self.criterion2.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    # if k in set(losses2.keys()):
                    #     losses[k] += self.criterion2.weight_dict[k]*losses2[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            for k in list(losses2.keys()):
                if k in self.criterion2.weight_dict:
                    losses[k] = self.criterion2.weight_dict[k]*losses2[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)       
            if outputs["example_query_loss"]:
                losses["loss_example"] = outputs["example_query_loss"]
            if outputs["example_query_loss_inbatch"]:
                losses["loss_exampleb"] = outputs["example_query_loss_inbatch"]
            if self.example_sup :
                #losses["loss_example_sup"] = 0
                for k in list(losse3.keys()):
                    if k in self.criterion3.weight_dict:
                        losses[k+'3_'] = self.criterion3.weight_dict[k]*losse3[k]
                losses[k+'3_'] += 0*fakeoutput['pred_logits'].sum()
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
        
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            N,Q,H,W = mask_pred_results.shape
            mask_result = mask_pred_results.permute(0,2,3,1)
            mask_result = mask_result.view(N,H*W,Q)

            
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    #torchshow.save(images[-1],str("lvis.jpg"))
                    processed_results[-1]["instances"] = instance_r

            return processed_results
   

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        if self.query_repeat == 1:
            for targets_per_image in targets:
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes[:150],
                        "masks": padded_masks[:150],  # limit too much gt[:100]
                    }
                )
        else: 
            for targets_per_image in targets:
                # need to filter per class label more than self.query_repeat
                label = targets_per_image.gt_classes
                label_uni , label_fre = torch.unique(label,return_counts=True)
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                # 将每个类别的超过self.query_repeat数目的gt去掉，仅仅保留self.query_repeat个
                indexs = []
                for i in range(len(label_uni)):
                    if label_fre[i] > self.query_repeat:
                        index = torch.where(label == label_uni[i])[0]
                        # random choose self.query_repeat gt in index
                        index = index[torch.randperm(index.shape[0])[:self.query_repeat]]
                    else:
                        index = torch.where(label == label_uni[i])[0]
                    # append index
                    indexs =  indexs + index.tolist()
                indexs = torch.tensor(indexs)
                if len(indexs):
                    label = label[indexs]
                    padded_masks = padded_masks[indexs]
                
                
                new_targets.append(
                    {
                        "labels": label,
                        "masks": padded_masks,  # limit too much gt[:100]
                    }
                )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):

        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # here we use only top20 and sorted query to check
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(20, sorted=True)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        #topk_indices = torch.arange(100,device = self.device)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if self.binary_thres > 0:
            keep = scores_per_image > self.binary_thres
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            # here if  we use binary training , all class
            #  is to be 0 ,so keep will be all true
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()


        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))   # do not predict box to sort area
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        if self.binary_thres < 0: # here we let all class to be zero
            labels_per_image.zero_()
        labels_per_image.zero_()
        result.pred_classes = labels_per_image
        return result

    
    