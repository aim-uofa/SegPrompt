# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import json
from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torch.nn.functional as F
from pycocotools import mask as maskUtils
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithAnno"]


class DatasetMapperWithAnno:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        logit_path,
        query_path,   
        json_path,
        max_classes,
        neg_sample,
        dataset_train,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        if 'clip' in logit_path:
            self.logit_path = logit_path + '/'
        else:
            self.logit_path = None
        self.query_path = query_path + '/'
        self.json_path = json_path
        # according to json_path,get cate_to_annid, annid_to_imgid, imgid_to_annid,
        # _cate_to_ann_id.json,_ann_id_to_img_id.json, _img_id_to_ann_id.json, 
        #debug for not found error, 查看当前路径
        import os 
        if len(self.json_path):
            self.cate_to_annid = json.load(open(self.json_path[:-5] + '_cate_to_ann_id.json'))
            self.annid_to_imgid = json.load(open(self.json_path[:-5] + '_ann_id_to_img_id.json'))
            self.imgid_to_annid = json.load(open(self.json_path[:-5] + '_img_id_to_ann_id.json'))
        else:
            self.cate_to_annid = None
            self.annid_to_imgid = None
            self.imgid_to_annid = None
        # according to cate_to_annid , get annid_to_cate
        self.annid_to_cate = {}
        self.max_classes = max_classes
        self.dataset = dataset_train
        self.neg_sample = neg_sample
        # if self.is_train==False:
        #     assert self.neg_sample == False
       
        if self.dataset[0] == 'lvis_v1_val':
            self.id_set = set(range(1203)) # 0 - 1202
        else:
            raise NotImplementedError
        for cate, annids in self.cate_to_annid.items():
            for annid in annids:
                self.annid_to_cate[annid] = cate
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "logit_path" : cfg.DATASETS.CLIP_LOGIT_PATH,
            "query_path": cfg.DATASETS.QUERY_PATH,
            "json_path" : cfg.DATASETS.JSON_PATH,
            "max_classes": cfg.DATASETS.MAX_CLASSES,
            "neg_sample": cfg.DATASETS.NEG_SAMPLE, # for eval, no need to neg sample
            "dataset_train": cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST,

        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None) # mask need to change to correct format
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # annos = [
        #     utils.transform_instance_annotations(
        #         obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
        #     )
        #     for obj in dataset_dict.pop("annotations")
        #     if obj.get("iscrowd", 0) == 0
        # ]
        annos = [ obj 
         for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        # keep ann_ids 
        if len(annos) == 0:
            dataset_dict['ann_ids'] = []
            ann_ids = []
        else:
            if 'annid' in annos[0]:
                ann_ids = [obj['annid'] for obj in annos]
                dataset_dict['ann_ids'] = ann_ids
        #assert len(annos) == len(dataset_dict['ann_ids'])
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        #assert len(annos) == len(dataset_dict['ann_ids'])

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # dataset_dict["instances"] = utils.filter_empty_instances(instances)
        # NOTE didn't filt out empty
        dataset_dict["instances"] = instances
        # if len(dataset_dict["instances"]) != len(ann_ids):
        #     print('warning: empty instance')


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        ori_image_shape = image.shape[:2]  # h, w
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        img_id = dataset_dict["image_id"]
        # need to padding zero to img_id 145337  -> 0000000145337 total 12 digits
        img_id = str(img_id).zfill(12)
        def padding_zero(img_id, total_digits=12):
            return str(img_id).zfill(total_digits)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        img_id  = dataset_dict['image_id'] 
        image_shape = image.shape[:2]  # h, w
        if self.logit_path is not None:
            logit = torch.load(self.logit_path + str(img_id) + '.pth',map_location= 'cpu') # C x H x W 
            # to numpy float32
            logit = logit.numpy().astype(np.float32)

            #logit = F.interpolate(logit, size=image_shape, mode='bilinear', align_corners=False)
            dataset_dict['logit'] = torch.as_tensor(np.ascontiguousarray(logit))
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if False:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, ori_image_shape)
            gt_classes = dataset_dict["instances"].gt_classes
            max_total_num = 100
            total_num =  0
             # 尽可能均匀的分配每个类别的ann_num
            if len(gt_classes) == 0:
                query_dim = 256
                query = torch.zeros((max_total_num,query_dim ))
                query_label = torch.zeros((max_total_num, 1))
            else:
                # unique
                gt_classes = torch.unique(gt_classes)
                ori_gt_classes = gt_classes
                if self.max_classes > 0: # onlt use max_classes
                    index = torch.randperm(len(gt_classes))[:self.max_classes]
                    gt_classes = gt_classes[index]
                    if self.neg_sample: # only when max_classes > 0, use neg_sample
                        # sample neg_classes
                        neg_sampled = True
                        if len(gt_classes) < self.max_classes: # only sample neg_classes when gt_classes < max_classes
                            # sample neg_classes from self.id_set - gt_classes
                            neg_classes = torch.tensor(list(self.id_set - set(gt_classes.tolist())))
                            neg_index = torch.randperm(len(neg_classes))[:self.max_classes - len(gt_classes)]
                            neg_classes = neg_classes[neg_index]
                            gt_classes = torch.cat([gt_classes, neg_classes])
                        else:
                            neg_sampled = False
                if not  (gt_classes.max() < 1203 and gt_classes.min() >= 0):
                    print('gt_classes', gt_classes)
                # for each gt_classes, get the corresponding ann_num
                ann_num = np.zeros(len(gt_classes))
                # for gt_class in gt_classes:
                #     # ann_num.append(len(self.cate_to_annid[str(gt_class.item())]))
                #     ann_num[gt_class.item()] = len(self.cate_to_annid[str(gt_class.item())])
                for i, gt_class in enumerate(gt_classes):
                    ann_num[i] = len(self.cate_to_annid[str(gt_class.item()+1)])

                
                alloc_num = np.zeros_like (ann_num)
                avg_num = max_total_num // len(gt_classes)  # 向下取整,
                for i in range(len(gt_classes)):  # 为每个类别分配ann_num, 如果ann_num[i] <= avg_num, 则分配ann_num[i], 否则分配avg_num
                    if ann_num[i] <= avg_num:
                        alloc_num[i] = ann_num[i]
                    else:
                        alloc_num[i] = avg_num
                    # 记录下总共分配了多少个ann_num， 以及哪些i的ann已经被分配完了
                    total_num += alloc_num[i]
                # 如果总共分配的ann_num不够max_total_num, 则从剩下的类别中均匀分配
                while total_num < max_total_num:
                    flag = False
                    for i in range(len(gt_classes)):
                        if ann_num[i] > alloc_num[i]:
                            alloc_num[i] += 1
                            total_num += 1
                            flag = True
                            if total_num == max_total_num:
                                break
                    if not flag:
                        break # 所有的类别的ann都已经分配完了，但是还是没有分配够max_total_num个ann
                # 根据 alloc_num, 为每个gt_classes随机选择对应的alloc_num个ann
                choosen_ann_id = []
                for i, gt_class in enumerate(gt_classes):
                    ann_id = self.cate_to_annid[str(gt_class.item()+1)]
                    ann_id = np.random.choice(ann_id, int(alloc_num[i]), replace=False) # 不重复的随机选择
                    # to list
                    ann_id = ann_id.tolist()
                    choosen_ann_id += ann_id
                # if  len(choosen_ann_id) != max_total_num:
                #     print('error, len(choosen_ann_id) != max_total_num {} != {}'.format(len(choosen_ann_id), max_total_num))
                # 根据 choosen_ann_id, 从.pt 文件中读取对应的query
                query_dim = 256
                query = torch.zeros((max_total_num,query_dim ))
                query_label = torch.zeros((max_total_num, 1))
                for i, ann_id in enumerate(choosen_ann_id):
                    img_id, ann_index = self.get_ann_index(ann_id)
                    img_id = padding_zero(img_id, 12)
                    temp_pt = torch.load(self.query_path + img_id + '.pt',map_location=torch.device('cpu'))
                    assert temp_pt.shape[2] == query_dim
                    assert temp_pt.shape[0] == len(self.imgid_to_annid[str(int(img_id))]), 'missing anno'
                    query[i] = temp_pt[ann_index]
                    query_label[i] = int(self.annid_to_cate[int(ann_id)]) - 1  # 0-1202
            dataset_dict['query'] = query
            dataset_dict['query_label'] = query_label




        return dataset_dict
    def get_ann_index(self, ann_id):
        # given ann_id, return its imgid, and index of the ann in its corresponding .pt file
        img_id = self.annid_to_imgid[str(ann_id)]
        anns_for_img = self.imgid_to_annid[str(img_id)] # anns_for_img is a list
        return img_id, anns_for_img.index(ann_id)
