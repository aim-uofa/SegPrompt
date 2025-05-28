# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import json
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

__all__ = ["COCOInstanceFewshotBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceFewshotBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        mask_format, 
        query_path,   
        json_path,
        max_classes,
        neg_sample,
        dataset_train,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.mask_format = mask_format
        self.query_path = query_path + '/'
        self.json_path = json_path
        # according to json_path,get cate_to_annid, annid_to_imgid, imgid_to_annid,
        # _cate_to_ann_id.json,_ann_id_to_img_id.json, _img_id_to_ann_id.json, 
        #debug for not found error, 查看当前路径
        import os 
        self.cate_to_annid = json.load(open(self.json_path[:-5] + '_cate_to_ann_id.json'))
        self.annid_to_imgid = json.load(open(self.json_path[:-5] + '_ann_id_to_img_id.json'))
        self.imgid_to_annid = json.load(open(self.json_path[:-5] + '_img_id_to_ann_id.json'))
        # according to cate_to_annid , get annid_to_cate
        self.annid_to_cate = {}
        self.max_classes = max_classes
        self.dataset = dataset_train
        self.neg_sample = neg_sample
        if self.dataset[0] == 'lvis_v1_train':
            self.id_set = set(range(1203)) # 0 - 1202
        else:
            raise NotImplementedError
        for cate, annids in self.cate_to_annid.items():
            for annid in annids:
                self.annid_to_cate[annid] = cate
        

    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "mask_format": cfg.INPUT.MASK_FORMAT,
            "query_path": cfg.DATASETS.QUERY_PATH,
            "json_path" : cfg.DATASETS.JSON_PATH,
            "max_classes": cfg.DATASETS.MAX_CLASSES,
            "neg_sample": cfg.DATASETS.NEG_SAMPLE,
            "dataset_train": cfg.DATASETS.TRAIN,

        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        img_id = dataset_dict["image_id"]
        # need to padding zero to img_id 145337  -> 0000000145337 total 12 digits
        img_id = str(img_id).zfill(12)
        def padding_zero(img_id, total_digits=12):
            return str(img_id).zfill(total_digits)
        # query = torch.load(self.query_path + img_id  + '.pt')
        # given dataset_name, return json file path
        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                # RLE

                try:
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                except:
                    gt_masks = gt_masks.tensor
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances
            gt_classes = instances.gt_classes
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
            # assert all query_label are in ori_gt_classes or neg_sampled
            
            # if not neg_sampled:
            #     assert set(query_label[:,0].int().tolist()) <= set(ori_gt_classes.tolist())
            

        return dataset_dict
    def get_ann_index(self, ann_id):
        # given ann_id, return its imgid, and index of the ann in its corresponding .pt file
        img_id = self.annid_to_imgid[str(ann_id)]
        anns_for_img = self.imgid_to_annid[str(img_id)] # anns_for_img is a list
        return img_id, anns_for_img.index(ann_id)
