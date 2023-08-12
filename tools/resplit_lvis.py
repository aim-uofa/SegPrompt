#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import json
import os
from collections import defaultdict
import sys
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    print('add path')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvis_ow import lvis_categories_010,lvis_categories_r,lvis_categories_020,lvis_categories_005
from lvis_ow.lvis_categories_tr import LVIS_CATEGORIES

# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]
from pycocotools.coco import COCO

def resplit_lvis(input_filename1, input_filename2, output_filename1, output_filename2,categories):
    """
    Resplit the LVIS train /val dataset , move some images from train to val
    Args:
        input_filename (str): path to the LVIS json file.
        output_filename (str): path to the COCOfied json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json_tr = json.load(f)
    LVIS_CAT_ = categories
    lvis_annos_tr = lvis_json_tr.pop("annotations")
    lvis_imgs_tr = lvis_json_tr.pop("images")
    with open(input_filename2, "r") as f:
        lvis_json_val = json.load(f)
    lvis_annos_val = lvis_json_val.pop("annotations")
    lvis_imgs_val = lvis_json_val.pop("images")

    lvis_tr = COCO(input_filename1)
    catid2move = [cat["id"] for cat in LVIS_CAT_ ]
    imgid2move = set()
    for catid in catid2move:
        imgid2move.update(lvis_tr.getImgIds(catIds=[catid]))
    #imgid2move = set(imgid2move)
    newlvis_imgs_tr = []
    for img in lvis_imgs_tr:
        if img["id"] in imgid2move:
            lvis_imgs_val.append(img)
        else:
            newlvis_imgs_tr.append(img)
    newlvis_annos_tr = []
    for anno in lvis_annos_tr:
        if anno["image_id"] in imgid2move:
            # check ann id is unique
            anno['id'] = anno['id'] + 2000000
            lvis_annos_val.append(anno)
        else: 
            newlvis_annos_tr.append(anno) 
    #  # need to check ann id is unique
    ann_id_tr = set ( [anno["id"] for anno in newlvis_annos_tr] )
    ann_id_val = set ( [anno["id"] for anno in lvis_annos_val] )
    #assert len(ann_id_tr.intersection(ann_id_val)) == 0
    lvis_json_tr["images"] = newlvis_imgs_tr
    lvis_json_tr["annotations"] = newlvis_annos_tr
   
        
    lvis_json_val["images"] = lvis_imgs_val
    lvis_json_val["annotations"] = lvis_annos_val
    with open(output_filename1, "w") as f:
        json.dump(lvis_json_tr, f)
    with open(output_filename2, "w") as f:
        json.dump(lvis_json_val, f)



if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "lvis")
    s1 = 'lvis_v1_train'
    s2 = 'lvis_v1_val'
    # resplit_lvis(
    #         os.path.join(dataset_dir, "{}.json".format(s1)),
    #         os.path.join(dataset_dir, "{}.json".format(s2)),
    #         os.path.join(dataset_dir, "{}_resplit010.json".format(s1)),
    #         os.path.join(dataset_dir, "{}_resplit010.json".format(s2)),
    #         lvis_categories_010.LVIS_CATEGORIES_010,
    #     )
    # resplit_lvis(
    #         os.path.join(dataset_dir, "{}.json".format(s1)),
    #         os.path.join(dataset_dir, "{}.json".format(s2)),
    #         os.path.join(dataset_dir, "{}_resplit020.json".format(s1)),
    #         os.path.join(dataset_dir, "{}_resplit020.json".format(s2)),
    #         lvis_categories_020.LVIS_CATEGORIES_020,
    #     )
    # resplit_lvis(
    #         os.path.join(dataset_dir, "{}.json".format(s1)),
    #         os.path.join(dataset_dir, "{}.json".format(s2)),
    #         os.path.join(dataset_dir, "{}_resplit005.json".format(s1)),
    #         os.path.join(dataset_dir, "{}_resplit005.json".format(s2)),
    #         lvis_categories_005.LVIS_CATEGORIES_005,
    #     )
    resplit_lvis(
            os.path.join(dataset_dir, "{}.json".format(s1)),
            os.path.join(dataset_dir, "{}.json".format(s2)),
            os.path.join(dataset_dir, "{}_resplit_r.json".format(s1)),
            os.path.join(dataset_dir, "{}_resplit_r.json".format(s2)),
            lvis_categories_r.LVIS_CATEGORIES_R,
        )