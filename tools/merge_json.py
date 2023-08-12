#!/usr/bin/env python3


import copy
import json
import os
from collections import defaultdict
import sys

import lvis
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    print('add path')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvis_ow.lvis_categories_tr import LVIS_CATEGORIES
#from mask2former.data.lvis_info.lvis_categories_tr import LVIS_CATEGORIES
from cv2 import merge

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
synset2lvis_cat_id = {x["synset"]: x["id"] for x in LVIS_CATEGORIES}
lviscatid2synset = {x["id"]: x["synset"] for x in LVIS_CATEGORIES}
cococatid2synset = {x['coco_cat_id'] : x['synset'] for x in COCO_SYNSET_CATEGORIES}
synset2cococatid = {x['synset'] : x['coco_cat_id'] for x in COCO_SYNSET_CATEGORIES}

def merge_json(input_filename1,input_filename2, output_filename):
    """
    merge lvis annotatons into original coco
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(coco_json) #{ x['id'] for x in coco_json["categories"]}
    lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    coco_json['annotations'] 
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    
    coco_ann_list = list(coco_ann_set)
    coco_ann_list.sort()
    max_coco_ann_id = max(coco_ann_list)
    for ann in lvis_json["annotations"] :
        ann['id'] += max_coco_ann_id
    merged_coco['annotations'] += lvis_json["annotations"] 
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    lvis_cat = lvis_json['categories']
    coco_cat = coco_json['categories']
    lvis_id_set = {class_info['id'] for class_info in lvis_cat}
    coco_id_set = {class_info['id'] for class_info in coco_cat}
    for class_info in lvis_cat:
        if class_info['id'] in coco_id_set:
            continue
        else:
            coco_cat.append(class_info)
    coco_cat.sort(key = lambda x :x ['id'])
    merged_coco['categories'] = coco_cat
    # lvis_ann_list = list(lvis_ann_set)
    # lvis_ann_list.sort()
    
    


    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))

def merge_cocojson2lvis(input_filename1,input_filename2, output_filename):
    """
    merge cocojson into original lvis, remove the not exist id in lvis
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    #lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(lvis_json) #{ x['id'] for x in coco_json["categories"]}
    #lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    
    lvis_ann_list = list(lvis_ann_set)
    lvis_ann_list.sort()
    max_lvis_ann_id = max(lvis_ann_list)
    new_coco_ann = []
    for ann in merged_coco['annotations']:
        ann['iscrowd'] = 0
    for ann in coco_json["annotations"] :
        ann['id'] += max_lvis_ann_id
        if ann['image_id'] not in lvis_img_set:
            continue
        #ann.pop('iscrowd')
        if cococatid2synset[ann['category_id']] not in set(synset2lvis_cat_id.keys()):
            #print(ann['category_id'])
            continue
        ann['category_id'] = synset2lvis_cat_id[cococatid2synset[ann['category_id']]]
        new_coco_ann.append(ann)
    merged_coco['annotations'] += new_coco_ann
    # lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    # lvis_cat = lvis_json['categories']
    # coco_cat = coco_json['categories']
    # lvis_id_set = {class_info['id'] for class_info in lvis_cat}
    # coco_id_set = {class_info['id'] for class_info in coco_cat}
    # for class_info in lvis_cat:
    #     if class_info['id'] in coco_id_set:
    #         continue
    #     else:
    #         coco_cat.append(class_info)
    # coco_cat.sort(key = lambda x :x ['id'])
    # merged_coco['categories'] = coco_cat
    # lvis_ann_list = list(lvis_ann_set)
    # lvis_ann_list.sort()
    
    


    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))



def merge_cocojson2lvis_remove_ann(input_filename1,input_filename2, output_filename):
    """
    merge cocojson into original lvis, remove the not exist id in lvis, remove coco class anno in lvis
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    #lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(lvis_json) #{ x['id'] for x in coco_json["categories"]}
    #lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    
    lvis_ann_list = list(lvis_ann_set)
    lvis_ann_list.sort()
    max_lvis_ann_id = max(lvis_ann_list)
    new_coco_ann = []
    lvis_annos = merged_coco.pop("annotations")
    merged_coco["annotations"] = []
    for ann in lvis_annos:
        ann['iscrowd'] = 0
        synset = lviscatid2synset[ann['category_id']]
        if synset in set(synset2cococatid.keys()): # REMOVE COCO CLASS ANNO
            continue
        else:
            merged_coco["annotations"].append(ann)
    



        
    for ann in coco_json["annotations"] :
        ann['id'] += max_lvis_ann_id
        if ann['image_id'] not in lvis_img_set:
            continue
        #ann.pop('iscrowd')
        if cococatid2synset[ann['category_id']] not in set(synset2lvis_cat_id.keys()):
            #print(ann['category_id'])
            continue
        ann['category_id'] = synset2lvis_cat_id[cococatid2synset[ann['category_id']]]
        new_coco_ann.append(ann)
    merged_coco['annotations'] += new_coco_ann
    # lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    # lvis_cat = lvis_json['categories']
    # coco_cat = coco_json['categories']
    # lvis_id_set = {class_info['id'] for class_info in lvis_cat}
    # coco_id_set = {class_info['id'] for class_info in coco_cat}
    # for class_info in lvis_cat:
    #     if class_info['id'] in coco_id_set:
    #         continue
    #     else:
    #         coco_cat.append(class_info)
    # coco_cat.sort(key = lambda x :x ['id'])
    # merged_coco['categories'] = coco_cat
    # lvis_ann_list = list(lvis_ann_set)
    # lvis_ann_list.sort()
    
    


    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))

def lvisfied_coco_json2(input_filename1,input_filename2,input_filename3, output_filename):
    """
    convert coco_json 2 lvis_json, not drop img ,which means will use lvis_val
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    with open(input_filename3, "r") as f:
        lvis_val = json.load(f)
    #lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(lvis_json) #{ x['id'] for x in coco_json["categories"]}
    #lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    lvis_img_set = lvis_img_set | set([x['id'] for x in lvis_val['images']])
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    
    lvis_ann_list = list(lvis_ann_set)
    lvis_ann_list.sort()
    max_lvis_ann_id = max(lvis_ann_list)
    new_coco_ann = []
    lvis_annos = merged_coco.pop("annotations")
    merged_coco["annotations"] = []
    # for ann in lvis_annos:
    #     ann['iscrowd'] = 0
    #     synset = lviscatid2synset[ann['category_id']]
    #     if synset in set(synset2cococatid.keys()): # REMOVE COCO CLASS ANNO
    #         continue
    #     else:
    #         merged_coco["annotations"].append(ann)
    merged_coco['images'] = []
    for img in lvis_json['images']:
        if img['id'] in coco_img_set:
            merged_coco['images'].append(img)
    for img in lvis_val['images']:
        if img['id'] in coco_img_set:
            merged_coco['images'].append(img)



        
    for ann in coco_json["annotations"] :
        ann['id'] += max_lvis_ann_id
        if ann['image_id'] not in lvis_img_set:
            continue
        #ann.pop('iscrowd')
        if cococatid2synset[ann['category_id']] not in set(synset2lvis_cat_id.keys()):
            #print(ann['category_id'])
            continue
        ann['category_id'] = synset2lvis_cat_id[cococatid2synset[ann['category_id']]]
        new_coco_ann.append(ann)
    merged_coco['annotations'] = new_coco_ann
    # lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    # lvis_cat = lvis_json['categories']
    # coco_cat = coco_json['categories']
    # lvis_id_set = {class_info['id'] for class_info in lvis_cat}
    # coco_id_set = {class_info['id'] for class_info in coco_cat}
    # for class_info in lvis_cat:
    #     if class_info['id'] in coco_id_set:
    #         continue
    #     else:
    #         coco_cat.append(class_info)
    # coco_cat.sort(key = lambda x :x ['id'])
    # merged_coco['categories'] = coco_cat
    # lvis_ann_list = list(lvis_ann_set)
    # lvis_ann_list.sort()
    
    


    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))

def lvisfied_coco_json(input_filename1,input_filename2, output_filename):
    """
    convert coco_json 2 lvis_json, this would drop some img and ann
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    #lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(lvis_json) #{ x['id'] for x in coco_json["categories"]}
    #lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    
    lvis_ann_list = list(lvis_ann_set)
    lvis_ann_list.sort()
    max_lvis_ann_id = max(lvis_ann_list)
    new_coco_ann = []
    lvis_annos = merged_coco.pop("annotations")
    merged_coco["annotations"] = []
    # for ann in lvis_annos:
    #     ann['iscrowd'] = 0
    #     synset = lviscatid2synset[ann['category_id']]
    #     if synset in set(synset2cococatid.keys()): # REMOVE COCO CLASS ANNO
    #         continue
    #     else:
    #         merged_coco["annotations"].append(ann)
    
    for ann in coco_json["annotations"] :
        ann['id'] += max_lvis_ann_id
        if ann['image_id'] not in lvis_img_set:
            continue
        #ann.pop('iscrowd')
        if cococatid2synset[ann['category_id']] not in set(synset2lvis_cat_id.keys()):
            #print(ann['category_id'])
            continue
        ann['category_id'] = synset2lvis_cat_id[cococatid2synset[ann['category_id']]]
        new_coco_ann.append(ann)
    merged_coco['annotations'] = new_coco_ann

    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))

def merge_cocojson2lvis_remove_ann2(input_filename1,input_filename2,input_filename3, output_filename):
    """
    merge cocojson into original lvis, remove the not exist id in lvis, remove coco class anno in lvis
    Args:
        input_filename1 (str): path to the LVIS json file.
        input_filename2 (str): path to the COCO json file.
        output_filename (str): path to the merged json file.
    """

    with open(input_filename1, "r") as f:
        lvis_json = json.load(f)
    with open(input_filename2, "r") as f:
        coco_json = json.load(f)
    with open(input_filename3, "r") as f:
        lvis_val = json.load(f)
    #lvis_annos = lvis_json.pop("annotations")
    merged_coco = copy.deepcopy(lvis_json) #{ x['id'] for x in coco_json["categories"]}
    #lvis_json["annotations"] = lvis_annos
    #len(coco_json['images']) # 118287
    coco_img_set = set([x['id'] for x in coco_json['images']])
    lvis_img_set = set([x['id'] for x in lvis_json['images']])
    lvis_val_img_set = set([x['id'] for x in lvis_val['images']])
    lvis_img_set = lvis_img_set | lvis_val_img_set
    coco_ann_set = set([x['id'] for x in coco_json['annotations']])
    lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    
    lvis_ann_list = list(lvis_ann_set)
    lvis_ann_list.sort()
    max_lvis_ann_id = max(lvis_ann_list)
    new_coco_ann = []
    lvis_annos = merged_coco.pop("annotations")
    merged_coco["annotations"] = []
    merged_coco ["images"] = []    
    for img in lvis_json["images"]:
        if img['id'] in coco_img_set:
            merged_coco["images"].append(img)
    for img in coco_json["images"]:
        if img['id'] in coco_img_set:
            merged_coco["images"].append(img)
    for ann in lvis_annos:
        ann['iscrowd'] = 0
        # check if the image is in coco
        if ann['image_id'] not in coco_img_set:
            continue
        synset = lviscatid2synset[ann['category_id']]
        if synset in set(synset2cococatid.keys()): # REMOVE COCO CLASS ANNO
            continue
        else:
            merged_coco["annotations"].append(ann)
    for ann in lvis_val["annotations"]:
        ann['iscrowd'] = 0
        # check if the image is in coco
        if ann['image_id'] not in coco_img_set:
            continue
        synset = lviscatid2synset[ann['category_id']]
        if synset in set(synset2cococatid.keys()): 
            continue
        else:
            merged_coco["annotations"].append(ann)
    



        
    for ann in coco_json["annotations"] :
        ann['id'] += max_lvis_ann_id
        if ann['image_id'] not in lvis_img_set:
            continue
        #ann.pop('iscrowd')
        if cococatid2synset[ann['category_id']] not in set(synset2lvis_cat_id.keys()):
            #print(ann['category_id'])
            continue
        ann['category_id'] = synset2lvis_cat_id[cococatid2synset[ann['category_id']]]
        new_coco_ann.append(ann)
    merged_coco['annotations'] += new_coco_ann
    # lvis_ann_set = set([x['id'] for x in lvis_json['annotations']])
    # lvis_cat = lvis_json['categories']
    # coco_cat = coco_json['categories']
    # lvis_id_set = {class_info['id'] for class_info in lvis_cat}
    # coco_id_set = {class_info['id'] for class_info in coco_cat}
    # for class_info in lvis_cat:
    #     if class_info['id'] in coco_id_set:
    #         continue
    #     else:
    #         coco_cat.append(class_info)
    # coco_cat.sort(key = lambda x :x ['id'])
    # merged_coco['categories'] = coco_cat
    # lvis_ann_list = list(lvis_ann_set)
    # lvis_ann_list.sort()
    
    


    with open(output_filename, "w") as f:
        json.dump(merged_coco, f)
    print("{} and {} is merged in {}.".format(input_filename1,input_filename2, output_filename))
if __name__ == "__main__":
    lvis_dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "lvis")
    coco_dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco/annotations")
    lvis_name = "lvis_v1_train"
    coco_name = "instances_train2017"
    lvis_val = "lvis_v1_val"
    merge_cocojson2lvis(
            os.path.join(lvis_dataset_dir, "{}.json".format(lvis_name)),
            os.path.join(coco_dataset_dir, "{}.json".format(coco_name)),
            os.path.join(lvis_dataset_dir, "{}_with_coco80.json".format(lvis_name)),
        )
