from operator import imod
from pycocotools.coco import COCO
import copy
import json
import os
from collections import defaultdict
#'datasets/coco/annotations/instances_train2017_with_64classlvis_keepid.json'
import sys
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    print('add path')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvis_ow.lvis_categories_tr import NAME2ID,FRE_NAME_LIST

def filter_with_cate(input_filename, output_filename,cat_name):
    """
    Filter LVIS instance segmentation annotations to remove all categories that are not included in
    COCO. The new json files can be used to evaluate COCO AP using `lvis-api`. The category ids in
    the output json are the incontiguous COCO dataset ids. 
    this will only filter images, not annotations
    Args:
        input_filename (str): path to the LVIS json file.
        output_filename (str): path to the COCOfied json file.
        cat_name (list) :   a list of cat_name to be kept
    """
    coco_lvis = COCO(input_filename)
    coco_lvis.info()
    names_candidate = cat_name
    name2id = {}
    for key in coco_lvis.cats.keys():
        name2id [coco_lvis.cats[key]['name']] = coco_lvis.cats[key]['id']
    ids_candidate = [ name2id[name] for name in names_candidate]
    imgid_list = []
    for id in ids_candidate:
        imgid_list.extend(coco_lvis.getImgIds(catIds=[id]))
        
    #imgid_list = set(coco_lvis.getImgIds(catIds = ids_candidate))
    # simultaneously save the img address to a txt file
    with open(output_filename[:-5]+'.txt', 'w') as f:
        for imgid in imgid_list:
            # here we need to remove url prefix
            f.write('/'.join(coco_lvis.loadImgs(imgid)[0]['coco_url'].split('/')[-2:])+'\n')
            
    with open(input_filename, "r") as f:
        lvis_json = json.load(f)

    lvis_annos = lvis_json.pop("annotations")
    lvis_imgs = lvis_json.pop('images')
    cocofied_lvis = copy.deepcopy(lvis_json)
    #lvis_json["annotations"] = lvis_annos

    # # Mapping from lvis cat id to coco cat id via synset
    # lvis_cat_id_to_synset = {cat["id"]: cat["synset"] for cat in lvis_json["categories"]}
    # synset_to_coco_cat_id = {x["synset"]: x["coco_cat_id"] for x in COCO_SYNSET_CATEGORIES}
    # # Synsets that we will keep in the dataset
    # synsets_to_keep = set(synset_to_coco_cat_id.keys())
    # coco_cat_id_with_instances = defaultdict(int)

    new_annos = []
    # for id in imgid_list:
    #     anns = coco_lvis.imgToAnns[id]
    #     cat_set = {ann['category_id'] for ann in anns}
    #     if 995 not in cat_set:
    #         print(id)
    for ann in lvis_annos:
        if ann['image_id'] not in imgid_list:
            continue
        new_annos.append(ann) #注意，这里没有考虑 ann id 的连续性，而保留了原有的id
        
    cocofied_lvis["annotations"] = new_annos
    new_images = []
    for image in lvis_imgs:
        if image['id'] not in imgid_list:
            continue
        new_images.append(image)
    cocofied_lvis["images"] = new_images
    with open(output_filename, "w") as f:
        json.dump(cocofied_lvis, f)
    print("{} total {} images".format(''.join(cat_name),str(len(imgid_list))))
    print("{} is COCOfied and stored in {}.".format(input_filename, output_filename))

def filter_with_cate2(input_filename, output_filename,cat_name):
    """
    Filter LVIS instance segmentation annotations to remove all categories that are not included in
    COCO. The new json files can be used to evaluate COCO AP using `lvis-api`. The category ids in
    the output json are the incontiguous COCO dataset ids. 
    this will filter annotations
    Args:
        input_filename (str): path to the LVIS json file.
        output_filename (str): path to the COCOfied json file.
        cat_name (list) :   a list of cat_name to be kept
    """
    coco_lvis = COCO(input_filename)
    coco_lvis.info()
    names_candidate = cat_name
    name2id = {}
    for key in coco_lvis.cats.keys():
        name2id [coco_lvis.cats[key]['name']] = coco_lvis.cats[key]['id']
    ids_candidate = [ name2id[name] for name in names_candidate]
    imgid_list = set(coco_lvis.getImgIds(catIds = ids_candidate))
    imgid_list = set()
    for id in ids_candidate:
        imgid_list = imgid_list.union(set(coco_lvis.getImgIds(catIds = [id])))
    print('total {} images'.format(len(imgid_list)))
    ids_candidate = set(ids_candidate)
    # simultaneously save the img address to a txt file
    with open(output_filename[:-5]+'.txt', 'w') as f:
        for imgid in imgid_list:
            # here we need to remove url prefix
            f.write('/'.join(coco_lvis.loadImgs(imgid)[0]['coco_url'].split('/')[-2:])+'\n')
            
    with open(input_filename, "r") as f:
        lvis_json = json.load(f)

    lvis_annos = lvis_json.pop("annotations")
    lvis_imgs = lvis_json.pop('images')
    cocofied_lvis = copy.deepcopy(lvis_json)
    #lvis_json["annotations"] = lvis_annos

    # # Mapping from lvis cat id to coco cat id via synset
    # lvis_cat_id_to_synset = {cat["id"]: cat["synset"] for cat in lvis_json["categories"]}
    # synset_to_coco_cat_id = {x["synset"]: x["coco_cat_id"] for x in COCO_SYNSET_CATEGORIES}
    # # Synsets that we will keep in the dataset
    # synsets_to_keep = set(synset_to_coco_cat_id.keys())
    # coco_cat_id_with_instances = defaultdict(int)

    new_annos = []
    # for id in imgid_list:
    #     anns = coco_lvis.imgToAnns[id]
    #     cat_set = {ann['category_id'] for ann in anns}
    #     if 995 not in cat_set:
    #         print(id)
    for ann in lvis_annos:
        if ann['image_id'] not in imgid_list:
            continue
        if ann['category_id'] not in ids_candidate:
            continue
        new_annos.append(ann) 
    print('ori {} ,keep {} annotations'.format(len(lvis_annos),len(new_annos)))
        
    cocofied_lvis["annotations"] = new_annos
    new_images = []
    for image in lvis_imgs:
        if image['id'] not in imgid_list:
            continue
        new_images.append(image)
    cocofied_lvis["images"] = new_images
    print('ori {} ,keep {} images'.format(len(lvis_imgs),len(new_images)))
    with open(output_filename, "w") as f:
        json.dump(cocofied_lvis, f)
    print("{} total {} images".format(len(lvis_imgs),str(len(imgid_list))))
    print("{} is COCOfied and stored in {}.".format(input_filename, output_filename))

def filter_with_imgid(input_filename, output_filename, imgid_list):

    with open(input_filename, "r") as f:
        lvis_json = json.load(f)

    lvis_annos = lvis_json.pop("annotations")
    lvis_imgs = lvis_json.pop('images')
    cocofied_lvis = copy.deepcopy(lvis_json)
    #lvis_json["annotations"] = lvis_annos

    # # Mapping from lvis cat id to coco cat id via synset
    # lvis_cat_id_to_synset = {cat["id"]: cat["synset"] for cat in lvis_json["categories"]}
    # synset_to_coco_cat_id = {x["synset"]: x["coco_cat_id"] for x in COCO_SYNSET_CATEGORIES}
    # # Synsets that we will keep in the dataset
    # synsets_to_keep = set(synset_to_coco_cat_id.keys())
    # coco_cat_id_with_instances = defaultdict(int)

    new_annos = []
    # for id in imgid_list:
    #     anns = coco_lvis.imgToAnns[id]
    #     cat_set = {ann['category_id'] for ann in anns}
    #     if 995 not in cat_set:
    #         print(id)
    id2remove = []
    new_images = []
    for image in lvis_imgs:
        if image['ytid']  in imgid_list:
            id2remove.append(image['id'])
            continue
        new_images.append(image)
    cocofied_lvis["images"] = new_images
    for ann in lvis_annos:
        if ann['image_id']  in id2remove:
            continue
        new_annos.append(ann) 
        
    cocofied_lvis["annotations"] = new_annos
    
    
    with open(output_filename, "w") as f:
        json.dump(cocofied_lvis, f)
    # print("{} total {} images".format(''.join(cat_name),str(len(imgid_list))))
    print("{} is COCOfied and stored in {}.".format(input_filename, output_filename))
def filter_with_imgid(input_filename, output_filename, imgid_list):

    with open(input_filename, "r") as f:
        lvis_json = json.load(f)

    lvis_annos = lvis_json.pop("annotations")
    lvis_imgs = lvis_json.pop('images')
    cocofied_lvis = copy.deepcopy(lvis_json)
    #lvis_json["annotations"] = lvis_annos

    # # Mapping from lvis cat id to coco cat id via synset
    # lvis_cat_id_to_synset = {cat["id"]: cat["synset"] for cat in lvis_json["categories"]}
    # synset_to_coco_cat_id = {x["synset"]: x["coco_cat_id"] for x in COCO_SYNSET_CATEGORIES}
    # # Synsets that we will keep in the dataset
    # synsets_to_keep = set(synset_to_coco_cat_id.keys())
    # coco_cat_id_with_instances = defaultdict(int)

    new_annos = []
    # for id in imgid_list:
    #     anns = coco_lvis.imgToAnns[id]
    #     cat_set = {ann['category_id'] for ann in anns}
    #     if 995 not in cat_set:
    #         print(id)
    id2remove = []
    new_images = []
    for image in lvis_imgs:
        if image['ytid']  in imgid_list:
            id2remove.append(image['id'])
            continue
        new_images.append(image)
    cocofied_lvis["images"] = new_images
    for ann in lvis_annos:
        if ann['image_id']  in id2remove:
            continue
        new_annos.append(ann) #注意，这里没有考虑 ann id 的连续性，而保留了原有的id
        
    cocofied_lvis["annotations"] = new_annos
    
    
    with open(output_filename, "w") as f:
        json.dump(cocofied_lvis, f)
    # print("{} total {} images".format(''.join(cat_name),str(len(imgid_list))))
    print("{} is COCOfied and stored in {}.".format(input_filename, output_filename))



def filter_with_another_json(input_filename,input_filename2,output_filename):
    # filter the input_filename with the imgid in input_filename2
    with open(input_filename2, "r") as f:
        lvis_json2 = json.load(f)  
    
    img_idset = set([img['id'] for img in lvis_json2['images']])
    lvis_json2 = None
    with open(input_filename, "r") as f:
        lvis_json = json.load(f)
    lvis_annos = lvis_json.pop("annotations")
    lvis_imgs = lvis_json.pop('images')
    new_imgs = []
    new_annos = []
    for img in lvis_imgs:
        if img['id'] not in img_idset:
           continue
        new_imgs.append(img)
    assert len(new_imgs) <= len(img_idset)
    for ann in lvis_annos:
        if ann['image_id'] not in img_idset:
            continue
        new_annos.append(ann)
    lvis_json['images'] = new_imgs
    lvis_json['annotations'] = new_annos
    with open(output_filename, "w") as f:
        json.dump(lvis_json, f)
    print("{} is COCOfied and stored in {}.".format(input_filename, output_filename))

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "lvis")
    lvis_dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "lvis")
    coco_dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco/annotations")
      
    filter_with_another_json(
        os.path.join(lvis_dataset_dir, "lvis_v1_train_with_coco80_123class.json"),
        os.path.join(lvis_dataset_dir, "lvis_v1_train_resplit_r.json"),
        os.path.join(lvis_dataset_dir, "lvis_v1_train_ow.json")
    )
  

