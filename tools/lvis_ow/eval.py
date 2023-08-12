import datetime
import logging
from collections import OrderedDict
from collections import defaultdict
# from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool as ThreadPool
from multiprocessing import Manager, managers
# mask sure mask2former is in sys
import sys,os
from tkinter.messagebox import NO
sys.path.append('../')
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    #print('add path')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(sys.path)
from .lvis_categories_tr import NAME2ID, ID2NAME
from .lvis_categories_tr_part50 import IN_VOC_ID_SET, OUT_VOC_ID_SET
from joblib import Parallel, delayed
#from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool as ThreadPool
import multiprocessing as mp

import numpy as np

from lvis_ow.lvis import LVIS
from lvis_ow.results import LVISResults
from tqdm import tqdm
import pycocotools.mask as mask_utils
import time
# ious_thrs [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
def _get_gt_dt(img_id, cat_id,_gts, _dts):
    """Create gt, dt which are list of anns/dets. If use_cats is true
    only anns/dets corresponding to tuple (img_id, cat_id) will be
    used. Else, all anns/dets in image are used and cat_id is not used.
    """
    global lvis_self
    assert lvis_self.params.use_cats

    gt = _gts[img_id, cat_id]
    dt = _dts[img_id, cat_id]

    return gt, dt

def compute_iou2(img_id, cat_id,_gts, _dts,dt_cat_id=1):
    global lvis_self
    self = lvis_self
    #print(cat_id,len(_gts),len(_dts))
    gt, dt2 = _get_gt_dt(img_id, cat_id,_gts, _dts) # 这里我们只需要gt
    _, dt = _get_gt_dt(img_id, dt_cat_id,_gts, _dts)#
    # merge dt and dt2
    dt = dt + dt2
    if len(gt) == 0 and len(dt) == 0:
        return []

    # Sort detections in decreasing order of score.
    idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in idx]

    iscrowd = [int(False)] * len(gt)

    if self.params.iou_type == "segm":
        ann_type = "segmentation"
    elif self.params.iou_type == "bbox":
        ann_type = "bbox"
    else:
        raise ValueError("Unknown iou_type for iou computation.")
    gt = [g[ann_type] for g in gt]
    dt = [d[ann_type] for d in dt]

    # compute iou between each dt and gt region
    # will return array of shape len(dt), len(gt)
    ious = mask_utils.iou(dt, gt, iscrowd)
    return ious
def evaluate_img2(img_id, cat_id, area_rng,ious = None, _gts=None, _dts=None,per_img_info=None):
        """Perform evaluation for single category and image.
        这里需要考虑所有dt, AND ALL DT CLASS ID IS 1
        
        """
        #print("img_id: ", img_id, "cat_id: ", cat_id)
        # print(cat_id,len(_gts),len(_dts))
        gt, dt2 = _get_gt_dt(img_id, cat_id,_gts, _dts) # 这里我们只需要gt
        _, dt = _get_gt_dt(img_id, 1,_gts, _dts)#
        # dt = dt + dt2
        if len(gt) == 0 and len(dt) == 0:
            return [None,None]

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        if ious is None:
            ious = (
                self.ious[img_id, cat_id][:, gt_idx]
                if len(self.ious[img_id, cat_id]) > 0
                else self.ious[img_id, cat_id]
            )
        else:
            ious = (
                ious[img_id, cat_id][:, gt_idx]
                if len(ious[img_id, cat_id]) > 0
                else ious[img_id, cat_id]
            )

        num_thrs = len(iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            # or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]

        if area_rng == [0, 10000000000.0]: # here we count all size
            gt_m_50 = gt_m[0]
            matched_num = (gt_m_50>0).sum()
            missed_num = gt_m.shape[-1] - matched_num
            per_img_info = {#"category_id": cat_id,
            "area_rng": area_rng,
            "gt_num": num_gt,
            "dt_num": num_dt,
            "missed_num": missed_num}
        
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        result = [{
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }   ]
        result.append(per_img_info)
        return  result
class single_pro():
    def __init__(self,id2idx,father_self,num_area_rngs, num_imgs, num_recalls,_gts,_dts,per_img_info):
        self.id2idx = id2idx
        # self.params = father_self.params
        # self.compute_iou2 = compute_iou2
        # self.evaluate_img2 = father_self.evaluate_img2
        # self.evaluate_img2 = evaluate_img2
        self.num_area_rngs = num_area_rngs
        self.num_imgs = num_imgs
        self.num_recalls = num_recalls
        self._gts, self._dts = _gts, _dts
        self.per_img_info = per_img_info
    def __call__(self,cat_id):
        global lvis_self
        start_t = time.time()
        eval_imgs = []
        ious = {}
        cat_idx = self.id2idx[cat_id]
        return_dict = {}
        return_dict['cat_id'] = cat_id
        return_dict['recall'] = {}
        return_dict['precision'] = {}
        return_dict['per_img_info'] = {}
        
        
        i = 0
        for img_id in (lvis_self.params.img_ids):
            i += 1
            # ious [(img_id, cat_id)]= lvis_self.compute_iou2(img_id, cat_id)
            ious [(img_id, cat_id)]= compute_iou2(img_id, cat_id, self._gts, self._dts)
            
        for area_rng in lvis_self.params.area_rng:
            for img_id in lvis_self.params.img_ids:
                result = evaluate_img2(img_id, cat_id, area_rng, ious,self._gts, self._dts,self.per_img_info)
                eval_imgs.append( result[0])
                if area_rng == [0, 10000000000.0] and result[1] is not None:
                    return_dict["per_img_info"][img_id,cat_id] = result[1]


        print("cat_id: {} eval time: {}".format(cat_id, time.time() - start_t))
        for area_idx in range(self.num_area_rngs):
            Na = area_idx * self.num_imgs
            E = [
                eval_imgs[0 + Na + img_idx]
                for img_idx in range(self.num_imgs)
            ]
            # Remove elements which are None
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue

            # Append all scores: shape (N,)
            dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
            dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

            dt_idx = np.argsort(-dt_scores, kind="mergesort")
            dt_scores = dt_scores[dt_idx]
            dt_ids = dt_ids[dt_idx]

            dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
            dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

            gt_ig = np.concatenate([e["gt_ignore"] for e in E])
            # num gt anns to consider
            num_gt = np.count_nonzero(gt_ig == 0)

            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m, np.logical_not(dt_ig))
            fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            # dt_pointers[cat_id][area_idx] = {
            #     "dt_ids": dt_ids,
            #     "tps": tps,
            #     "fps": fps,
            # }

            for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_tp = len(tp)
                rc = tp / num_gt
                # if num_tp:
                #     recall[iou_thr_idx, cat_idx, area_idx] = rc[
                #         -1
                #     ]
                # else:
                #     recall[iou_thr_idx, cat_idx, area_idx] = 0
                return_dict['recall'][(iou_thr_idx, cat_idx, area_idx)] = rc[-1]  if num_tp else 0


                # np.spacing(1) ~= eps
                pr = tp / (fp + tp + np.spacing(1))
                pr = pr.tolist()

                # Replace each precision value with the maximum precision
                # value to the right of that recall level. This ensures
                # that the  calculated AP value will be less suspectable
                # to small variations in the ranking.
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                rec_thrs_insert_idx = np.searchsorted(
                    rc, lvis_self.params.rec_thrs, side="left"
                )

                pr_at_recall = [0.0] * self.num_recalls

                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except:
                    pass
                #precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)
                return_dict['precision'][(iou_thr_idx,cat_idx, area_idx)] = np.array(pr_at_recall)
        #print("cat_id: {} time: {}".format(cat_id, time.time() - start_t))
        return return_dict

class LVISEval:
    def __init__(self, lvis_gt, lvis_dt, iou_type="segm",output_dir='./',class_agnostic=False,class_merge=False,multi_process = 1
    , max_dets=300):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))
        # use lvis_dt to form self.output_dir if lvis_dt is a path
        self.output_dir = output_dir
        # mkdir if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        #self.logger.addHandler(logging.FileHandler(os.path.join(self.output_dir,'eval.log')))
        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))
        # conver all class to 1
        if class_merge:
            for ann in self.lvis_gt.dataset['annotations']:
                ann['category_id'] = 1
            self.lvis_gt.dataset['categories']+=[{'id':1,'name':'person'}]
            self.lvis_gt._create_index()
        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = (defaultdict(list))  # gt for evaluation
        self._dts = (defaultdict(list))  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts
        self.per_img_info = mp.Manager().dict()# zmz,add this to record missing num per img 
        self.per_img_info = {}

        self.multi_process = multi_process

        self.class_agnostic = class_agnostic
        # save self.logger to txt file 
        self.resume = True
        self.logger.info("Start evaluate")
        # save self.logger to file
        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids()) #len(self.lvis_gt.get_cat_ids()) = 1124, maybe val only have 1124?
        self.params.max_dets = max_dets

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None
        

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=[1] if self.class_agnostic else cat_ids)
        )
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        # img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}  # not used
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"]) # checked all 1
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {d["id"]: [] for d in img_data} # not used

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            # if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
            #     if cat_id != 1: #为了保留预测结果，
            #         continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()
    def _prepare_ori(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for false positives.
        self.img_nel = {d["id"]: d["not_exhaustive_category_ids"] for d in img_data}

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        freq_groups = [[] for _ in self.params.img_count_lbl]
        cat_data = self.lvis_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            # frequency = _cat_data["frequency"]
            frequency = _cat_data.get("frequency", "f")
            freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info("Running per image evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]
        if not self.class_agnostic:
            self._prepare_ori()
        else:
            self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id)
            for img_id in tqdm(self.params.img_ids)
            for cat_id in cat_ids
        }
        #我需要保留 cat_id的结果但是，在真正算的是又是与所有预测结果算的呢。
       
        self.logger.info("IOU finished.")
        #loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng)
            for cat_id in tqdm(cat_ids)
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]
        # self.ious = {
        #     (img_id, cat_id): self.compute_iou2(img_id, cat_id)
        #     for img_id in tqdm(self.params.img_ids)
        #     for cat_id in cat_ids
        # }
        # self.logger.info("IOU finished.")
        # self.eval_imgs = [
        #     self.evaluate_img2(img_id, cat_id, area_rng)
        #     for cat_id in tqdm(cat_ids)
        #     for area_rng in self.params.area_rng
        #     for img_id in self.params.img_ids
        # ]
    # from memory_profiler import profile

    # @profile
    def evaluate_and_accumulate(self):
        """
        原始版本的evaluate太占用内存，这里考虑每次只记录一类的结果
        """
        self.logger.info("Running per class evaluation and accumulate.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))
        import sys
        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()
        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)
        last_cat_id = 0
        # -1 for absent categories
        if self.resume and os.path.exists(os.path.join(self.output_dir, "precision.npy")):
            # load existing results
            # np.save(os.path.join(self.output_dir, "precision.npy"), precision)
            # np.save(os.path.join(self.output_dir, "recall.npy"), recall)
            # check existing results     
            precision = np.load(os.path.join(self.output_dir, "precision.npy"))
            recall = np.load(os.path.join(self.output_dir, "recall.npy"))
            # with open(os.path.join(self.output_dir, "cat_id.txt"), "w") as f:
            #             f.write(str(cat_id))
            last_cat_id = int(open(os.path.join(self.output_dir, "cat_id.txt")).read())
        else:    
            precision = -np.ones(
                (num_thrs, num_recalls, num_cats, num_area_rngs)
            )
            recall = -np.ones((num_thrs, num_cats, num_area_rngs))
        dt_pointers = {}
        id2idx = {self.params.cat_ids[x]:x for x in range(num_cats)}
        
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}
        
        # import tracemalloc
        # tracemalloc.start()
        # snapshot1 = tracemalloc.take_snapshot()
        import time
        start = time.time()
        if self.multi_process == 1:
            for cat_id in tqdm(self.params.cat_ids):
                t = time.time()
                if self.resume and cat_id <= last_cat_id:
                    continue
                #print(sys.getsizeof(precision),"precision size")
                self.eval_imgs = []
                self.ious = {}
                cat_idx = id2idx[cat_id]
                for img_id in (self.params.img_ids):
                    
                    self.ious [(img_id, cat_id)]= self.compute_iou2(img_id, cat_id)
                for area_rng in self.params.area_rng:
                    for img_id in self.params.img_ids:
                        self.eval_imgs.append(self.evaluate_img2(img_id, cat_id, area_rng))
                for area_idx in range(num_area_rngs):
                    Na = area_idx * num_imgs
                    E = [
                        self.eval_imgs[0 + Na + img_idx]
                        for img_idx in range(num_imgs)
                    ]
                    # Remove elements which are None
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue

                    # Append all scores: shape (N,)
                    dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                    dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                    dt_idx = np.argsort(-dt_scores, kind="mergesort")
                    dt_scores = dt_scores[dt_idx]
                    dt_ids = dt_ids[dt_idx]

                    dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                    dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                    gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                    # num gt anns to consider
                    num_gt = np.count_nonzero(gt_ig == 0)

                    if num_gt == 0:
                        continue

                    tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                    fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    # dt_pointers[cat_id][area_idx] = {
                    #     "dt_ids": dt_ids,
                    #     "tps": tps,
                    #     "fps": fps,
                    # }

                    for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        num_tp = len(tp)
                        rc = tp / num_gt
                        if num_tp:
                            recall[iou_thr_idx, cat_idx, area_idx] = rc[
                                -1
                            ]
                        else:
                            recall[iou_thr_idx, cat_idx, area_idx] = 0

                        # np.spacing(1) ~= eps
                        pr = tp / (fp + tp + np.spacing(1))
                        pr = pr.tolist()

                        # Replace each precision value with the maximum precision
                        # value to the right of that recall level. This ensures
                        # that the  calculated AP value will be less suspectable
                        # to small variations in the ranking.
                        for i in range(num_tp - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        rec_thrs_insert_idx = np.searchsorted(
                            rc, self.params.rec_thrs, side="left"
                        )

                        pr_at_recall = [0.0] * num_recalls

                        try:
                            for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                                pr_at_recall[_idx] = pr[pr_idx]
                        except:
                            pass
                        precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)
                e_t = time.time()
                if self.resume:
                    # save precision and recall
                    np.save(os.path.join(self.output_dir, "precision.npy"), precision)
                    np.save(os.path.join(self.output_dir, "recall.npy"), recall)
                    # save cat_id
                    with open(os.path.join(self.output_dir, "cat_id.txt"), "w") as f:
                        f.write(str(cat_id))
                if cat_idx % 100 == 0:
                    # clean use_memory in self._gts and self._dts
                    for cat_id_ in self.params.cat_ids:
                        if cat_id_ >=  cat_id:
                            break
                        for img_id_ in self.params.img_ids:
                            self._gts.pop((img_id_, cat_id_), None)
                            #self._dts.pop((img_id_, cat_id_), None)
                    print("cat_idx: {}, time: {}".format(cat_idx, e_t - t))
                #print("time for cat_id",cat_id,"is",e_t-t)
                # snapshot2 = tracemalloc.take_snapshot()

                # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

                # stat = top_stats[0]
                # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
                # for line in stat.traceback.format():
                #     print(line)
                

            self.eval = {
                "params": self.params,
                "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "precision": precision,  #(10, 101, 1124, 4)
                "recall": recall,    #(10,1124,4) 除去第一类 其余都是-1
                "dt_pointers": dt_pointers,
            }
           
            
        else:
            print("multi process")
            
            num_cores = self.multi_process
            # def single_cat(cat_id):
            #     start_t = time.time()
            #     eval_imgs = []
            #     ious = {}
            #     cat_idx = id2idx[cat_id]
            #     return_dict = {}
            #     return_dict['cat_id'] = cat_id
            #     return_dict['recall'] = {}
            #     return_dict['precision'] = {}
            #     for img_id in (self.params.img_ids):
                    
            #         ious [(img_id, cat_id)]= self.compute_iou2(img_id, cat_id)
            #     for area_rng in self.params.area_rng:
            #         for img_id in self.params.img_ids:
            #             eval_imgs.append(self.evaluate_img2(img_id, cat_id, area_rng, ious))
            #     print("cat_id: {} eval time: {}".format(cat_id, time.time() - start_t))
            #     for area_idx in range(num_area_rngs):
            #         Na = area_idx * num_imgs
            #         E = [
            #             eval_imgs[0 + Na + img_idx]
            #             for img_idx in range(num_imgs)
            #         ]
            #         # Remove elements which are None
            #         E = [e for e in E if not e is None]
            #         if len(E) == 0:
            #             continue

            #         # Append all scores: shape (N,)
            #         dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
            #         dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

            #         dt_idx = np.argsort(-dt_scores, kind="mergesort")
            #         dt_scores = dt_scores[dt_idx]
            #         dt_ids = dt_ids[dt_idx]

            #         dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
            #         dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

            #         gt_ig = np.concatenate([e["gt_ignore"] for e in E])
            #         # num gt anns to consider
            #         num_gt = np.count_nonzero(gt_ig == 0)

            #         if num_gt == 0:
            #             continue

            #         tps = np.logical_and(dt_m, np.logical_not(dt_ig))
            #         fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

            #         tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            #         fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            #         # dt_pointers[cat_id][area_idx] = {
            #         #     "dt_ids": dt_ids,
            #         #     "tps": tps,
            #         #     "fps": fps,
            #         # }

            #         for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            #             tp = np.array(tp)
            #             fp = np.array(fp)
            #             num_tp = len(tp)
            #             rc = tp / num_gt
            #             # if num_tp:
            #             #     recall[iou_thr_idx, cat_idx, area_idx] = rc[
            #             #         -1
            #             #     ]
            #             # else:
            #             #     recall[iou_thr_idx, cat_idx, area_idx] = 0
            #             return_dict['recall'][(iou_thr_idx, cat_idx, area_idx)] = rc[-1]  if num_tp else 0
                        

            #             # np.spacing(1) ~= eps
            #             pr = tp / (fp + tp + np.spacing(1))
            #             pr = pr.tolist()

            #             # Replace each precision value with the maximum precision
            #             # value to the right of that recall level. This ensures
            #             # that the  calculated AP value will be less suspectable
            #             # to small variations in the ranking.
            #             for i in range(num_tp - 1, 0, -1):
            #                 if pr[i] > pr[i - 1]:
            #                     pr[i - 1] = pr[i]

            #             rec_thrs_insert_idx = np.searchsorted(
            #                 rc, self.params.rec_thrs, side="left"
            #             )

            #             pr_at_recall = [0.0] * num_recalls

            #             try:
            #                 for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
            #                     pr_at_recall[_idx] = pr[pr_idx]
            #             except:
            #                 pass
            #             #precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)
            #             return_dict['precision'][(iou_thr_idx,cat_idx, area_idx)] = np.array(pr_at_recall)
            #     print("cat_id: {} time: {}".format(cat_id, time.time() - start_t))
            #     return return_dict
            manager = mp.Manager()
            _gts = manager.dict()
            _dts = manager.dict()
            
            _gts, _dts = self._gts, self._dts
            
            global lvis_self
            lvis_self = self
            
            single_cat = single_pro(id2idx, self, num_area_rngs, num_imgs, num_recalls, _gts, _dts,self.per_img_info)
            # single_cat = single_pro(id2idx, self, num_area_rngs, num_imgs, num_recalls)
            
            with ThreadPool(processes=num_cores) as pool:
                results = pool.map(single_cat, self.params.cat_ids)
                pool.close()
                pool.join()
            #results = Parallel(n_jobs=num_cores)(delayed(single_cat)(cat_id) for cat_id in tqdm(self.params.cat_ids))
            for result in results:
                cat_id = result['cat_id']
                cat_idx = id2idx[cat_id]
                for key, value in result['recall'].items():
                    recall[key] = value
                for key, value in result['precision'].items():
                    precision[key[0],:,key[1],key[2]] = value
                for key, value in result['per_img_info'].items():
                    self.per_img_info[key] = value
            self.eval = {
                "params": self.params,
                "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "precision": precision,  #(10, 101, 1124, 4)
                "recall": recall,    #(10,1124,4) 除去第一类 其余都是-1
                "dt_pointers": dt_pointers,
            }
        end_time = time.time()
        print("Evaluation time: {:.2f} seconds.".format(end_time - start))
        



            

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._gts[img_id, cat_id]
            ]
            dt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._dts[img_id, cat_id]
            ]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious
    def check_all(self,):
        for img_id in self.params.img_ids:
            for cat_id in (self.params.cat_ids):
                res = self._get_gt_dt(img_id,cat_id)
                if res != ([],[]):
                    print(img_id,cat_id)
    def compute_iou2(self, img_id, cat_id,dt_cat_id=1):
        gt, dt2 = self._get_gt_dt(img_id, cat_id) # 这里我们只需要gt
        _, dt = self._get_gt_dt(img_id, dt_cat_id)#
        # merge dt and dt2
        dt = dt + dt2
        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img(self, img_id, cat_id, area_rng):
        """Perform evaluation for single category and image."""
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (
            self.ious[img_id, cat_id][:, gt_idx]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]

        # add this to count missing number per img.
        if area_rng == [0, 10000000000.0]: # here we count all size
            gt_m_50 = gt_m[0]
            matched_num = (gt_m_50>0).sum()
            missed_num = gt_m.shape[-1] - matched_num
            self.per_img_info[img_id,cat_id] = {#"category_id": cat_id,
            "area_rng": area_rng,
            "gt_num": num_gt,
            "dt_num": num_dt,
            "missed_num": missed_num}
        
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }
    def evaluate_img2(self, img_id, cat_id, area_rng,ious = None):
        """Perform evaluation for single category and image.
        这里需要考虑所有dt, AND ALL DT CLASS ID IS 1
        
        """
        #print("img_id: ", img_id, "cat_id: ", cat_id)
        gt, dt2 = self._get_gt_dt(img_id, cat_id)
        _ , dt = self._get_gt_dt(img_id, 1)
        # dt = dt + dt2
        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        if ious is None:
            ious = (
                self.ious[img_id, cat_id][:, gt_idx]
                if len(self.ious[img_id, cat_id]) > 0
                else self.ious[img_id, cat_id]
            )
        else:
            ious = (
                ious[img_id, cat_id][:, gt_idx]
                if len(ious[img_id, cat_id]) > 0
                else ious[img_id, cat_id]
            )

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            # or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]

        if area_rng == [0, 10000000000.0]: # here we count all size
            gt_m_50 = gt_m[0]
            matched_num = (gt_m_50>0).sum()
            missed_num = gt_m.shape[-1] - matched_num
            self.per_img_info[img_id,cat_id] = {#"category_id": cat_id,
            "area_rng": area_rng,
            "gt_num": num_gt,
            "dt_num": num_dt,
            "missed_num": missed_num}
        
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }
    def save_per_img_info(self):
        #  self.per_img_info[img_id,cat_id] = {#"category_id": cat_id,
        #     "area_rng": area_rng,
        #     "gt_num": num_gt,
        #     "dt_num": num_dt,
        #     "missed_num": missed_num}
        
        import os
        save_path = os.path.join(self.output_dir, "per_img_info.json")
        import json
        # traverse all classes to get classwise results
        class_images_fre = np.zeros((1203,)) # how much images for each class occur
        class_images_det = np.zeros((1203,)) # how much images for each class detected
        for key in self.per_img_info.keys():
            class_images_fre[key[1]-1] += 1 if self.per_img_info[key]["gt_num"] else 0
            class_images_det[key[1]-1] += 1 \
                if self.per_img_info[key]["gt_num"] != self.per_img_info[key]["missed_num"] else 0
        
        # save classwise results
        class_results = {}
        for i in range(1203):
            class_results[i] = {"ID":i+1,"name":ID2NAME[i+1],"gt": class_images_fre[i], "dt": class_images_det[i]
            , 'recall': class_images_det[i]/class_images_fre[i] if class_images_fre[i] != 0 else 0,"invoc":
            1 if i+1 in IN_VOC_ID_SET else 0}
        # save 2 csv
        import csv
        invoc_average_recall = 0
        outvoc_average_recall = 0
        in_num  = 0
        out_num = 0
        csv_save_path = os.path.join(self.output_dir, "class_results_.csv")
        with open(csv_save_path, 'w') as csvfile:
            fieldnames = ['ID', 'name', 'gt', 'dt', 'recall', 'invoc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key in class_results.keys():
                writer.writerow(class_results[key])
                if class_results[key]["invoc"]:
                    invoc_average_recall += class_results[key]["recall"]
                    in_num += 1 if class_results[key]["gt"] else 0
                else:
                    outvoc_average_recall += class_results[key]["recall"]
                    out_num += 1 if class_results[key]["gt"] else 0
            average_recall = (invoc_average_recall + outvoc_average_recall) / (in_num + out_num+1e-6)
            invoc_average_recall /= in_num + 1e-6
            outvoc_average_recall /= out_num + 1e-6
            writer.writerow({"ID": "", "name": "", "gt": "", "dt": "", "recall": invoc_average_recall, "invoc": "in_average"})
            writer.writerow({"ID": "", "name": "", "gt": "", "dt": "", "recall": outvoc_average_recall, "invoc": "out_average"}) 
            writer.writerow({"ID": "", "name": "", "gt": "", "dt": "", "recall": average_recall, "invoc": "average"})

        

        # with open(save_path, "w") as f:
        #     json.dump(self.per_img_info, f)
        # print("save per_img_info to {}".format(save_path))
        # # compute per class results
        # # for each img , for each category, successfully seg one object is enough.
        # if len(self.per_img_info)> 0:
        #     PER_IMG_INFO = repr(self.per_img_info) + "  # noqa" + "\n"
        #     with open("./tmp/per_img_info_val_nococoandlvis64.py", "wt") as f:
        #         f.write(f"PER_IMG_INFO = {PER_IMG_INFO}")
        


    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, num_cats, num_area_rngs)
        )
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[
                            -1
                        ]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(
                        rc, self.params.rec_thrs, side="left"
                    )

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)

        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,  #(10, 101, 1124, 4)
            "recall": recall,    #(10,1124,4) 除去第一类 其余都是-1
            "dt_pointers": dt_pointers,
        }

    def _summarize(
        self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None
    ):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s
    def _summarize_classwise(
        self, id,summary_type, iou_thr=None, area_rng="all", freq_group_idx=None
    ):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]
        id = id  # start from 0
        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, id, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, id, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"]   = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)
        self.results["APs"]  = self._summarize('ap', area_rng="small")
        self.results["APm"]  = self._summarize('ap', area_rng="medium")
        self.results["APl"]  = self._summarize('ap', area_rng="large")
        self.results["APr"]  = self._summarize('ap', freq_group_idx=0)
        self.results["APc"]  = self._summarize('ap', freq_group_idx=1)
        self.results["APf"]  = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)
    def save_classwise_info(self):
        max_dets = self.params.max_dets
        class_dict ={}
        for id in range(len(self.params.cat_ids)):
            results = {}
            results["AP"]   = self._summarize_classwise(id,'ap')
            results["AP50"] = self._summarize_classwise(id,'ap', iou_thr=0.50)
            results["AP75"] = self._summarize_classwise(id,'ap', iou_thr=0.75)
            results["APs"]  = self._summarize_classwise(id,'ap', area_rng="small")
            results["APm"]  = self._summarize_classwise(id,'ap', area_rng="medium")
            results["APl"]  = self._summarize_classwise(id,'ap', area_rng="large")
            # results["APr"]  = self._summarize_classwise(id,'ap', freq_group_idx=0)
            # results["APc"]  = self._summarize_classwise(id,'ap', freq_group_idx=1)
            # results["APf"]  = self._summarize_classwise(id,'ap', freq_group_idx=2)
            # class级别故没有group
            key = "AR@{}".format(max_dets)
            results[key] = self._summarize_classwise(id,'ar')

            for area_rng in ["small", "medium", "large"]:
                key = "AR{}@{}".format(area_rng[0], max_dets)
                results[key] = self._summarize_classwise(id,'ar', area_rng=area_rng)
            class_dict[id] = results
        import csv,os
        out_voc_AR = 0
        out_voc_num = 0
        in_voc_AR =   0
        in_voc_num =  0
        with open(os.path.join(self.output_dir,"class_wise_result_.csv"), 'w', newline='') as csvfile:
            fieldnames = ['id',"category",'ins_num',"in_voc"]
            fieldnames += list(class_dict[id].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for id in range(len(self.params.cat_ids)):
                temp_dict = {}
                temp_dict['id'] = self.params.cat_ids[id]
                temp_dict['category'] = self.lvis_gt.load_cats([temp_dict['id']])[0]['name'] # this may not right
                #temp_dict['ins_num'] = self.lvis_gt.load_cats([temp_dict['id']])[0]['instance_count'] # this may not right
                temp_dict['ins_num'] = self.lvis_gt.load_cats([temp_dict['id']])[0].get('instance_count',0) # this may not right
                temp_dict['in_voc'] = 1 if self.params.cat_ids[id] in IN_VOC_ID_SET else 0
                for key in class_dict[id].keys():
                    temp_dict[key]=class_dict[id][key]
                key = "AR@{}".format(max_dets)
                if temp_dict["in_voc"] == 0:
                    if temp_dict["AP"] >= 0:
                        out_voc_AR += temp_dict[key]
                        out_voc_num += 1
                else:
                    if temp_dict["AP"] >= 0:
                        in_voc_AR += temp_dict[key]
                        in_voc_num += 1
                
                
                writer.writerow(temp_dict)
            # Write out_voc_AR 2 csv 
            writer.writerow({key:out_voc_AR/out_voc_num})
            writer.writerow({key:in_voc_AR/(in_voc_num+1e-6)})



    def run(self):
        """Wrapper function which calculates the results."""
        if self.myself:
            self.evaluate_and_accumulate()
        else:
            self.evaluate()
            self.save_per_img_info()
            self.accumulate()
        self.save_classwise_info()
        self.summarize()

    def print_results(self):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[2:]) / 100)
                iou = "{:0.2f}".format(iou_thr)
            else:
                iou = "{:0.2f}:{:0.2f}".format(
                    self.params.iou_thrs[0], self.params.iou_thrs[-1]
                )

            if len(key) > 2 and key[2] in ["r", "c", "f"]:
                cat_group_name = key[2]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[2] in ["s", "m", "l"]:
                area_rng = key[2]
            else:
                area_rng = "all"
            
            print(template.format(title, _type, iou, area_rng, max_dets, cat_group_name, value))
            # save result to txt 
            with open(os.path.join(self.output_dir,"result.txt"), 'a') as f:
                f.write(template.format(title, _type, iou, area_rng, max_dets, cat_group_name, value)+'\n')
           
            
            
             


    def get_results(self):
        if not self.results:
            self.logger.warn("results is empty. Call run().")
        return self.results


class Params:
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = 300
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1
        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type
