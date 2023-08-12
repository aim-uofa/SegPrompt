from doctest import OutputChecker
from re import L
from lvis_ow import LVIS, LVISResults, LVISEval
import os

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file", default="datasets/lvis/lvis_v1_val_resplit_r.json")
    parser.add_argument("--dt-json-file", default='hfai2_result/result_gnn/result/lvis_v1_val.segm.json')
    args = parser.parse_args()
    '''
    example: python tools/eval_rmlvisapi.py --gt-json-file datasets/lvis/lvis_v1_val_resplit010.json --dt-json-file hfai_result/m2f_124i_lvisvalnococo/lvis_010/inference/lvis_instances_results.json
    '''
    dataDir='datasets/LVIS/'
    dataType='val2017'
    annFile = args.gt_json_file
    import torch,itertools
    annType = 'segm'
    LVISGt=LVIS(annFile)
    #LVISGt.cat_img_map(0)
    resFile = args.dt_json_file
    from detectron2.utils.file_io import PathManager
    import json         
    # with PathManager.open(resFile, "w") as f:
    #     f.write(json.dumps(lvis_results))
    #     f.flush()
    #resFile = 'demo/instances_val2017_densecl_r101.json'
    class_agnostic = True
    class_merge = False
    LVISDt= LVISResults(LVISGt,resFile,class_agnostic=class_agnostic)
    #  CANNOT USE THIS 
    # according to resFile,get output_dir
    output_dir = os.path.dirname(resFile) + '100_r'
    print("output_dir",output_dir)
    LvisEval = LVISEval(LVISGt,LVISDt,annType,output_dir=output_dir,class_agnostic=class_agnostic,class_merge= class_merge,multi_process=1,max_dets = 100)
    #LvisEval.params.use_cats = 0
    if not class_agnostic or class_merge:
        # class-specific
        LvisEval.evaluate()
        LvisEval.save_per_img_info()
        LvisEval.accumulate()
        LvisEval.summarize()
        LvisEval.save_classwise_info()
        LvisEval.print_results()
    else:
        # class-agnostic
        LvisEval.evaluate_and_accumulate()
        LvisEval.save_per_img_info()
        LvisEval.summarize()
        LvisEval.save_classwise_info()
        LvisEval.print_results()


# main
if __name__ == '__main__':
    main()
