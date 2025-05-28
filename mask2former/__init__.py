# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config,add_GLIP_config,add_region_clip_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_instance_clip_baseline_dataset_mapper import COCOInstanceCLIPBaselineDatasetMapper
from .data.dataset_mappers.coco_instance_fewshot_baseline_dataset_mapper import COCOInstanceFewshotBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.nyu_dataset_mapper import NYUDatasetMapper
from .data.dataset_mappers.dataset_mapper_anno import DatasetMapperWithAnno
from .data.utils.lvis_tools import register_lvis_instances_with_id
# models
from .maskformer_model import MaskFormer
from .segprompt import SegPrompt
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator

