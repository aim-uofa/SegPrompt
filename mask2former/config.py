# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from black import T
from detectron2.config import CfgNode as CN
import os

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY 

def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.OPEN_NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.GROUNDING_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.OPEN_BRANCH_WEIGHT = [1.0, 1.0, 1.0]

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.NUM_OPEN_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    # add config for example query
    cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_SIGMA = 0.9
    cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_PROJ = False
    cfg.MODEL.MASK_FORMER.FREEZE_BACKBONE = False
    cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_LOSS = 0.0
    cfg.MODEL.MASK_FORMER.EXAMPLE_QUERY_BATCHWISE_LOSS = 0.0
    # add config for using clip
    cfg.MODEL.MASK_FORMER.USE_CLIP_TEXTEMB = False
    cfg.MODEL.MASK_FORMER.USE_CLIP_IMGFEAT = False
    cfg.MODEL.MASK_FORMER.USE_VOC_EMB = True
    cfg.MODEL.MASK_FORMER.OPEN_BRANCH_DETACHED = False
    # add MaskNoise 
    cfg.MODEL.MASK_FORMER.ADD_MASK_NOISE = False
    cfg.MODEL.MASK_FORMER.ENFORCE_CLIP_PROJ = False
    cfg.MODEL.MASK_FORMER.OPEN_CLASSIFICATION = False
    cfg.MODEL.MASK_FORMER.CLIP_LOGIT_WEIGHT = (1.0,0.0)
    cfg.MODEL.MASK_FORMER.OPEN_ATT_MASK = True
    cfg.MODEL.MASK_FORMER.QUERY_REPEAT = 1
    cfg.MODEL.MASK_FORMER.QUERY_REPEAT_ONLY_MATCH = False
    cfg.MODEL.MASK_FORMER.ADDTIONAL_QUERY = False
    cfg.MODEL.MASK_FORMER.OPEN_SELF_ATT = False
    cfg.MODEL.MASK_FORMER.CLASS_AWARED = False
    cfg.MODEL.MASK_FORMER.DELETE_MASK = False
    cfg.MODEL.MASK_FORMER.SELF_ATT_INDEPENDENT = False
    cfg.MODEL.MASK_FORMER.BINARY_CLASSIFICATION = True
    cfg.MODEL.MASK_FORMER.MIN_MASK_AREA = 100
    cfg.MODEL.MASK_FORMER.NEW_MASK_IDSET = False
    cfg.MODEL.MASK_FORMER.HYBRID_MATCH = 1
    cfg.MODEL.MASK_FORMER.EXAMPLE_SUP = False
    cfg.MODEL.MASK_FORMER.MAX_EXAMPLE_NUM = 5

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.NO_ATT_MASK = False
    # for few shot, 0 means no few shot, 1 means fine-tune, 2 means freeze, 3 means eval
    cfg.MODEL.MASK_FORMER.TEST.FEW_SHOT = 0


    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8


    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.MASK_FORMER.USE_TASK_EMBEDDING = False
    cfg.MODEL.MASK_FORMER.ADD_TO_POSTION_EMBEDDING = True
    cfg.MODEL.MASK_FORMER.TASK_CLASSIFIER = False
    cfg.MODEL.MASK_FORMER.UPDATE_EXAMPLE_QUERY = True
    cfg.MODEL.MASK_FORMER.USE_GROUNDING_LOSS = False
    cfg.MODEL.MASK_FORMER.LOG_SCALE = 0.0
    cfg.MODEL.MASK_FORMER.PRIOR_PROB = 0.01
    cfg.MODEL.MASK_FORMER.FREEZE_PIXELDECODER = False

    # add to adapt glip training
    cfg.MODEL.MASK_FORMER.META_ARCHITECTURE = "MaskFormer"
    # to fix key error while glip training
    # for bianry train, when = 0 ,we dont use binary train.
    cfg.TEST.BINARY_THRES = 0.0
    cfg.TEST.IMS_PER_BATCH = 1

    cfg.DATASETS.CLIP_LOGIT_PATH = ""
    cfg.DATASETS.MAX_CLASSES = -1 # FOR few shot, -1 means all classes
    cfg.DATASETS.QUERY_PATH = ""
    cfg.DATASETS.JSON_PATH = ""
    cfg.DATASETS.NEG_SAMPLE = False
    cfg.DATASETS.FILTER_GT = False

    




def add_GLIP_config(cfg):
    """
    Add config for GLIP , modified from defaults.py
    """
    # cfg.MODEL = CN()  #这会把所有都清空哦
    cfg.MODEL.RPN_ONLY = False
    cfg.MODEL.BOX_ON = True
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.DEVICE = "cuda"

    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

    cfg.MODEL.RPN_ARCHITECTURE = "RPN"
    cfg.MODEL.DEBUG = False  # add debug flag
    cfg.MODEL.ONNX = False  # add onnx flag

    # If the WEIGHT starts with a catalog://, like :R-50, the code will look for
    # the path in paths_catalog. Else, it will use it as the specified absolute
    # path
    cfg.MODEL.WEIGHT = ""
    cfg.MODEL.PRETRAIN_NAME = ""

    # If LINEAR_PROB = True, only the last linear layers in rpn and roi_head are trainable
    cfg.MODEL.LINEAR_PROB = False

    # -----------------------------------------------------------------------------
    # Multitask Training / Test specific parameters
    # -----------------------------------------------------------------------------
    cfg.MODEL.MULTITASK = CN(new_allowed=True)

    # -----------------------------------------------------------------------------
    # INPUT
    # -----------------------------------------------------------------------------
    #cfg.INPUT = CN() #need to check
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing
    cfg.INPUT.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1333
    # Values to be used for image normalization
    cfg.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    # Values to be used for image normalization
    cfg.INPUT.PIXEL_STD = [1., 1., 1.]
    # Convert image to BGR format (for Caffe2 models), in range 0-255
    cfg.INPUT.TO_BGR255 = True
    cfg.INPUT.FORMAT = ''
    cfg.INPUT.FIX_RES = False

    # -----------------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------------
    cfg.AUGMENT = CN()
    cfg.AUGMENT.USE_RA = 0
    cfg.AUGMENT.FLIP_PROB_TRAIN = 0.5
    cfg.AUGMENT.VERTICAL_FLIP_PROB_TRAIN = 0.0
    cfg.AUGMENT.MULT_MIN_SIZE_TRAIN = ()

    cfg.AUGMENT.BRIGHTNESS = 0.0
    cfg.AUGMENT.CONTRAST = 0.0
    cfg.AUGMENT.SATURATION = 0.0
    cfg.AUGMENT.HUE = 0.0

    cfg.AUGMENT.CROP_PROB = 0.5
    cfg.AUGMENT.CROP_MIN_IOUS = (0.1, 0.3, 0.5, 0.7, 0.9)
    cfg.AUGMENT.CROP_MIN_SIZE = 0.3

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    #cfg.DATASETS = CN()
    # List of the dataset names for training, as present in paths_catalog.py
    cfg.DATASETS.TRAIN = ()
    # List of the dataset names for testing, as present in paths_catalog.py
    cfg.DATASETS.TEST = ()
    # Use is_crowd label
    cfg.DATASETS.USE_CROWD = False
    cfg.DATASETS.CLASS_AGNOSTIC = False
    cfg.DATASETS.CLASS_CONCAT = False
    cfg.DATASETS.MAX_BOX = -1
    cfg.DATASETS.SAMPLE_RATIO = 0.0
    cfg.DATASETS.FEW_SHOT = 0
    # SHUFFLE_SEED != 0 means shuffle the dataset in the few shot setting
    cfg.DATASETS.SHUFFLE_SEED = 0
    cfg.DATASETS.PREDEFINED_TEXT = ''
    cfg.DATASETS.ALTERNATIVE_TRAINING = False
    cfg.DATASETS.MULTISTAGE_TRAINING = False
    cfg.DATASETS.REGISTER = CN(new_allowed=True)
    cfg.DATASETS.BOX_THRESHOLD = 0.1
    # Duplicate Dataset
    cfg.DATASETS.COCO_COPY = 1
    cfg.DATASETS.LVIS_COPY = 1
    cfg.DATASETS.FLICKR_COPY = 1
    cfg.DATASETS.MIXED_COPY = 1
    cfg.DATASETS.OBJECT365_COPY = 1
    cfg.DATASETS.VG_COPY = 1
    cfg.DATASETS.OI_COPY = 1
    cfg.DATASETS.IN_COPY = 1

    # Duplicate Dataset
    cfg.DATASETS.COCO_COPY = 1
    cfg.DATASETS.FLICKR_COPY = 1
    cfg.DATASETS.MIXED_COPY = 1
    cfg.DATASETS.OBJECT365_COPY = 1
    cfg.DATASETS.VG_COPY = 1
    cfg.DATASETS.OI_COPY = 1
    cfg.DATASETS.IN_COPY = 1
    cfg.DATASETS.GENERAL_COPY = -1
    cfg.DATASETS.GENERAL_COPY_TEST = -1

    # OD to Grounding
    cfg.DATASETS.RANDOM_SAMPLE_NEG = -1
    cfg.DATASETS.ADD_DET_PROMPT = False
    cfg.DATASETS.ADD_DET_PROMPT_ADVANCED = False
    cfg.DATASETS.USE_OD_AUG = False
    cfg.DATASETS.USE_COCO_FORMAT = False
    cfg.DATASETS.CONTROL_PROB = ()
    cfg.DATASETS.DISABLE_SHUFFLE = False
    cfg.DATASETS.PROMPT_VERSION = ""
    cfg.DATASETS.PROMPT_LIMIT_NEG = -1
    cfg.DATASETS.POS_QUESTION_PROB = 0.6
    cfg.DATASETS.NEG_QUESTION_PROB = 0.8
    cfg.DATASETS.FULL_QUESTION_PROB = 0.5
    cfg.DATASETS.ONE_HOT = False
    cfg.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT = False

    cfg.DATASETS.DISABLE_CLIP_TO_IMAGE = False
    cfg.DATASETS.SEPARATION_TOKENS = " "

    # LVIS
    cfg.DATASETS.LVIS_USE_NORMAL_AP = False
    cfg.DATASETS.SPECIAL_SAFEGUARD_FOR_COCO_GROUNDING = False

    # Caption
    cfg.DATASETS.BING_INDEX_LIST = []
    cfg.DATASETS.CAPTION_MIN_BOX = 1
    cfg.DATASETS.REPLACE_CLEAN_LABEL = False
    cfg.DATASETS.FURTHER_SCREEN = False
    cfg.DATASETS.CAPTION_CONF = 0.9
    cfg.DATASETS.CAPTION_NMS = 0.9
    cfg.DATASETS.PACK_RANDOM_CAPTION_NUMBER = 0
    cfg.DATASETS.INFERENCE_CAPTION = False
    cfg.DATASETS.SAMPLE_NEGATIVE_FOR_GROUNDING_DATA = -1.0
    cfg.DATASETS.RANDOM_PACK_PROB = -1.0
    cfg.DATASETS.NO_RANDOM_PACK_PROBABILITY = 0.0
    cfg.DATASETS.SAFEGUARD_POSITIVE_CAPTION = True
    cfg.DATASETS.CAPTION_FORMAT_VERSION = "v1"
    cfg.DATASETS.LOCAL_DEBUG = False


    # Od in the wild
    cfg.DATASETS.PREDEFINED_TEXT = None
    cfg.DATASETS.TRAIN_DATASETNAME_SUFFIX = ""
    cfg.DATASETS.TEST_DATASETNAME_SUFFIX = ""
    cfg.DATASETS.OVERRIDE_CATEGORY = None
    cfg.DATASETS.USE_OVERRIDE_CATEGORY = False
    cfg.DATASETS.SUPRESS_QUERY = None
    cfg.DATASETS.USE_SUPRESS_QUERY = False
    cfg.DATASETS.USE_CAPTION_PROMPT = False
    cfg.DATASETS.CAPTION_PROMPT = None

    cfg.DATASETS.FLICKR_GT_TYPE = "separate"

    # VQA
    cfg.DATASETS.DIVER_BOX_FOR_VQA = False
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    #cfg.DATALOADER = CN()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # If > 0, this enforces that each collated batch should have a size divisible
    # by SIZE_DIVISIBILITY
    cfg.DATALOADER.SIZE_DIVISIBILITY = 0
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    # Define min number of keypoints required from GT, for example 10 out of 17
    cfg.DATALOADER.MIN_KPS_PER_IMS = 0
    # Use random sampler during training
    cfg.DATALOADER.USE_RANDOM_SEED = False

    cfg.DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE = False
    # ---------------------------------------------------------------------------- #
    # Backbone options
    # ---------------------------------------------------------------------------- #
    # cfg.MODEL.BACKBONE = CN() # 需要check会不会有问题

    # The backbone conv body to use
    # The string must match a function that is imported in modeling.model_builder
    # (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
    # backbone)
    cfg.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

    # Add StopGrad at a specified stage so the bottom layers are frozen
    cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
    cfg.MODEL.BACKBONE.FREEZE = False
    cfg.MODEL.BACKBONE.GROUP = 1
    cfg.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4
    # Option to reset bn running statics
    cfg.MODEL.BACKBONE.RESET_BN = False
    # Backbone Normalization Level
    cfg.MODEL.BACKBONE.NORM_LEVEL = 3
    # BN for backbone
    cfg.MODEL.BACKBONE.USE_BN = False
    # Sync BN for backbone
    cfg.MODEL.BACKBONE.USE_SYNCBN = False
    cfg.MODEL.BACKBONE.USE_NSYNCBN = False
    # GN for backbone
    cfg.MODEL.BACKBONE.USE_GN = False
    # Evo Norm for backbone
    cfg.MODEL.BACKBONE.USE_EN = False
    # Layers for backbone
    cfg.MODEL.BACKBONE.USE_DFCONV = False
    cfg.MODEL.BACKBONE.USE_DYRELU = False
    cfg.MODEL.BACKBONE.USE_SE = False
    cfg.MODEL.BACKBONE.LAYER_SETUP = (3, 4, 6, 3)
    cfg.MODEL.BACKBONE.LAYER_SEARCH = CN(new_allowed=True)
    cfg.MODEL.BACKBONE.OUT_FEATURES = ("stage2", "stage3", "stage4", "stage5")
    cfg.MODEL.BACKBONE.FPN_LAYER = ()
    cfg.MODEL.BACKBONE.USE_CHECKPOINT = False
    # Add JF efficient det cfgs
    cfg.MODEL.BACKBONE.EFFICIENT_DET_START_FROM = 3
    cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND = 0
    cfg.MODEL.BACKBONE.EFFICIENT_DET_BIFPN_VERSION = 0

    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.WEIGHT = ""
    cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
    cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 256
    cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
    cfg.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
    cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

    cfg.MODEL.LANGUAGE_BACKBONE.RNN_TYPE = "lstm"
    cfg.MODEL.LANGUAGE_BACKBONE.VARIABLE_LENGTH = True
    cfg.MODEL.LANGUAGE_BACKBONE.WORD_EMBEDDING_SIZE = 512
    cfg.MODEL.LANGUAGE_BACKBONE.WORD_VEC_SIZE = 512
    cfg.MODEL.LANGUAGE_BACKBONE.HIDDEN_SIZE = 512
    cfg.MODEL.LANGUAGE_BACKBONE.BIDIRECTIONAL = True
    cfg.MODEL.LANGUAGE_BACKBONE.INPUT_DROPOUT_P = 0.5
    cfg.MODEL.LANGUAGE_BACKBONE.DROPOUT_P = 0.2
    cfg.MODEL.LANGUAGE_BACKBONE.CORPUS_PATH = ""
    cfg.MODEL.LANGUAGE_BACKBONE.VOCAB_SIZE = 0

    cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True
    # ---------------------------------------------------------------------------- #
    # FPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.FPN = CN()
    cfg.MODEL.FPN.FREEZE = False
    cfg.MODEL.FPN.USE_GN = False
    cfg.MODEL.FPN.USE_RELU = False
    cfg.MODEL.FPN.USE_DYRELU = False
    cfg.MODEL.FPN.DROP_BLOCK = True
    cfg.MODEL.FPN.DROP_PROB = 0.3
    cfg.MODEL.FPN.DROP_SIZE = 3
    cfg.MODEL.FPN.USE_SPP = False
    cfg.MODEL.FPN.USE_PAN = False
    cfg.MODEL.FPN.USE_DYHEAD = False
    cfg.MODEL.FPN.RETURN_SWINT_FEATURE_BEFORE_FUSION = False
    # ---------------------------------------------------------------------------- #
    # BIFPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.BIFPN = CN()
    cfg.MODEL.BIFPN.NUM_REPEATS = 1
    cfg.MODEL.BIFPN.USE_ATTENTION = True

    # ---------------------------------------------------------------------------- #
    # Group Norm options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GROUP_NORM = CN()
    # Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
    cfg.MODEL.GROUP_NORM.DIM_PER_GP = -1
    # Number of groups in GroupNorm (-1 if using DIM_PER_GP)
    cfg.MODEL.GROUP_NORM.NUM_GROUPS = 16
    # GroupNorm's small constant in the denominator
    cfg.MODEL.GROUP_NORM.EPSILON = 1e-5

    # ---------------------------------------------------------------------------- #
    # Evo Norm options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.EVO_NORM = CN()
    # Number of groups in EvoNorm (-1 if using DIM_PER_GP)
    cfg.MODEL.EVO_NORM.NUM_GROUPS = 8
    # EvoNorm's small constant in the denominator
    cfg.MODEL.EVO_NORM.EPSILON = 1e-5

    # ---------------------------------------------------------------------------- #
    # RetinaNet Options (Follow the Detectron version)
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RETINANET = CN()
    # This is the number of foreground classes and background.
    cfg.MODEL.RETINANET.NUM_CLASSES = 81
    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    cfg.MODEL.RETINANET.NUM_CONVS = 4
    # During inference, #locs to select based on cls score before NMS is performed
    # per FPN level
    cfg.MODEL.RETINANET.PRE_NMS_TOP_N = 1000
    # Prior prob for the positives at the beginning of training. This is used to set
    # the bias init for the logits layer
    cfg.MODEL.RETINANET.PRIOR_PROB = 0.01
    # Inference cls score threshold, anchors with score > INFERENCE_TH are
    # considered for inference
    cfg.MODEL.RETINANET.INFERENCE_TH = 0.05
    # NMS threshold used in RetinaNet
    cfg.MODEL.RETINANET.NMS_TH = 0.4
    cfg.MODEL.RETINANET.DETECTIONS_PER_IMG = 100

    # ---------------------------------------------------------------------------- #
    # Focal Loss Options (Follow the Detectron version)
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.FOCAL = CN()
    # Weight for bbox_regression loss
    cfg.MODEL.FOCAL.BBOX_REG_WEIGHT = 4.0
    # Smooth L1 loss beta for bbox regression
    cfg.MODEL.FOCAL.BBOX_REG_BETA = 0.11
    # IoU overlap ratio for labeling an anchor as positive
    # Anchors with >= iou overlap are labeled positive
    cfg.MODEL.FOCAL.FG_IOU_THRESHOLD = 0.5
    # IoU overlap ratio for labeling an anchor as negative
    # Anchors with < iou overlap are labeled negative
    cfg.MODEL.FOCAL.BG_IOU_THRESHOLD = 0.4
    # Focal loss parameter: alpha
    cfg.MODEL.FOCAL.LOSS_ALPHA = 0.25
    # Focal loss parameter: gamma
    cfg.MODEL.FOCAL.LOSS_GAMMA = 2.0

    # ---------------------------------------------------------------------------- #
    # FCOS Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.FCOS = CN()
    cfg.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.INFERENCE_TH = 0.05
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOP_N = 1000

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.FCOS.NUM_CONVS = 4
    # if use deformable conv to align features
    cfg.MODEL.FCOS.USE_DFCONV = False

    # if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
    cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
    # IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
    cfg.MODEL.FCOS.IOU_LOSS_TYPE = "iou"

    cfg.MODEL.FCOS.NORM_REG_TARGETS = False
    cfg.MODEL.FCOS.CENTERNESS_ON_REG = False
    cfg.MODEL.FCOS.USE_GT_CENTER = False

    cfg.MODEL.FCOS.DETECTIONS_PER_IMG = 100
    cfg.MODEL.FCOS.USE_GN = False
    cfg.MODEL.FCOS.USE_BN = False

    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.0
    cfg.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN = 3000
    cfg.MODEL.FCOS.POST_NMS_TOP_N_TRAIN = 1000

    # ---------------------------------------------------------------------------- #
    # ATSS Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.ATSS = CN()
    cfg.MODEL.ATSS.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.ATSS.PRIOR_PROB = 0.01
    cfg.MODEL.ATSS.INFERENCE_TH = 0.05
    cfg.MODEL.ATSS.NMS_TH = 0.6
    cfg.MODEL.ATSS.PRE_NMS_TOP_N = 1000

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.ATSS.NUM_CONVS = 4
    # the channels of convolutions used in the cls and bbox tower
    cfg.MODEL.ATSS.CHANNELS = 128
    # if use deformable conv to align features
    cfg.MODEL.ATSS.USE_DFCONV = False

    # topk for selecting candidate positive samples from each level
    cfg.MODEL.ATSS.TOPK = 9

    # Weight for bbox_regression loss
    cfg.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

    cfg.MODEL.ATSS.DETECTIONS_PER_IMG = 100
    cfg.MODEL.ATSS.USE_GN = False
    cfg.MODEL.ATSS.USE_BN = False

    cfg.MODEL.ATSS.USE_DYRELU = False
    cfg.MODEL.ATSS.USE_SE = False

    cfg.MODEL.ATSS.INFERENCE_TH_TRAIN = 0.0
    cfg.MODEL.ATSS.PRE_NMS_TOP_N_TRAIN = 3000
    cfg.MODEL.ATSS.POST_NMS_TOP_N_TRAIN = 1000
    # ---------------------------------------------------------------------------- #
    # DYHEAD Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.DYHEAD = CN()
    cfg.MODEL.DYHEAD.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.DYHEAD.PRIOR_PROB = 0.01

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.DYHEAD.NUM_CONVS = 4
    # the channels of convolutions used in the cls and bbox tower
    cfg.MODEL.DYHEAD.CHANNELS = 128
    cfg.MODEL.DYHEAD.GROUPS = 1
    # if use deformable conv to align features
    cfg.MODEL.DYHEAD.USE_DFCONV = False

    # topk for selecting candidate positive samples from each level
    cfg.MODEL.DYHEAD.TOPK = 9

    cfg.MODEL.DYHEAD.SCORE_AGG = "MEAN"  # MEAN or MAX, for binary focal loss score aggregation

    cfg.MODEL.DYHEAD.LOG_SCALE = 0.0  # temperature (dot product)
    cfg.MODEL.DYHEAD.SHALLOW_LOG_SCALE = 0.0  # # temperature (shallow contrastive)

    cfg.MODEL.DYHEAD.USE_GN = False
    cfg.MODEL.DYHEAD.USE_NSYNCBN = False
    cfg.MODEL.DYHEAD.USE_SYNCBN = False

    cfg.MODEL.DYHEAD.USE_DYFUSE = False
    cfg.MODEL.DYHEAD.USE_DYRELU = False

    cfg.MODEL.DYHEAD.CONV_FUNC = ''

    # CosineSimOutputLayers: https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L448-L464
    cfg.MODEL.DYHEAD.COSINE_SCALE = -1.0

    cfg.MODEL.DYHEAD.FUSE_CONFIG = CN()
    cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE = ""
    cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE = 256
    cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE = 256
    cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT = 0.1
    cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS = 2

    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False

    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT = 1.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA = 2.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA = 0.25

    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT = 1.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT = 1.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D = False

    cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT = False

    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = False

    # Controls for 
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = False

    # MLM Loss
    cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_OD = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_GOLD = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF = 1.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_OBJ_FOR_ONLY_POSITIVE  = False

    # Shallow Contrastive Loss (FPN)
    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_MAX_POSITIVE_ANCHORS = 100
    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_ZERO_PADS = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_HIDDEN_DIM = 64
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT = 1.0

    # Shallow Contrastive Loss (BACKBONE)
    cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS = False

    cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False

    # use checkpoint to save memory
    cfg.MODEL.DYHEAD.USE_CHECKPOINT = False

    # ---------------------------------------------------------------------------- #
    # RPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RPN = CN()
    cfg.MODEL.RPN.USE_FPN = False
    # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
    cfg.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
    # Stride of the feature map that RPN is attached.
    # For FPN, number of strides should match number of scales
    cfg.MODEL.RPN.ANCHOR_STRIDE = (16,)
    # RPN anchor aspect ratios
    cfg.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
    # Anchor shift away ration from the center for r,t,l,d
    cfg.MODEL.RPN.ANCHOR_SHIFT = (0.0, 0.0, 0.0, 0.0)
    # Use center to decide anchor size
    cfg.MODEL.RPN.USE_RELATIVE_SIZE = False
    # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    cfg.MODEL.RPN.STRADDLE_THRESH = 0
    # Anchor scales per octave for complex anchors
    cfg.MODEL.RPN.OCTAVE = 2.0
    cfg.MODEL.RPN.SCALES_PER_OCTAVE = 3
    # Minimum overlap required between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
    # ==> positive RPN example)
    cfg.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
    # Maximum overlap allowed between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
    # ==> negative RPN example)
    cfg.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
    # Total number of RPN examples per image
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    # Target fraction of foreground (positive) examples per RPN minibatch
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
    # Number of top scoring RPN proposals to keep after applying NMS
    cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
    # NMS threshold used on RPN proposals
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Proposal height and width both need to be greater than RPN_MIN_SIZE
    # (a the scale used during training or inference)
    cfg.MODEL.RPN.MIN_SIZE = 0
    # Number of top scoring RPN proposals to keep after combining proposals from
    # all FPN levels
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
    # Custom rpn head, empty to use default conv or separable conv
    cfg.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"
    cfg.MODEL.RPN.FREEZE = False
    cfg.MODEL.RPN.FORCE_BOXES = False
    cfg.MODEL.RPN.RETURN_FUSED_FEATURES = False

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.ROI_HEADS = CN()
    cfg.MODEL.ROI_HEADS.USE_FPN = False
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
    # Overlap threshold for an RoI to be considered background
    # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
    cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
    # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
    # These are empirically chosen to approximately lead to unit variance targets
    cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
    # RoI minibatch size *per image* (number of regions of interest [ROIs])
    # Total number of RoIs per training minibatch =
    #   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
    # E.g., a common configuration is: 512 * 2 * 8 = 8192
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    # Only used on test mode

    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
    # Overlap threshold used for non-maximum suppression (suppress boxes with
    # IoU >= this threshold)
    cfg.MODEL.ROI_HEADS.NMS = 0.5
    # Maximum number of detections to return per image (100 is based on the limit
    # established for the COCO dataset)
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100

    cfg.MODEL.ROI_BOX_HEAD = CN()
    cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
    cfg.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
    # Hidden layer dimension when using an MLP for the RoI box head
    cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
    # GN
    cfg.MODEL.ROI_BOX_HEAD.USE_GN = False
    # Dilation
    cfg.MODEL.ROI_BOX_HEAD.DILATION = 1
    cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
    cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
    # Use D2 style ROIAlignV2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_ALIGNED = False

    cfg.MODEL.ROI_MASK_HEAD = CN()
    cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
    cfg.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
    cfg.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
    # Whether or not resize and translate masks to the input image.
    cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
    cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
    # Dilation
    cfg.MODEL.ROI_MASK_HEAD.DILATION = 1
    # GN
    cfg.MODEL.ROI_MASK_HEAD.USE_GN = False
    # HG
    cfg.MODEL.ROI_MASK_HEAD.HG_SCALE = 1

    cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
    cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
    cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
    cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
    cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
    cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME = ()  # If left empty, use default names
    cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

    # ---------------------------------------------------------------------------- #
    # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    #cfg.MODEL.RESNETS = CN()  #需要check

    cfg.MODEL.RESNETS.USE_STEM3X3 = False
    cfg.MODEL.RESNETS.WITH_SE = False
    cfg.MODEL.RESNETS.USE_AVG_DOWN = False

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    cfg.MODEL.RESNETS.NUM_GROUPS = 1

    # Baseline width of each group
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True

    # Residual transformation function
    cfg.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
    # ResNet's stem function (conv1 and pool1)
    cfg.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

    # Apply dilation in stage "res5"
    cfg.MODEL.RESNETS.RES5_DILATION = 1

    cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

    cfg.MODEL.RESNETS.REVISION = "resnet_light"
    # Deformable convolutions
    cfg.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
    cfg.MODEL.RESNETS.WITH_MODULATED_DCN = False
    cfg.MODEL.RESNETS.DEFORMABLE_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # Swin Transformer
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_CHANNELS = (96, 192, 384, 768)
    cfg.MODEL.SWINT.DEPTHS = (2, 2, 6, 2)
    cfg.MODEL.SWINT.NUM_HEADS = (3, 6, 12, 24)
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.SWINT.VERSION = "v1"
    cfg.MODEL.SWINT.OUT_NORM = True
    cfg.MODEL.SWINT.LAYER_SCALE = 0

    # ---------------------------------------------------------------------------- #
    # CVT SPEC
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.SPEC = CN(new_allowed=True)

    # ---------------------------------------------------------------------------- #
    # CLIP SPEC
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.CONTEXT_LENGTH = 256  # default 77
    cfg.MODEL.CLIP.WIDTH = 512
    cfg.MODEL.CLIP.LAYERS = 12
    cfg.MODEL.CLIP.HEADS = 8
    cfg.MODEL.CLIP.DROP_PATH = 0.0
    cfg.MODEL.CLIP.TOKENIZER = "clip"
    cfg.MODEL.CLIP.VOCAB_SIZE = 49408

    # ---------------------------------------------------------------------------- #
    # SEARCH
    # ---------------------------------------------------------------------------- #

    cfg.SEARCH = CN()
    cfg.SEARCH.MAX_EPOCH = 20
    cfg.SEARCH.SELECT_NUM = 20
    cfg.SEARCH.POPULATION_NUM = 64
    cfg.SEARCH.MUTATION_NUM = 24
    cfg.SEARCH.CROSSOVER_NUM = 24
    cfg.SEARCH.MUTATION_PROB = 0.1

    # ---------------------------------------------------------------------------- #
    # Solver
    # ---------------------------------------------------------------------------- #
    # cfg.SOLVER = CN()  #需要check
    cfg.SOLVER.USE_AMP = False

    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.MULTI_MAX_ITER = ()  # set different max epoch for different stage
    cfg.SOLVER.MAX_EPOCH = 0  # any epoch number>0 will overwrite max_iter
    cfg.SOLVER.MULTI_MAX_EPOCH = ()  # set different max epoch for different stage

    cfg.SOLVER.OPTIMIZER = "SGD"  # "ADAMW"

    cfg.SOLVER.BASE_LR = 0.001

    cfg.SOLVER.LANG_LR = 0.00001
    cfg.SOLVER.BACKBONE_BODY_LR_FACTOR = 1.0

    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.GRAD_CLIP = 0.0
    # D2 gradient clip
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.MODEL_EMA = 0.0

    cfg.SOLVER.MOMENTUM = 0.9

    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR = 1.0

    # use cosine lr to replace default multistage
    cfg.SOLVER.USE_COSINE = False
    cfg.SOLVER.MIN_LR = 0.000001

    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)

    cfg.SOLVER.USE_AUTOSTEP = False
    cfg.SOLVER.STEP_PATIENCE = 5

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.SOLVER.CHECKPOINT_PER_EPOCH = -1.0
    cfg.SOLVER.TEST_WITH_INFERENCE = False
    cfg.SOLVER.AUTO_TERMINATE_PATIENCE = -1
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    cfg.SOLVER.IMS_PER_BATCH = 16
    # This is the max negative ratio allowed per batch
    cfg.SOLVER.MAX_NEG_PER_BATCH = 0.1

    cfg.SOLVER.SEED = 0
    cfg.SOLVER.DISABLE_OUTPUT_DISTRIBUTED = False


    cfg.SOLVER.PROMPT_PROBING_LEVEL = -1.0 
    # -1 means tuning the whole model; 
    # 1 means tuning the whole language model; 1.5 means tuning the box head as well

    cfg.SOLVER.FIND_UNUSED_PARAMETERS = True
    cfg.SOLVER.DATASET_LENGTH = -1 # Just for logging purpose
    cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE = None
    cfg.SOLVER.USE_EMA_FOR_MONITOR = False

    cfg.SOLVER.WEIGHT_DECAY_SCHEDULE = False
    cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO = 0.667

    # ---------------------------------------------------------------------------- #
    # Specific test options
    # ---------------------------------------------------------------------------- #
    # cfg.TEST = CN() #need to check
    cfg.TEST.EXPECTED_RESULTS = []
    cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
    cfg.TEST.DURING_TRAINING = False
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    cfg.TEST.IMS_PER_BATCH = 16
    # Special Test Configuration
    cfg.TEST.USE_MULTISCALE = False
    # cfg.TEST.SCALES = (400, 600, 800, 1000, 1200, 1400)
    # cfg.TEST.RANGES = ((96, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 192))
    cfg.TEST.SCALES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800)
    cfg.TEST.RANGES = ((96, 10000), (96, 10000), (64, 10000), (64, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 256), (0, 192), (0, 192), (0, 96))
    cfg.TEST.MAX_SIZE = 2500
    cfg.TEST.FLIP = True
    cfg.TEST.SPECIAL_NMS = 'none'  # ('none', 'soft-nms', 'vote', 'soft-vote')
    cfg.TEST.TH = 0.6  # threshold for nms or vote
    cfg.TEST.PRE_NMS_TOP_N = 1000
    cfg.TEST.NUM_CLASSES = 81
    cfg.TEST.SELECT_CLASSES = ()

    cfg.TEST.EVAL_TASK = ""
    cfg.TEST.SUBSET = -1
    cfg.TEST.CHUNKED_EVALUATION = -1
    cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
    # ---------------------------------------------------------------------------- #
    # Misc options
    # ---------------------------------------------------------------------------- #
    cfg.OUTPUT_DIR = "OUTPUT"

    #cfg.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
    #cfg.PATHS_CATALOG = "/home/muzhi/Mask2Former/maskrcnn_benchmark/config/paths_catalog.py"
    cfg.PATHS_CATALOG = "/home/zmz/Code/Mask2former_GLIP/maskrcnn_benchmark/config/paths_catalog.py"
    # TensorBoard experiment location
    cfg.TENSORBOARD_EXP = "OUTPUT"


# ---------------------------------------------------------------------------- #
# add region clip  below
"""
_C.MODEL.CLIP = CN()

_C.MODEL.CLIP.CROP_REGION_TYPE = "" # options: "GT", "RPN" 
_C.MODEL.CLIP.BB_RPN_WEIGHTS = None # the weights of pretrained MaskRCNN
_C.MODEL.CLIP.IMS_PER_BATCH_TEST = 8 # the #images during inference per batch

_C.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = False # if True, use the CLIP text embedding as the classifier's weights
_C.MODEL.CLIP.TEXT_EMB_PATH = None # "/mnt/output_storage/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth"
_C.MODEL.CLIP.OFFLINE_RPN_CONFIG = None # option: all configs of pretrained RPN
_C.MODEL.CLIP.NO_BOX_DELTA = False  # if True, during inference, no box delta will be applied to region proposals

_C.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None # if not None, it is the loss weight for bg regions
_C.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals
_C.MODEL.CLIP.MULTIPLY_RPN_SCORE = False  # if True, during inference, multiply RPN scores with classification scores
_C.MODEL.CLIP.VIS = False # if True, when visualizing the object scores, we convert them to the scores before multiplying RPN scores

_C.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
_C.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None # if not None, enables the openset/zero-shot training, the category embeddings during test

_C.MODEL.CLIP.CLSS_TEMP = 0.01 # normalization + dot product + temperature
_C.MODEL.CLIP.RUN_CVPR_OVR = False # if True, train CVPR OVR model with their text embeddings
_C.MODEL.CLIP.FOCAL_SCALED_LOSS = None # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss

_C.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH = None # the threshold of NMS in offline RPN
_C.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST = None # the number of region proposals from offline RPN
_C.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True # if True, pretrain model using image-text level matching
_C.MODEL.CLIP.PRETRAIN_ONLY_EOT = False # if True, use end-of-token emb to match region features, in image-text level matching
_C.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None # if not None, the number of RPN regions per image during pretraining
_C.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
_C.MODEL.CLIP.GATHER_GPUS = False # if True, gather tensors across GPUS to increase batch size
_C.MODEL.CLIP.GRID_REGIONS = False # if True, use grid boxes to extract grid features, instead of object proposals
_C.MODEL.CLIP.CONCEPT_POOL_EMB = None # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
_C.MODEL.CLIP.CONCEPT_THRES = None # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)

_C.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED = False # if True, use large-scale jittering (LSJ) pretrained RPN
_C.MODEL.CLIP.TEACHER_RESNETS_DEPTH = 50 # the type of visual encoder of teacher model, sucha as ResNet 50, 101, 200 (a flag for 50x4)
_C.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB = None # if not None, it uses the same concept embedding as student model; otherwise, uses a seperate embedding of teacher model
_C.MODEL.CLIP.TEACHER_POOLER_RESOLUTION = 14 # RoIpooling resolution of teacher model

_C.MODEL.CLIP.TEXT_EMB_DIM = 1024 # the dimension of precomputed class embeddings
_C.INPUT_DIR = "./datasets/custom_images" # the folder that includes the images for region feature extraction
_C.MODEL.CLIP.GET_CONCEPT_EMB = False # if True (extract concept embedding), a language encoder will be created
"""
def add_region_clip_config(cfg):

    cfg.MODEL.CLIP = CN()

    cfg.MODEL.CLIP.CROP_REGION_TYPE = "" # options: "GT", "RPN" 
    cfg.MODEL.CLIP.BB_RPN_WEIGHTS = None # the weights of pretrained MaskRCNN
    cfg.MODEL.CLIP.IMS_PER_BATCH_TEST = 8 # the #images during inference per batch

    cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = False # if True, use the CLIP text embedding as the classifier's weights
    cfg.MODEL.CLIP.TEXT_EMB_PATH = None # "/mnt/output_storage/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth"
    cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG = None # option: all configs of pretrained RPN
    cfg.MODEL.CLIP.NO_BOX_DELTA = False  # if True, during inference, no box delta will be applied to region proposals

    cfg.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None # if not None, it is the loss weight for bg regions
    cfg.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals
    cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE = False  # if True, during inference, multiply RPN scores with classification scores
    cfg.MODEL.CLIP.VIS = False # if True, when visualizing the object scores, we convert them to the scores before multiplying RPN scores

    cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
    cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None # if not None, enables the openset/zero-shot training, the category embeddings during test

    cfg.MODEL.CLIP.CLSS_TEMP = 0.01 # normalization + dot product + temperature
    cfg.MODEL.CLIP.RUN_CVPR_OVR = False # if True, train CVPR OVR model with their text embeddings
    cfg.MODEL.CLIP.FOCAL_SCALED_LOSS = None # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss

    cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH = None # the threshold of NMS in offline RPN
    cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST = None #
    cfg.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True # if True, pretrain model using image-text level matching
    cfg.MODEL.CLIP.PRETRAIN_ONLY_EOT = False # if True, use end-of-token emb to match region features, in image-text level matching
    cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None # if not None, the number of RPN regions per image during pretraining
    cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
    cfg.MODEL.CLIP.GATHER_GPUS = False # if True, gather tensors across GPUS to increase batch size
    cfg.MODEL.CLIP.GRID_REGIONS = False # if True, use grid boxes to extract grid features, instead of object proposals
    cfg.MODEL.CLIP.CONCEPT_POOL_EMB = None # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
    cfg.MODEL.CLIP.CONCEPT_THRES = None # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)
    
    cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED = False # if True, use large-scale jittering (LSJ) pretrained RPN
    cfg.MODEL.CLIP.TEACHER_RESNETS_DEPTH = 50 # the type of visual encoder of teacher model, sucha as ResNet 50, 101, 200 (a flag for 50x4)
    cfg.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB = None # if not None, it uses the same concept embedding as student model; otherwise, uses a seperate embedding of teacher model
    cfg.MODEL.CLIP.TEACHER_POOLER_RESOLUTION = 14 # RoIpooling resolution of teacher model

    cfg.MODEL.CLIP.TEXT_EMB_DIM = 1024 # the dimension of precomputed class embeddings
    cfg.INPUT_DIR = "./datasets/custom_images" # the folder that includes the images for region feature extraction
    cfg.MODEL.CLIP.GET_CONCEPT_EMB = False # if True (extract concept embedding), a language encoder will be created

    
