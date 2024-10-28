import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# cmd = f"python tools/train.py configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"

cmd = f"python -m torch.distributed.launch --nproc_per_node=4 \
    ./tools/train.py  \
    --deterministic \
    --launcher pytorch \
    ./configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"

os.system(cmd)