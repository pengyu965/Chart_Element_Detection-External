from mmdet.apis import init_detector, inference_detector
import mmcv
import os

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/epoch_20.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


image_folder = './images'
save_path = './tested_images'


for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)

    # test a single image and show the results
    # img = './sample.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, file_path)
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(file_path, result, out_file=os.path.join(save_path, file))