import json
import os
import cv2 as cv
import numpy as np

dir_json_label = '/media/fjc/JUNCONG/dataset/Cityscapes/leftImg8bit/train_1280x800/bremen/json/outputs'
dir_label_mapped = '/media/fjc/JUNCONG/dataset/Cityscapes/leftImg8bit/train_1280x800/bremen/Labels_mapped'
dir_label = '/media/fjc/JUNCONG/dataset/Cityscapes/leftImg8bit/train_1280x800/bremen/Labels'

if not os.path.exists(dir_label_mapped):
    os.makedirs(dir_label_mapped)
if not os.path.exists(dir_label):
    os.makedirs(dir_label)

file_list = os.listdir(dir_json_label)
for json_file in file_list:
    with open(os.path.join(dir_json_label, json_file), 'r') as f:
        data = json.load(f)
        size = data['size']
        label_mapped = np.zeros((size['height'], size['width']), np.uint8)
        label = np.zeros((size['height'], size['width'], size['depth']), np.uint8) + 50
        is_labeled = data['labeled']
        if is_labeled:
            bbox_dict = data['outputs']
            bbox_list = bbox_dict['object']
            for bbox in bbox_list:
                bbox_coordinate = bbox['bndbox']
                xmin = bbox_coordinate['xmin']
                xmax = bbox_coordinate['xmax']
                ymin = bbox_coordinate['ymin']
                ymax = bbox_coordinate['ymax']
                if bbox['name'] == 'license plate':
                    label_mapped[ymin:ymax + 1, xmin:xmax + 1] = 2
                    label[ymin:ymax + 1, xmin:xmax + 1, :] = 255
                elif bbox['name'] == 'face':
                    label_mapped[ymin:ymax + 1, xmin:xmax + 1] = 1
                    label[ymin:ymax + 1, xmin:xmax + 1, :] = 125
                else:
                    print('Unknown class in {}'.format(json_file))
        else:
            print('No label in {}'.format(json_file))
    output_name = os.path.splitext(json_file)[0] + '.png'
    cv.imwrite(os.path.join(dir_label_mapped, output_name), label_mapped)
    cv.imwrite(os.path.join(dir_label, output_name), label)
    print('{} finished'.format(output_name))



