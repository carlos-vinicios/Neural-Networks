import matplotlib as plt
import numpy as np
import random
import shutil
import cv2
import os

def images_resize(base_path, save_path, width, height):
    for filename in os.listdir(base_path):
        img = cv2.imread(base_path + filename)

        res = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(save_path + filename, res)
        

def images_mean_witdth_height(base_path):
    width = 0
    height = 0
    count = 0

    for filename in os.listdir(base_path):
        img = cv2.imread(base_path + filename)
        
        shape = img.shape
        height += shape[0]
        width += shape[1]
        count += 1

    return width/count, height/count

def image_rois(path_image, path_segment, save_path):
    images = sorted(os.listdir(path_image))
    masks = sorted(os.listdir(path_segment))
    for image, mask in zip(images, masks):
        img = cv2.imread(path_image + image)
        msk = cv2.imread(path_segment + mask)
        res = cv2.bitwise_and(img, msk, img) #READ ABOUT BITWISE
        cv2.imwrite(save_path + image, res)


#def apply_rois_base(base_path_normais, base_path_melanomas, save_path_normais, save_path_melanomas):

def normalize_base_sizes(base_path_normais, base_path_melanomas, save_path_normais, save_path_melanomas):
    mean_w_n, mean_h_n = images_mean_witdth_height(base_path_normais)
    mean_w_m, mean_h_m = images_mean_witdth_height(base_path_melanomas)

    #pegar o maior tamanho para evitar tantas perdas devido a distorções
    if mean_w_m > mean_w_n or mean_h_m > mean_h_n:
        width = mean_w_m
        height = mean_h_m
    else:
        width = mean_w_n
        height = mean_h_n

    print(width)
    print(height)

    images_resize(base_path_melanomas, save_path_melanomas, width, height)
    images_resize(base_path_normais, save_path_normais, width, height)


base_path_melanomas = "../../../Bases ISIC/Archive/melanomas/"
base_path_melanomas_segment = "../../../Bases ISIC/Archive/melanomas_seg/"

base_path_normais = "../../../Bases ISIC/Archive/normais/"
base_path_normais_segment = "../../../Bases ISIC/Archive/normais_seg/"

save_path_melanomas = "../../../Bases ISIC/Archive_rois/melanomas/"
save_path_normais = "../../../Bases ISIC/Archive_rois/normais/"


image_rois(path_image=base_path_normais, path_segment=base_path_normais_segment, save_path=save_path_normais)
image_rois(path_image=base_path_melanomas, path_segment=base_path_melanomas_segment, save_path=save_path_melanomas)
