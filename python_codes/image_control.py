from matplotlib import pyplot as plt
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

def create_mask(img):
    kernel = np.ones((3,3), np.uint8)

    #utilizando o morphologyEx e blur
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    blur = cv2.blur(closing,(15,15))
    
    #binarização da imagem
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #Preenche os quatro cantos da imagem binária
    w, h = mask.shape[::-1]
    cv2.floodFill(mask, None, (0, 0), 0)
    cv2.floodFill(mask, None, (w-1, 0), 0)
    cv2.floodFill(mask, None, (0, h-1), 0)
    cv2.floodFill(mask, None, (w-1, h-1), 0) 

    #lógica AND para obter da imagem original a encontrada pela criação do mask
    img = cv2.bitwise_and(img, img, mask=mask)

    #Canny Edges
    edges = cv2.Canny(img, 100,200)
    dilate = cv2.dilate(edges,kernel,iterations=1)
    dilate = cv2.bitwise_not(dilate)

    #Lógica OR para retirar da imagem original os pêlos encontrados
    img = cv2.bitwise_or(img, img, mask=dilate )
    
    #Interpolação da imagem para preencher os vazios
    dilate = cv2.bitwise_not(dilate)
    inpaint = cv2.inpaint(img, dilate, 3,cv2.INPAINT_TELEA)
    
    return inpaint

def image_rois(path_image, path_segment, save_path):
    images = sorted(os.listdir(path_image))
    masks = sorted(os.listdir(path_segment))
    for image in images:
        msk = None
        img = cv2.imread(path_image + image)
        for mask in masks:
            if image[:-5] in mask:
                if "expert" in mask:
                    msk = cv2.imread(path_segment + mask)
                    break
        if msk is None:
            res = create_mask(img)                
        else:
            res = cv2.bitwise_and(img, msk, img)
        cv2.imwrite(save_path + image, res)
        print("Imagem salva: " + image)

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


base_dir = "1000melanomas_normais"

base_path_melanomas = "../../../Bases ISIC/"+ base_dir +"/Archive/melanomas/"
base_path_melanomas_segment = "../../../Bases ISIC/"+ base_dir +"/Archive/melanomas_seg/"

base_path_normais = "../../../Bases ISIC/"+ base_dir +"/Archive/normais/"
base_path_normais_segment = "../../../Bases ISIC/"+ base_dir +"/Archive/normais_seg/"

save_path_melanomas = "../../../Bases ISIC/"+ base_dir +"/Archive_rois/melanomas/"
save_path_normais = "../../../Bases ISIC/"+ base_dir +"/Archive_rois/normais/"

#image_rois(path_image=base_path_normais, path_segment=base_path_normais_segment, save_path=save_path_normais)
#image_rois(path_image=base_path_melanomas, path_segment=base_path_melanomas_segment, save_path=save_path_melanomas)