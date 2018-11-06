from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np
import os

opposite_by_gs = "R2_OPPOSITE_COLORS_BY_GS"
opposite_gr_gs = "R2_OPPOSITE_COLORS_GR_GS"
opposite_rg_gs = "R2_OPPOSITE_COLORS_RG_GS"

opposite_by_n = "R2_OPPOSITE_COLORS_BY_N"
opposite_gr_n = "R2_OPPOSITE_COLORS_GR_N"
opposite_rg_n = "R2_OPPOSITE_COLORS_RG_N"


'''
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''
		
datagen = ImageDataGenerator(
	shear_range=0.02,
	rotation_range=20,
	width_shift_range=0.05, 
	height_shift_range=0.05,
	zoom_range=0.3,
	fill_mode='constant',
	horizontal_flip=True,	
	cval=0
)
'''
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''

print("Salvando - opposite_by_gs")
for filename1 in os.listdir(opposite_by_gs):

	img = load_img(opposite_by_gs + "\\" +filename1)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename1, save_to_dir='R2_OPPOSITE_COLORS_BY_GS_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 
			
print("Salvando - opposite_gr_gs")
for filename2 in os.listdir(opposite_gr_gs):

	img = load_img(opposite_gr_gs + "\\" +filename2)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename2, save_to_dir='R2_OPPOSITE_COLORS_GR_GS_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 

print("Salvando - opposite_rg_gs")
for filename3 in os.listdir(opposite_rg_gs):
		
	img = load_img(opposite_rg_gs + "\\" +filename3)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename3, save_to_dir='R2_OPPOSITE_COLORS_RG_GS_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 

print("Salvando - opposite_by_n")			
for filename4 in os.listdir(opposite_by_n):

	img = load_img(opposite_by_n + "\\" +filename4)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename4, save_to_dir='R2_OPPOSITE_COLORS_BY_N_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 

print("Salvando - opposite_gr_n")
for filename5 in os.listdir(opposite_gr_n):

	img = load_img(opposite_gr_n + "\\" +filename5)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename5, save_to_dir='R2_OPPOSITE_COLORS_GR_N_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 

print("Salvando - opposite_rg_n")
for filename6 in os.listdir(opposite_rg_n):
		
	img = load_img(opposite_rg_n + "\\" +filename6)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename6, save_to_dir='R2_OPPOSITE_COLORS_RG_N_AUG', save_format='jpeg'):
		i += 1
		if i > 9:
			break 