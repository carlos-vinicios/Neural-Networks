from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np
import os

laod_path = "train_base_unbalanced/valid/melanoma"
save_path = "train_base_over/valid/melanoma"

datagen = ImageDataGenerator(
	shear_range=0.02,
	rotation_range=180,
	width_shift_range=0.05, 
	height_shift_range=0.05,
	zoom_range=0.3,
	fill_mode='constant',
	horizontal_flip=True,	
	cval=0
)

print("Salvando")
for filename1 in os.listdir(laod_path):

	img = load_img(laod_path + "/" +filename1)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape) 

	i = 0
	for batch in datagen.flow(x, save_prefix=filename1, save_to_dir=save_path, save_format='jpeg'):
		i += 1
		if i > 3:
			break 