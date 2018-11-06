from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import shutil

pastas = ["/test", "/train", "/valid"]

load_path_melanomas = "base/melanomas"
load_path_normais = "base/normais"

img_melanomas = os.listdir(load_path_melanomas)
img_normais = os.listdir(load_path_normais)

for split in range(0, 5):
	path = "split"+str(split+1)
	os.mkdir(path)
	#criando pastas
	for pasta in pastas:
		sub_path = path + pasta
		os.mkdir(sub_path)
		path_n = sub_path + "/normais" #criando pasta de normais
		path_m = sub_path + "/melanomas" #criando pasta de melanomas
		os.mkdir(path_m)
		os.mkdir(path_n)
		random.shuffle(img_melanomas)
		random.shuffle(img_normais)
		if(pasta == '/test' or pasta == '/valid'):
			num_m = 8
			num_n = 32
		else:
			num_m = 24
			num_n = 96
		
		#melanomas
		for i in range(0, num_m):
			shutil.copy(load_path_melanomas + "/" + img_melanomas[i], path_m)
		#normais
		for i in range(0, num_n):
			shutil.copy(load_path_normais + "/" + img_normais[i], path_n)

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

for i in range(0, 5):
	load_path = "split" + str(i+1)+"/train/melanomas"
	save_path = "split" + str(i+1)+"/train/melanomas"
	print("Salvando")
	for filename1 in os.listdir(load_path):

		img = load_img(load_path + "/" +filename1)
		x = img_to_array(img)  
		x = x.reshape((1,) + x.shape) 

		i = 0
		for batch in datagen.flow(x, save_prefix=filename1, save_to_dir=save_path, save_format='jpeg'):
			i += 1
			if i > 2:
				break 