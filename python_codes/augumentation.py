from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np
import random
import shutil
import os

pastas = ["/test", "/train", "/valid"]

load_path_melanomas = "../bases/PH2_original/melanomas"
load_path_normais = "../bases/PH2_original/normais"

img_melanomas = os.listdir(load_path_melanomas)
img_normais = os.listdir(load_path_normais)

base_size = 200
base_size_m = 40
base_size_n = 160
train_split_size = 0.4
test_split_size = 0.3
valid_split_size = 0.3

base = "../bases/" + str(int(train_split_size * 100)) + "_" + str(int(valid_split_size * 100)) + "_" + str(int(test_split_size * 100))

os.mkdir(base) #cria a pasta para armazenar os splits

for split in range(0, 5):
	path = base + "/split"+str(split+1)
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
		if pasta == '/test':
			num_m = base_size_m * test_split_size
			num_n = base_size_n * test_split_size
		elif pasta == '/valid':
			num_m = base_size_m * valid_split_size
			num_n = base_size_n * valid_split_size 
		else:
			num_m = base_size_m * train_split_size
			num_n = base_size_n * train_split_size
		
		#melanomas
		for i in range(0, int(num_m)):
			shutil.copy(load_path_melanomas + "/" + img_melanomas[i], path_m)
		#normais
		for i in range(0, int(num_n)):
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
	load_path = base + "/split" + str(i+1)+"/train/melanomas"
	save_path = base + "/split" + str(i+1)+"/train/melanomas"
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