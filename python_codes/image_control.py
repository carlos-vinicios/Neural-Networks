from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np
import random
import shutil
import cv2
import os

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()
		print()

def images_mean_witdth_height(base_path):
	width = 0
	height = 0
	iteration = 0
	total_iter = len(os.listdir(base_path))
	printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
	iteration+=1

	for filename in os.listdir(base_path):
		img = cv2.imread(base_path + filename)

		shape = img.shape
		height += shape[0]
		width += shape[1]
		printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
		iteration+=1

	return width/iteration, height/iteration

def images_resize(base_path, save_path, width, height):
	iteration = 0
	total_iter = len(os.listdir(base_path))
	printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
	iteration+=1
	for filename in os.listdir(base_path):
		img = cv2.imread(base_path + filename)
		res = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
		cv2.imwrite(save_path + filename, res)
		printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
		iteration+=1

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

	resp = input("Prosseguir com " + str(width) + "x" + str(height) + "?\n (Y|n)")
	
	if "n" == resp:
		width = int(input("Entre com a largura:"))
		height = int(input("Entre com a altura:"))
	images_resize(base_path_melanomas, save_path_melanomas, width, height)
	images_resize(base_path_normais, save_path_normais, width, height)

def image_rois(path_image, path_segment, save_path1, save_path2):
	images = sorted(os.listdir(path_image))
	masks = sorted(os.listdir(path_segment))
	iteration = 0
	total_iter = len(images)
	printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
	iteration+=1

	for image in images:
		msk = None
		img = cv2.imread(path_image + image)
		for mask in masks:
			if image[:-5] in mask:
				if "expert" in mask:
					msk = cv2.imread(path_segment + mask)
					res = cv2.bitwise_and(img, msk, img)
					cv2.imwrite(save_path1 + image, res)
					break
				elif "novice" in mask:
					msk = cv2.imread(path_segment + mask)
					res = cv2.bitwise_and(img, msk, img)
					cv2.imwrite(save_path2 + image, res)   
					break
		if msk is None:
            #res = create_mask(img)                
			cv2.imwrite(save_path2 + image, img)
		printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
		iteration+=1

def augmentation(base, qtd_splits, total_m, total_n):
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

	prop_rou = 0
	div = total_n/total_m
	
	if div % 1 == 0:
		prop_abs = div
	else:
		prop_abs = int(div)
		prop_rou = round(total_m * (div % 1))
	
	iteration = 0
	total_iter = total_m * qtd_splits
	printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
	iteration+=1

	for i in range(0, qtd_splits):
		cont = 0
		load_path = base + "/split" + str(i+1)+"/train/melanomas"
		save_path = base + "/split" + str(i+1)+"/train/melanomas"
		for filename1 in os.listdir(load_path):
			cont += 1

			img = load_img(load_path + "/" +filename1)
			x = img_to_array(img)  
			x = x.reshape((1,) + x.shape) 

			i = 0
			for batch in datagen.flow(x, save_prefix=filename1, save_to_dir=save_path, save_format='jpeg'):
				i += 1
				if cont >= prop_rou:
					if i > (prop_abs-2):
						break
				else:
					if i > (prop_abs-1):
						break
			
			printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
			iteration+=1
		

def randomSelec(base, qtd_splits, train_split_size, test_split_size, valid_split_size, img_melanomas, base_size_m, img_normais, base_size_n):
	iteration = 0
	total_iter = base_size_m * qtd_splits + base_size_n * qtd_splits
	printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
	iteration+=1
	pastas = ["/test", "/train", "/valid"]
	for split in range(0, qtd_splits):
		total_m = 0
		total_n = 0
		total_train_m = 0
		total_train_n = 0
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
			if pasta == '/train':
				num_m = round(base_size_m * train_split_size)
				num_n = round(base_size_n * train_split_size)
				total_m+=num_m
				total_n+=num_n
				total_train_m = num_m
				total_train_n = num_n
			elif pasta == '/test':
				num_m = round(base_size_m * test_split_size)
				num_n = round(base_size_n * test_split_size)
				total_m+=num_m
				total_n+=num_n
			elif pasta == '/valid':
				num_m = round(base_size_m * valid_split_size)
				num_n = round(base_size_n * valid_split_size)
				total_m+=num_m
				total_n+=num_n
				num_m += base_size_m-total_m
				num_n += base_size_n-total_n 

			#melanomas
			for i in range(0, int(num_m)):
				shutil.copy(load_path_melanomas + "/" + img_melanomas[i], path_m)
				printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
				iteration+=1
			#normais
			for i in range(0, int(num_n)):
				shutil.copy(load_path_normais + "/" + img_normais[i], path_n)
				printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
				iteration+=1
  
	aug = input("Realizar augmentation? (Y|n)")
	if aug == "y" or aug == "Y":
		augmentation(base, qtd_splits, total_train_m, total_train_n)

#def kFold(base, pastas):

opc = int(input("Opção desejada:\n1-Separação da base em splits randomicos\n2-Separação da base k-fold\n3-Criação de rois\n4-Normalização das imagens em uma base\n"))

if opc == 1:
	load_path_melanomas = input("Caminho para a base de melanomas: ")
	load_path_normais = input("Caminho para a base de normais: ")

	img_melanomas = os.listdir(load_path_melanomas)
	img_normais = os.listdir(load_path_normais)

	base_size = int(input("Tamanho total da base: "))
	base_size_m = int(input("Quantidade de melanomas: "))
	base_size_n = int(input("Quantidade de normais: "))
	train_split_size = float(input("Porcentagem de treino: "))
	test_split_size = float(input("Porcentagem de teste: "))
	valid_split_size = float(input("Porcentagem de validação:" ))

	base = "../../../" + str(int(train_split_size * 100)) + "_" + str(int(valid_split_size * 100)) + "_" + str(int(test_split_size * 100))

	os.mkdir(base) #cria a pasta para armazenar os splits

	randomSelec(base, 5, train_split_size, test_split_size, valid_split_size, img_melanomas, base_size_m, img_normais, base_size_n)
elif opc == 2:
	print("Ainda não disponivél")
elif opc == 3:
	base_path_melanomas = input("Caminho para as imagens de melanoma: ")
	base_path_melanomas_segment = input("Caminho para as mascaras do melanoma: ")

	base_path_normais = input("Caminho para as imagens normais: ")
	base_path_normais_segment = input("Caminho para as mascaras normais: ")

	save_path_melanomas = input("Caminho para salvar as rois do melanoma: ")
	save_path_melanomas_non = input("Caminho para salvar as imagens do melanoma sem mascaras ou de baixa qualidade: ")
	save_path_normais = input("Caminho para salvar as rois normais: ")
	save_path_normais_non = input("Caminho para salvar as imagens normais sem mascaras ou de baixa qualidade: ")

	if base_path_normais != "":
		print("Normais:")
		image_rois(path_image=base_path_normais, path_segment=base_path_normais_segment, save_path1=save_path_normais, save_path2=save_path_normais_non)
	if base_path_melanomas != "":
		print("Melanomas:")
		image_rois(path_image=base_path_melanomas, path_segment=base_path_melanomas_segment, save_path1=save_path_melanomas, save_path2=save_path_melanomas_non)
elif opc == 4:
	base_path_melanomas = input("Caminho para as imagens de melanoma: ")
	base_path_normais = input("Caminho para as imagens normais: ")

	save_path_melanomas = input("Caminho para salvar as imagens de melanoma: ")
	save_path_normais = input("Caminho para salvar as imagens normais: ")

	normalize_base_sizes(base_path_normais, base_path_melanomas, save_path_normais, save_path_melanomas)
else:
	print("Opção invalida")