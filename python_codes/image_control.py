from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np
import shutil
import random
import cv2
import os

'''
#verificar a questão de criação de mascaras lendo o artigo
def create_mask(img):
    kernel = np.ones((3,3), np.uint8)

    #utilizando o morphologyEx e blur
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=10)
    cv2.imshow("Morphology", cv2.resize(closing, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    blur = cv2.blur(closing,(15,15))
    cv2.imshow("Blur", cv2.resize(blur, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #binarização da imagem
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", cv2.resize(gray, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, mask = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("Binary", cv2.resize(mask, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Preenche os quatro cantos da imagem binária
    w, h = mask.shape[::-1]
    cv2.floodFill(mask, None, (0, 0), 0)
    cv2.floodFill(mask, None, (w-1, 0), 0)
    cv2.floodFill(mask, None, (0, h-1), 0)
    cv2.floodFill(mask, None, (w-1, h-1), 0)
    cv2.imshow("Fill mask", cv2.resize(mask, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #lógica AND para obter da imagem original a encontrada pela criação do mask
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("BIT_AND", cv2.resize(img, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Canny Edges
    edges = cv2.Canny(img, 100,200)
    cv2.imshow("Canny", cv2.resize(edges, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dilate = cv2.dilate(edges,kernel,iterations=1)
    cv2.imshow("Dilate canny", cv2.resize(dilate, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dilate = cv2.bitwise_not(dilate)
    cv2.imshow("Bit not dilated canny", cv2.resize(dilate, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    #Lógica OR para retirar da imagem original os pêlos encontrados
    img = cv2.bitwise_or(img, img, mask=dilate )

    #Interpolação da imagem para preencher os vazios
    dilate = cv2.bitwise_not(dilate)
    inpaint = cv2.inpaint(img, dilate, 3,cv2.INPAINT_TELEA)
    cv2.imshow("Final", cv2.resize(inpaint, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return inpaint
'''

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

def images_resize(base_path, save_path, width, height):
    for filename in os.listdir(base_path):
        img = cv2.imread(base_path + filename)
        res = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path + filename, res)

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
        print("Imagem salva: " + image)

def augmentation(base, total_m, total_n):
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
  	
	for i in range(0, 5):
		cont = 0
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
				if cont == prop_rou:
					if i >= prop_abs:
						break
				else:
					if i > prop_abs:
						break
			cont += 1

def randomSelec(base, qtd_splits, train_split_size, test_split_size, valid_split_size, img_melanomas, base_size_m, img_normais, base_size_n):
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
			#normais
			for i in range(0, int(num_n)):
				shutil.copy(load_path_normais + "/" + img_normais[i], path_n)
  
	aug = input("Realizar augmentation? (Y|n)")
	if aug == "y" or aug == "Y":
		augmentation(base, total_train_m, total_train_n)

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