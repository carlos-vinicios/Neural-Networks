from keras.preprocessing.image import img_to_array, load_img
from scipy.stats.stats import pearsonr
import numpy as np
import xlwt
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

base_path_melanomas = "../../../Bases ISIC/Archive_full_s_size/melanomas/"
base_path_normais = "../../../Bases ISIC/Archive_full_s_size/normais/"

wb = xlwt.Workbook() #cria area de trabalho
sheet = wb.add_sheet('Correlacao')

titles = ['Normal', 'Melanomas', 'Indice r']

#escrevendo os titulos
for i in range(len(titles)):
    sheet.write(0, i, titles[i]) #sheet.write(line, col, text, styles)

line = 1

iteration = 0
total_iter = len(os.listdir(base_path_melanomas)) * len(os.listdir(base_path_normais))
printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
iteration+=1

for melanoma in os.listdir(base_path_melanomas):
    for normal in os.listdir(base_path_normais):
        img1 = img_to_array(load_img(base_path_melanomas + melanoma))
        img2 = img_to_array(load_img(base_path_normais + normal))
        r = pearsonr(img1.ravel(), img2.ravel())[0]
        sheet.write(line, 0, melanoma)
        sheet.write(line, 1, normal)
        sheet.write(line, 2, str("{:10.4f}").format(r))
        line+=1
        printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
        iteration+=1
    printProgressBar(iteration, total_iter, prefix = 'Progress:', suffix = 'Complete', length = 50)
    iteration+=1

wb.save('../../../correlacao.xls')