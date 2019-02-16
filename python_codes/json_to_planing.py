import xlsxwriter
import json
import os

wb = xlsxwriter.Workbook('../../../Hyper_PH2.xlsx') #cria area de trabalho
sheet = wb.add_worksheet('Hyper_PH2')

titles = ['Modelo', 'Accuracy', 'Precision', 'F1', 'Sensibility', 'Specificity', 'Class_weight', 'Classification', 'Dropout', 'Neurons', 'Output', 'Pooling', 'Convs', 'Denses']

for i in range(len(titles)):
    sheet.write(0, i, titles[i]) #sheet.write(line, col, text, styles)

path = '../../../results/'

results = os.listdir(path)

line = 1

for result in results:
    with open(path+result) as r:
        data = json.load(r)
    
    sheet.write(line, 0, result[-14:-9])
    sheet.write(line, 1, data["acurracy"])
    sheet.write(line, 2, data["report"]["weighted avg"]["precision"])
    sheet.write(line, 3, data["report"]["weighted avg"]["f1-score"])
    sheet.write(line, 4, data["report"]["0"]["recall"])
    sheet.write(line, 5, data["report"]["1"]["recall"])
    if data['space']['class_weight']:
        sheet.write(line, 6, "T")
    else:
        sheet.write(line, 6, "F")
    sheet.write(line, 7, "Sigmoid")
    sheet.write(line, 8, data["space"]["dropout"])
    sheet.write(line, 9, data["space"]["neurons"])
    sheet.write(line, 10, data["space"]["output_layer"])
    sheet.write(line, 11, data["space"]["pooling"])
    sheet.write(line, 12, data["space"]["qtd_conv"])
    sheet.write(line, 13, data["space"]["qtd_dense"])
    line+=1

wb.close()