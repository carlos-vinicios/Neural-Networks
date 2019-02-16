import os

bases_dir = '../../../80_10_10_ISIC_2016/'

for split in range(1, 6):
      print("Split: " + str(split))
      train_dir_m = os.listdir(bases_dir + "split"+str(split) + "/train/melanomas")
      train_dir_n = os.listdir(bases_dir + "split"+str(split) + "/train/normais")
      test_dir_m = os.listdir(bases_dir + "split"+str(split) + "/test/melanomas")
      test_dir_n = os.listdir(bases_dir + "split"+str(split) + "/test/normais")
      valid_dir_m = os.listdir(bases_dir + "split"+str(split) + "/valid/melanomas")
      valid_dir_n = os.listdir(bases_dir + "split"+str(split) + "/valid/normais")
      
      print("Melanomas:")
      
      cont=0
      print("Teste:")
      for test_image in test_dir_m:
        if test_image in train_dir_m:
          cont+=1
      print(cont)
      
      cont=0
      print("Validação:")
      for valid_image in valid_dir_m:
        if valid_image in train_dir_m:
          cont+=1
      print(cont)
      
      print("Normais:")
      
      cont=0
      print("Teste:")
      for test_image in test_dir_n:
        if test_image in train_dir_n:
          cont+=1
      print(cont)
      
      cont=0
      print("Validação:")
      for valid_image in valid_dir_n:
        if valid_image in train_dir_n:
          cont+=1
      print(cont)
 
      print()
      print()