import os, shutil

files = os.listdir('./')
for file in files:
	if '.' not in file:
		num = file.replace('learning_model','')
		print(num)
		if int(num) % 1000 != 0:
		    shutil.rmtree(file)
