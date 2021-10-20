import os

files = os.listdir('./')
for file in files:
    if '.png' in file:
        num = file[3:-4]
        print(num)
        if int(num) % 100 != 0:
            os.remove(file)