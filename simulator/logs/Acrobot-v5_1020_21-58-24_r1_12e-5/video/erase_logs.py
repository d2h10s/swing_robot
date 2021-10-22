import os

files = os.listdir('./')
for file in files:
    if '.avi' in file:
        num = file.split('.')[0]
        print(num)
        if int(num) % 1000 != 0:
            os.remove(file)
