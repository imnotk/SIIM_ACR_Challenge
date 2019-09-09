import os
import shutil 
import glob
import numpy as np 

non_diagnostic = np.genfromtxt("/4T/Public/zhujian/siim_acr/dataset/no_dia_test.txt",'U')

train_path = glob.glob(os.path.join("/4T/Public/zhujian/siim_acr/data/dicom-images-test/",'*','*','*'))

for i in non_diagnostic:
    print(i)
    a = glob.glob(os.path.join("/4T/Public/zhujian/siim_acr/data/dicom-images-test/",i))
    # a = glob.glob(os.path.join("/4T/Public/zhujian/siim_acr/data/dicom-images-train/",'*',i,'*'))
    print(a)
    for j in a:
        shutil.move(j,"/4T/Public/zhujian/siim_acr/data/dicom_non_diagnose/")