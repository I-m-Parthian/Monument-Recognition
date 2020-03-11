#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/ml/Desktop/Monument Recognition/Test/Aga Khan Palace"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

resize()
