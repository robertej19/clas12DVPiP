import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import sys
import os, subprocess
from pdf2image import convert_from_path
import math
from icecream import ic
import shutil
from PIL import Image, ImageDraw, ImageFont



def img_from_pdf(img_dir):
	image_files = []
	lists = os.listdir(img_dir)
	sort_list = sorted(lists)
	for img_file in sort_list:
		print("On file " + img_file)
		image1 = Image.open(img_dir+img_file)
		image_files.append(image1)

	return image_files


def append_images(images, xb_counter, direction='horizontal', 
                  bg_color=(255,255,255), aligment='center'):
    
    # Appends images in horizontal/vertical direction.

    # Args:
    #     images: List of PIL images
    #     direction: direction of concatenation, 'horizontal' or 'vertical'
    #     bg_color: Background color (default: white)
    #     aligment: alignment mode if images need padding;
    #        'left', 'right', 'top', 'bottom', or 'center'

    # Returns:
    #     Concatenated image as a new PIL image object.
    
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    if direction=='vertical':
        new_im = Image.new('RGB', (int(new_width+0), int(new_height+images[0].size[1]/2)), color=bg_color)


    offset = 0
    for im_counter,im in enumerate(images):
        ic(im_counter)
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def chunks(l, n):
	spits = (l[i:i+n] for i in range(0, len(l), n))
	return spits



#img_dir = "test_t_dep/"
img_dir = "assympics/"
images = img_from_pdf(img_dir)


print(len(images))
#print(images)
layers = []

q2_ranges = [0,1,2,3,4]
t_ranges = [0,1,2]
num_ver_slices = len(q2_ranges)
num_hori_slices = len(t_ranges)
#for i in range(0,int(len(images)/num_ver_slices)):
for i in range(0,num_hori_slices):
    #print("on step "+str(i))
    layer = list(images[i*num_ver_slices:i*num_ver_slices+num_ver_slices])
    #print(layer)
    #list(reversed(array))
    layers.append(layer)
#print(layers)
#sys.exit()
#print(layers[0])

horimg = []

#make vertical axis labels
#imglay1 = append_images(layers[0], -1, direction='vertical')
#imglay1.save("test1.png",optimize=True, quality=100)
#sys.exit()
#horimg.append(imglay1)


for xb_counter,layer in enumerate(layers):
    print("len of layers is {}".format(len(layer)))
    print("counter is {}".format(xb_counter))
    print("On vertical layer {}".format(xb_counter))
    #print(layer)
    imglay = append_images(layer, -1, direction='vertical')
    #imglay.save("testing1.jpg")
    horimg.append(imglay)

print("Joining images horizontally")
final = append_images(horimg, 0,  direction='horizontal')
final_name = "joined_pictures_{}.jpg".format(num_ver_slices)
final.save(final_name,optimize=True, quality=100)
print("saved {}".format(final_name))



