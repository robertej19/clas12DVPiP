import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from copy import copy
from utils.utils import dot
from utils.utils import mag
from utils.utils import mag2
from utils.utils import cosTheta
from utils.utils import angle
from utils.utils import cross
from utils.utils import vecAdd
from utils.utils import pi0Energy
from utils.utils import pi0InvMass
from utils.utils import getPhi
from utils.utils import getTheta
from utils.utils import getEnergy
from utils.utils import readFile
from utils import make_histos
from utils import histo_plotting
from utils import filestruct

pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

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


M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
E = 10.6


"""
        q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

        qrange = [q2bins[0], q2bins[-1]]
        xBrange = [xBbins[0], xBbins[-1]]
        trange = [tbins[0], tbins[-1]]


        sf_data_vals = []

        reduced_plot_dir = datafile_base_dir+reduced_xsection_plots_dir+run_identifiyer+"comparison/"
        print("SAVING PLOTS TO {}".format(reduced_plot_dir))
        if not os.path.exists(reduced_plot_dir):
            os.makedirs(reduced_plot_dir)

        for qindex, (qmin,qmax) in enumerate(zip(q2bins[0:-1],q2bins[1:])):
            print(" \n Q2 bin: {} to {}".format(qmin,qmax))
            #print(" Q INDEX: {}".format(qindex))
            for xindex, (xmin,xmax) in enumerate(zip(xBbins[0:-1],xBbins[1:])):
"""

def img_from_pdf(img_dir):
    image_files = []
    lists = os.listdir(img_dir)
    sort_list = sorted(lists)
    #short = [i[0:12] for i in sort_list]
    #for ii in short:
    #    print(ii)
    #sys.exit()
    #for f in sort_list:
    #    print(f)
    #sys.exit()
    # left = 450
    # right = 3300#2680
    # bottom = 1700#1515
    # top = 200
    left = 123 # to show vertical axis
    #left = 175 # to NOT show vertical axis

    right = 1225#2680
    bottom = 880#1515
    top = 120 #no title

    for img_file in sort_list:
        #print("On file " + img_file)
        image1 = Image.open(img_dir+img_file)
        #print(image1.size)
        image1 = image1.crop((left, top, right, bottom))
        #print(image1.size)
        #image1.show()
        #sys.exit()

        image_files.append(image1)

    return image_files

def append_images(images, xb_counter, direction='horizontal', 
                  bg_color=(255,255,255), aligment='center',special_layer=False,labeling=[]):
    
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

    scale_factor_w = 1

    if direction=='horizontal':
        new_width = sum(widths)+max(widths)*2
        new_height = max(heights)
        offset_origin = max(widths)
    else:
        new_width = max(widths)
        new_height = sum(heights)
        offset_origin = max(heights)

    
    if special_layer:
        new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

        offset = offset_origin
        for im_counter,im in enumerate(reversed(images)):
                y = 0
                if aligment == 'center':
                    y = int((new_height - im.size[1])/2)
                elif aligment == 'bottom':
                    y = new_height - im.size[1]
                fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 300)
                d = ImageDraw.Draw(new_im)
                d.text((offset,y), str(labeling[im_counter]), font=fnt, fill=(0,0,0))
                offset += int(im.size[0]*scale_factor_w)
        

        return new_im



    if direction=='vertical':
        new_im = Image.new('RGB', (int(new_width+0), int(new_height+images[0].size[1]/2)), color=bg_color)

        #make row of labels for
        #         fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 400)
        # d = ImageDraw.Draw(new_im)
        # d.text((max(widths)/2,max(heights)/2), "1.0", font=fnt, fill=(18,128,1))
        # print("appeneded text")
    else:
        new_im = Image.new('RGB', (new_width, new_height), color=bg_color)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 300)
        d = ImageDraw.Draw(new_im)
        d.text((int(max(widths)/2),int(max(heights)/1.8)), str(labeling[0]), font=fnt, fill=(0,0,0))
        print("appeneded text")
    #add text


    offset = offset_origin
    for im_counter,im in enumerate(reversed(images)):
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += int(im.size[0]*scale_factor_w)
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - scale_factor_w*im.size[0])/2)
            elif aligment == 'right':
                x = new_width - int(im.size[0]*scale_factor_w)
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def chunks(l, n):
	spits = (l[i:i+n] for i in range(0, len(l), n))
	return spits




fs = filestruct.fs()

#base_plot_dir = "Comparison_plots/"
#base_plot_dir = "tdepfigs/"
for t_count in range(0,1):
    #base_plot_dir = "/mnt/d/GLOBUS/CLAS12/APS2022/reduced_xsection_plots/outbending_rad_All_All_All_varied_t_phi_bins_small_phi_bins_excut_sigma_3comparison/t_0{}/".format(t_count)
    base_plot_dir = "/mnt/d/GLOBUS/CLAS12/APS2022/reduced_xsection_plots/outbending_rad_All_All_All_varied_t_phi_bins_small_phi_bins_excut_sigma_3comparison/t_combined_with_fit/"


    #q2bins,xBbins, tbins, phibins = fs.q2bins[0:7], fs.xBbins[0:11], np.array(fs.tbins[0:9]), fs.phibins
    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

    figures = os.listdir(base_plot_dir)

    #figures = figures[0:6]
    #print(figures)
    #figures.reverse()
    #print(figures)
    #sys.exit()

    images = img_from_pdf(base_plot_dir)
    #ys.exit()

    #images = images[0:6]

    num_ver_slices = len(q2bins)-1
    num_hori_slices = len(xBbins)-1
    #num_ver_slices = 2
    #num_hori_slices = 1


    layers = []

    layers = np.reshape(images,(num_ver_slices,num_hori_slices))
    layers = layers[:,:] #Last two columns are empty
    #layers = reversed(layers)
    #print(layers)


    horimg = []
    print(xBbins)
    #create special horizontal array with labels
    speciallayer = layers[0]
    imglay0 = append_images(speciallayer, -1, direction='horizontal',special_layer=True,labeling=xBbins)
    horimg.append(imglay0)



    for xb_counter,layer in enumerate(layers):
        layer = layer.tolist()
        layer.reverse()
        print("len of layers is {}".format(len(layer)))
        print("counter is {}".format(xb_counter))
        print("On vertical layer {}".format(xb_counter))
        #print(layer)
        print(q2bins)
        print(xb_counter)
        imglay = append_images(layer, -1, direction='horizontal',labeling=[q2bins[xb_counter]])
        #imglay.save("testing1.jpg")
        horimg.append(imglay)


    #    imglay.save("testing1.jpg")

    #     #print(horimg) 
    #     #ssreversed(horimg)   
    print("Joining images horizontally")
    final = append_images(horimg, 0,  direction='vertical')
    final_name = "joined_pictures_t_{}_{}.jpg".format(t_count,xb_counter)
    final.save(final_name,optimize=True, quality=100)
    print("saved {}".format(final_name))





    #for i in range(0,int(len(images)/num_ver_slices)):

    #print(layers)
    #sys.exit()
    #print(layers[0])

