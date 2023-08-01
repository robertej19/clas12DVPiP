# Specify the directory you want to resize the images in
import os
import PIL
from PIL import Image
#img_dir = '/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/t_xsec_unfolded'
img_dir = '/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/tdep_combined_plot'

Image.MAX_IMAGE_PIXELS = None
#make a dir with the same name as image dir but with _resized at the end
img_dir_resized = img_dir + '_resized'
#check that it doesn't exist first
if not os.path.exists(img_dir_resized):
    os.mkdir(img_dir_resized)
#

# Loop over every file in the directory
for filename in os.listdir(img_dir):
    if filename.endswith(".png"): 
        img_path = os.path.join(img_dir, filename)
        print("on file: ", img_path)
        output_dir = img_dir_resized + '/' + filename
        # Open an image file
        with Image.open(img_path) as img:
            # Decrease image quality with a smaller size
            #img.save(output_dir, quality=25, optimize=True)

            #get image size
            width, height = img.size
            # # Resize the image
            fact = 4
            img = img.resize((int(width/fact), int(height/fact)), PIL.Image.ANTIALIAS)
            # # Save the image with a new filename
            img.save(output_dir)
