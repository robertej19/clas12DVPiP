from PIL import Image
import os
import re

# Directory containing the images
image_dir = '/mnt/c/Users/haley/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/tdep_test/'

# Get a list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Parse the filenames to get the x and y coordinates
coords_images = []
for image_file in image_files:
    x, y = map(float, re.findall(r'\d+.\d+', image_file))
    img = Image.open(os.path.join(image_dir, image_file))
    coords_images.append((x, y, img))

# Sort by x and y coordinates
coords_images.sort()

# Stitch the images together
stitch_image = Image.new('RGB', (len(set(x for x, _, _ in coords_images)) * img.size[0], 
                                 len(set(y for _, y, _ in coords_images)) * img.size[1]))

for i, (x, y, img) in enumerate(coords_images):
    stitch_image.paste(img, (int(x * img.size[0]), int(y * img.size[1])))

# Save the stitched image
stitch_image.save('stitch_image.png')