import numpy as np
from PIL import Image
from pylab import *
import os
import sys

label_file = os.path.expanduser('~/.keras/datasets/weedspic/label_png/0000.png')
current_dir = os.path.dirname(os.path.realpath(__file__))
save_file = os.path.join(current_dir, 'lables_adjusted/test.png')

im = Image.open(label_file)
pix = im.load()

if im.mode == '1':
    value = int(shade >= 127) # Black-and-white (1-bit)
elif im.mode == 'L':
    value = 10 # Grayscale (Luminosity)
elif im.mode == 'RGB':
    value = (shade, shade, shade)
elif im.mode == 'RGBA':
    value = (shade, shade, shade, 255)
elif im.mode == 'P':
    raise NotImplementedError("TODO: Look up nearest color in palette")
else:
    raise ValueError("Unexpected mode for PNG image: %s" % im.mode)

for i in range (0, 512):
  for j in range(0, 424):
	if pix[i, j] == 255:
            pix[i, j] = 200
im.save(save_file)

