import sys
import os
import glob
import numpy
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm

###################################
#
#   IMAGE INFORMATION
#
#   Dimesions: 1328 * 952
#
#   Section LT:
#       x: 68, y: 48
#       len(x): 496, len(y): 379
#
#   Section LB:
#       x: 68, y: 524
#
#   Section RB:
#       x: 732, y: 524
#
#   Section RT:
#       x: 732, y: 48
#
#   Color Reference:
#       x: 566, y: 525
#       len(x): 30, len(y): 378
#
###################################

CORNERS = [(68, 48), (68, 524), (732, 524), (732, 48)]
LENGTH_X = 496
LENGTH_Y = 379

COLOR_REF_X_LEN = 30
COLOR_REF_Y_LEN = 378
COLOR_REF_X_START = 566
COLOR_REF_Y_START = 525

### Desired Tasks ###
def run():
  image = Image.open('output/4-Legs-0 Part 0.png')
  array = image_to_array(image)
  get_hog(array)

### Quarter Functions ###
# for example, quarter_all_images('output/')
def quarter_all_images(save_dir, open_dir='./Glitch Categories/*/*' ):
  files = glob.glob(open_dir)

  previous = ''
  image_category_count = 0
  processed_count = 0
  for url in files:
    print "Processed {0} of {1}...\r".format(processed_count, len(files)),
    sys.stdout.flush()
    tokens = url.split('/')
    if tokens[2] == previous:
      image_category_count += 1
    else:
      image_category_count = 0

    quarter_image(url, image_category_count, save_dir)
    previous = tokens[2]
    processed_count += 1

def quarter_image(url, count, save_dir):
  image = Image.open(url)
  pixels = get_pixels(image)
  images = get_images(pixels)
  name = url.split('/')[2]

  folder_names = ['domain1', 'domain16', 'domain32', 'domain4']
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  for folder in folder_names:
    if not os.path.exists(save_dir + folder + '/'):
      os.makedirs(save_dir + folder + '/')

  for i in range(len(images)):
    save_image(images[i], '{2}/{0}-{1}.png'.format(name, count, folder_names[i]), save_dir)

def get_images(pixels):
  images = []
  
  for i in CORNERS:
    section = [ pixels[y][i[0]:i[0] + LENGTH_X] for y in xrange(i[1], i[1] + LENGTH_Y) ]
    images.append(section)

  return images

def save_image(pixels, name, directory):
  image = Image.new('RGB', (len(pixels[0]), len(pixels)))
  image.putdata(flatten(pixels))
  image.save(directory + name)

### SVM Vector Creation ###
def extract_characteristics(image):
  return

# modifies in place
def pixels_to_scalar(pixels):
  ref = get_color_reference()
  for row in pixels:
    row[0:] = [color_to_scalar(i, ref) for i in row[0:]]
  return

def color_to_scalar(rgb, color_ref):
  closest = min([(i, tuple_diff(rgb, color_ref[i])) for i in xrange(len(color_ref))], 
    key=lambda t: t[1])
  return (closest[0] + 0.0) / (len(color_ref) - 1.0)

ERR_MARGIN = 10
def get_color_reference(image='colors.png'):
  image = Image.open('colors.png')
  pixels = get_pixels(image)
  return [pixels[y][5] for y in xrange(0, COLOR_REF_Y_LEN - ERR_MARGIN)]

def save_color_reference(image='reference.png'):
  image = Image.open('example.png')
  pixels = get_pixels(image)
  colors = [pixels[y][COLOR_REF_X_START + 1:COLOR_REF_X_START + ERR_MARGIN] \
   for y in xrange(COLOR_REF_Y_START, COLOR_REF_Y_START + COLOR_REF_Y_LEN - ERR_MARGIN)]

  save_image(colors, 'colors.png', '')

### Utilities ###
def get_pixels(image):
  width, height = image.size
  pixels = list(image.getdata())
  return [pixels[i * width:(i + 1) * width] for i in xrange(height)]  

def flatten(pixels):
  return [pixels[i][j] for i in xrange(len(pixels)) for j in xrange(len(pixels[i]))]

def tuple_diff(tuple1, tuple2):
  return sum([abs(tuple1[i] - tuple2[i]) for i in xrange(len(tuple1))])




















