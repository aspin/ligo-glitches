import glob
import numpy
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from skimage import data, color, exposure

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

LENGTH_X = 496
LENGTH_Y = 379

CORNERS = [(68, 48), (68, 524), (732, 524), (732, 48)]

### Desired Tasks ###
def run():
  image = Image.open('output/4-Legs-0 Part 0.png')
  array = image_to_array(image)
  get_hog(array)

### HOG Processing ###
def get_hog(image):
  bw = color.rgb2gray(image)
  fd, hog_image = hog(bw, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

  ax1.axis('off')
  ax1.imshow(bw, cmap=plt.cm.gray)
  ax1.set_title('Input image')

  print hog_image

  # Rescale histogram for better display
  hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

  ax2.axis('off')
  ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
  ax2.set_title('Histogram of Oriented Gradients')
  plt.show()

def image_to_array(image):
    return numpy.array(image.getdata(),
                    numpy.uint8).reshape(image.size[1], image.size[0], 3)


### Quarter Functions ###
def quarter_all_images(save_dir):
  files = glob.glob('./Glitch Categories/*/*')

  previous = ''
  image_category_count = 0
  for url in files:
    tokens = url.split('/')
    if tokens[2] == previous:
      image_category_count += 1
    else:
      image_category_count = 0

    quarter_image(url, image_category_count, save_dir)
    previous = tokens[2]

def quarter_image(url, count, save_dir):
  image = Image.open(url)
  pixels = get_pixels(image)
  images = get_images(pixels)
  name = url.split('/')[2]
  for i in range(len(images)):
    save_image(images[i], '{0}-{1} Part {2}.png'.format(name, count, i), save_dir)

def get_pixels(image):
  width, height = image.size
  pixels = list(image.getdata())
  return [pixels[i * width:(i + 1) * width] for i in xrange(height)]  

def get_images(pixels):
  images = []
  
  for i in CORNERS:
    section = [ pixels[y][i[0]:i[0]+LENGTH_X] for y in xrange(i[1], i[1]+LENGTH_Y) ]
    images.append(section)

  return images

def flatten(pixels):
  return [pixels[i][j] for i in xrange(len(pixels)) for j in xrange(len(pixels[i]))]

def save_image(pixels, name, directory):
  image = Image.new('RGB', (LENGTH_X, LENGTH_Y))
  image.putdata(flatten(pixels))
  image.save(directory+name)

run()