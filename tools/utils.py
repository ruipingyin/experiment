import skimage
import skimage.io
import skimage.transform
import numpy as np

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
  img = skimage.io.imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img[:,:,0:3], (224, 224))
  return resized_img

def ColorPrint(str, idx = 0):
  print ('\033[1;31m' if idx == 0 else '\033[1;32m'), str, '\033[0m'

  
