import PIL

import cv2
print(cv2.__version__, cv2.__spec__)

print(cv2.getBuildInformation())
PIL.__version__, PIL.__spec__
import os

this_path = os.path.dirname('.')



INPUT_PATH = os.path.abspath(os.path.join(this_path, '..', 'input'))

TRAIN_DATA = os.path.join(INPUT_PATH, "train")

from glob import glob

filenames = glob(os.path.join(TRAIN_DATA, "*.jpg"))

len(filenames)
import matplotlib.pylab as plt

import numpy as np

from PIL import Image, ImageOps



def stage_1_PIL(filename):

    img_pil = Image.open(filename)

    img_pil = ImageOps.box_blur(img_pil, radius=1)

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(img_pil)



def stage_1_cv2(filename):

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.blur(img, ksize=(3, 3))

    img = cv2.flip(img, flipCode=1)

    return img
f = filenames[0]

r1 = stage_1_PIL(f) 

r2 = stage_1_cv2(f)



plt.figure(figsize=(16, 16))

plt.subplot(131)

plt.imshow(r1)

plt.subplot(132)

plt.imshow(r2)

plt.subplot(133)

plt.imshow(np.abs(r1 - r2))
def stage_1b_PIL(img_pil):

    img_pil = ImageOps.box_blur(img_pil, radius=1)

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(img_pil)



def stage_1b_cv2(img):    

    img = cv2.blur(img, ksize=(3, 3))

    img = cv2.flip(img, flipCode=1)

    return img
imgs_PIL = [Image.open(filename) for filename in filenames[:100]]
def cv2_open(filename):

    img = cv2.imread(filename)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



imgs_cv2 = [cv2_open(filename) for filename in filenames[:100]]
import numpy as np

from PIL import Image, ImageOps





def stage_2_PIL(filename):

    img_pil = Image.open(filename)

    img_pil = img_pil.resize((512, 512), Image.CUBIC)

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)

    return np.asarray(img_pil)



def stage_2_cv2(filename):

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    img = cv2.flip(img, flipCode=1)

    img = cv2.flip(img, flipCode=0)

    return img
f = filenames[0]

r1 = stage_2_PIL(f) 

r2 = stage_2_cv2(f)



plt.figure(figsize=(16, 16))

plt.subplot(131)

plt.imshow(r1)

plt.subplot(132)

plt.imshow(r2)

plt.subplot(133)

plt.imshow(np.abs(r1 - r2))
