import cv2
import numpy as np

impath = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_images_undistorted/DSC_0680.JPG'
img = cv2.imread(impath)
print(img.shape)

def image_downsample(image,H=512):
    #convert image to 512x384
    W = int(img.shape[1]/img.shape[0]*H)
    image_resized = cv2.resize(image,(W,H),interpolation=cv2.INTER_LINEAR)
    #w1,w2 = int(W/2-H/2),int(W/2+H/2)
    #image_cropped = image_resized[:,w1:w2]
    return image_resized


img2 = image_downsample(img,512)
print(img2.shape)

    

def loss():
    return