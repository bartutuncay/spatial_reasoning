#from models.autoencoder import ImageGraphAE
from preprocessing_src.train_utils import image_downsample, loss
#from preprocessing_src.pcd_slice_extractor import get_slice
from models.image_encoder import ImageEncoder, convert_rotmat_quaternion
#from autoencoder import
from models.pcd_encoder import PCDEncoder
import cv2
import torch
import pandas as pd
import glob

torch.set_default_dtype(torch.float32)
image = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_images_undistorted/DSC_0675.JPG'
model_img = ImageEncoder(latent_dim=128,base_dim=32,token_dim=256,pose_dim=64,use_film=True,norm='group',groups=16,add_posenc=True)
model_pcd = PCDEncoder(latent_dim=128,layers=8,layers_mlp=4)
model_img.train()
model_pcd.train()

### data to forward pipeline
if False:
    model_img.train()
    img = cv2.imread('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_images_undistorted/DSC_0680.JPG')
    img_downsampled = torch.from_numpy(image_downsample(img,512)).unsqueeze(0).permute((0,3,1,2)).float()
    qw,qx,qy,qz = 0.546465,0.460047,-0.452757,0.533614
    tx,ty,tz = 3.80088,-3.73526,-1.28092
    R = convert_rotmat_quaternion(torch.Tensor([qw,qx,qy,qz])).unsqueeze(0)
    t = torch.Tensor([tx,ty,tz]).unsqueeze(0)
    #!apply .unsqueeze(0) to intr4 with single sample!
    print(R.shape)
    #module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6 eth_proxy
    model_img.forward(img_downsampled,R,t)

## data
image_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_images_undistorted/*.JPG'
images = glob.glob(image_path)
df_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_calibration_undistorted/images_parsed.csv'
pcd_dir = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/pcd_graph'
print(images)


def run_single_image(imgpath,df):
    img = cv2.imread(imgpath)
    img_name = str(imgpath).split('images/')[1]
    img_downsampled = torch.from_numpy(image_downsample(img,512)).unsqueeze(0).permute((0,3,1,2)).float()
    row = df.loc[df['image_name']==img_name]
    qw,qx,qy,qz = row['qw'],row['qx'],row['qy'],row['qz']
    tx,ty,tz = row['tx'],row['ty'],row['tz']

    R = convert_rotmat_quaternion(torch.Tensor([qw,qx,qy,qz])).unsqueeze(0)
    t = torch.Tensor([tx,ty,tz]).unsqueeze(0)

    i = model_img.forward(img_downsampled,R,t)
    return i

def run_pcd(df):
    # loads images from saved graph file

    return

def run_epoch(imgpath, dfpath, batch_size=4):
    meta_df = pd.read_csv(dfpath)
    
    # randomize image input
    latent_img = run_single_image(image_path,df_path)
    latent_pcd = 
    # loss
    return