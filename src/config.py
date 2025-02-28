import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
DATA_DIR  = r'data\*.png'
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 90
EPOCHS =20
MAX_SEQ_LENGTH = 6
DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAUSSIAN_BLUR =  A.GaussianBlur(p=1, sigma_limit=(0 , 2))
SALT_AND_PAPER = A.SaltAndPepper(amount=(0.02, 0.1),salt_vs_pepper=(0.5, 0.5), p=.5)
GRID_DISTORTION =  A.GridDistortion(num_steps=5, distort_limit=0.5, p=.5)
TRAIN_PROCESSING_PIPELINE = [GAUSSIAN_BLUR , SALT_AND_PAPER , GRID_DISTORTION, A.Normalize(), ToTensorV2()]
TEST_PROCESSING_PIPELINE = [A.Normalize(), ToTensorV2()]
