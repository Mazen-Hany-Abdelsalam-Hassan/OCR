from torch.utils.data import Dataset
from src.config import TRAIN_PROCESSING_PIPELINE , TEST_PROCESSING_PIPELINE ,IMAGE_WIDTH , IMAGE_HEIGHT
import cv2
import albumentations as A
import torch

class DatasetOCR(Dataset):
    def __init__(self , x , y , for_what = 'Train'):
        super().__init__()
        self._X = x
        self._Y = y
        if for_what =='Train':
            self._preprocessing_pipeline = A.Compose(TRAIN_PROCESSING_PIPELINE)
        else :self._preprocessing_pipeline = A.Compose(TEST_PROCESSING_PIPELINE)


    def __getitem__(self,index):
        image_dir   = self._X[index]
        image_label = self._Y[index]
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        image = self._preprocessing_pipeline(image = image)['image']
        length_data = len(os.path.split(sample[2])[1].split('.')[0])
        return image , torch.tensor(image_label) , torch.tensor(length_data)
    def __len__(self):
        return len(self._X)

