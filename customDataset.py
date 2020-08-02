import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
class PlantDataset(Dataset):
    def __init__(self,anno,root_dir=None,transform=None):
        self.annotations=anno

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path=self.annotations[index][0]
        image=cv2.imread(img_path,cv2.IMREAD_COLOR)
        image=cv2.resize(image,(224,224))
        y_label=torch.tensor(int(self.annotations[index][1]))
        tr2=transforms.ToTensor()
        tr3=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        image=tr2(image)
        image=tr3(image)
        
        return (image,y_label)
