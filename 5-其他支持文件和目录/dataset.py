from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule

# 包含真值的图像数据
class CustomDataset(Dataset):
    def __init__(self, split,data_root):
        
        assert split in ["train", "val", "test"]
        self.data = [os.path.join(data_root, split+'/', i) for i in os.listdir(data_root + split)]
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path)
        y = int(img_path.split("/")[-1][0])  # 获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        return x,y


 
class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4,data_root="./data/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root=data_root

    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        self.train_dataset = CustomDataset("train",self.data_root)
        self.val_dataset = CustomDataset("val",self.data_root)
        self.test_dataset = CustomDataset("test",self.data_root)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# 不包含真值的图像数据，用以预测分析
class PredDataset(Dataset):
    def __init__(self, data_root):
        self.data = [os.path.join(data_root, i) for i in os.listdir(data_root)]
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 将图像转换为Tensor
        ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path)
        x = self.transforms(x)
        return x
    
def pred_dataloader(pred):
    return DataLoader(PredDataset(pred), batch_size=1, shuffle=False, num_workers=4)
