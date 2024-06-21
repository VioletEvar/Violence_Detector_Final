import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from model import ViolenceClassifier
from dataset import CustomDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger




import  configparser
class ViolenceClass:
    def __init__(self):
        # Load a pre-trained ResNet model
        self.config_get()
        
        # modified by lyt 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def config_get(self):
        #load config
        self.cf = configparser.ConfigParser()
        self.cf.read("config.txt")  # 读取配置文件
        #self.log_name=self.cf.get("data","train_log_name")
        #self.log_name=self.cf.get("data","train_log_name")

        self.train_log_path=self.cf.get("log","train_log_path")
        self.train_log_name=self.cf.get("log","train_log_name")
        self.test_log_path=self.cf.get("log","test_log_path")

        self.gpu_enabled=self.cf.get("device","gpu_enabled")
        self.gpu_id=self.cf.get("device","gpu_id")

        self.lr=self.cf.get("training_settings","learning_rate")
        self.batch_size=int(self.cf.get("training_settings","batch_size"))
        self.version=self.cf.get("training_settings","version")
        self.ckpt_name=self.cf.get("training_settings","ckpt_name")

        if (self.ckpt_name):
            self.ckpt_path = self.train_log_path+self.train_log_name+'/version_'+str(self.version)+'/checkpoints/'+ self.ckpt_name
            self.model=ViolenceClassifier.load_from_checkpoint(self.ckpt_path)
        else:
            self.model = ViolenceClassifier(learning_rate=self.lr)


    def train_model(self):
        print("{} gpu: {}, batch size: {}, lr: {}".format(self.train_log_name, self.gpu_id, self.batch_size, self.lr))

        data_module = CustomDataModule(batch_size=self.batch_size)
        # 设置模型检查点，用于保存最佳模型
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename=self.train_log_name + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        logger = TensorBoardLogger("train_logs", name=self.train_log_name)

        # 实例化训练器
        trainer = Trainer(
            max_epochs=42,
            #max_epochs=1,
            accelerator='gpu',
            devices=self.gpu_id,
            logger=logger,
            callbacks=[checkpoint_callback]
        )

        # 实例化模型
        
        # 开始训练
        trainer.fit(self.model, data_module)

    def test(self):
        data_module = CustomDataModule(batch_size=self.batch_size)
        logger = TensorBoardLogger("test_logs", name=self.train_log_name)
        trainer = Trainer(accelerator='gpu', devices=self.gpu_id,logger=logger)
        trainer.test(self.model, data_module) 

    def classify(self, images):
        
        with torch.no_grad():
            # Apply the transformations
            images = torch.stack([self.transform(image) for image in images])

            # Make predictions
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)

            # Convert predictions to a Python list
            return preds.tolist()
    
    def classify_test(self):
        data_module = CustomDataModule(batch_size=1)
        data_module.setup()
        logger = TensorBoardLogger("predict_logs", name=self.train_log_name)
        trainer=Trainer(accelerator='gpu', devices=self.gpu_id,logger=logger)
        outputs=trainer.predict(self.model, data_module.train_dataloader(),ckpt_path=self.ckpt_path)
        '''
        # _,preds=torch.max(outputs, 1)
        # preds.tolist()
        print (outputs)
        aa=torch.tensor(outputs)
        a=aa[:][:][0]
        b=aa[:][:][1]
        zero=torch.zeros_like(a)
        one=torch.ones_like(b)
        output=torch.where(a>b,zero,one)
        '''
        def label(a,b):
            if a>=b:
                return 0
            else:
                return 1
        
        output=[label(outputs[i][0][0],outputs[i][0][1]) for i in range (len(outputs))]
        return output

if __name__=='__main__':
    test=ViolenceClass()
    print(test.classify_test())
