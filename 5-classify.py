import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from model import ViolenceClassifier
from dataset import CustomDataModule,pred_dataloader
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

        # load config
        self.cf = configparser.ConfigParser()
        self.cf.read("config.txt")  # 读取配置文件

        # log
        self.train_log_path=self.cf.get("log","train_log_path")
        self.train_log_name=self.cf.get("log","train_log_name")
        self.test_log_path=self.cf.get("log","test_log_path")
        
        # device
        self.gpu_enabled=self.cf.get("device","gpu_enabled")
        self.gpu_id=int(self.cf.get("device","gpu_id"))
        
        # training_settings
        self.lr=self.cf.get("training_settings","learning_rate")
        self.batch_size=int(self.cf.get("training_settings","batch_size"))
        self.version=self.cf.get("training_settings","version")# 训练集版本
        self.ckpt_name=self.cf.get("training_settings","ckpt_name")# 当前检查点的名称

        # data_set
        self.data_root=self.cf.get("data","root")# train,val,test所在的文件夹路径
        self.pred_root=self.cf.get("data","pred")# 需要进行预测的输入图像文件夹

        # 实例化模型
        if (self.ckpt_name):
            self.ckpt_path = self.train_log_path+self.train_log_name+'/version_'+str(self.version)+'/checkpoints/'+ self.ckpt_name
            self.model = ViolenceClassifier.load_from_checkpoint(self.ckpt_path)
        else:
            # 未配置模型
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
            accelerator='gpu',
            devices=self.gpu_id,
            logger=logger,
            callbacks=[checkpoint_callback]
        )

        # 开始训练
        trainer.fit(self.model, data_module)

    def test(self):
        data_module = CustomDataModule(batch_size=self.batch_size)
        logger = TensorBoardLogger("test_logs", name=self.train_log_name)
        trainer = Trainer(accelerator='gpu', devices=self.gpu_id,logger=logger)
        trainer.test(self.model, data_module) 

    #调用接口
    def classify(self, images):
        
        with torch.no_grad():
            # load to gpu
            images = torch.stack(images)
            images = images.cuda(self.gpu_id-1)
            # Make predictions
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)

            # Convert predictions to a Python list
            return preds.tolist()
    
    

    def classify2(self,path=None):
        #确定路径
        if path==None:
            path=self.pred_root
        
        # 初始化dataloader
        dataloader=pred_dataloader(path)

        # 初始化trainer
        logger = TensorBoardLogger("predict_logs", name=self.train_log_name)
        trainer=Trainer(accelerator='gpu', devices=self.gpu_id,logger=logger)

        # 预测每一类的可能性
        outputs=trainer.predict(self.model, dataloader,ckpt_path=self.ckpt_path)

        # 预测标签函数
        def label(a,b):
            #返回可能性高的类标签
            if a>=b:
                return 0
            else:
                return 1
        
        # 预测标签
        output=[label(outputs[i][0][0],outputs[i][0][1]) for i in range (len(outputs))]
        return output

if __name__=='__main__':
    test=ViolenceClass()
    print(test.classify2())
