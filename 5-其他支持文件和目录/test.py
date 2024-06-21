from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

gpu_id = [0]
batch_size = 128
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
#ckpt_root = "/your/data/root/"
ckpt_root = "/home/sjtu/lyt_file/Violence-Detector/train_logs/"
ckpt_path = ckpt_root + "resnet18_pretrain_test/version_3/checkpoints/resnet18_pretrain_test-epoch=23-val_loss=0.05.ckpt"
logger = TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(accelerator='gpu', devices=gpu_id)
trainer.test(model, data_module) 
