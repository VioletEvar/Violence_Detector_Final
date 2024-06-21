# Violence_Detector_Final
# classify.py接口使用说明

本文档提供了如何使用 `ViolenceClass` 接口和 `run_classify.py` 脚本进行图像暴力检测的说明。

## 环境配置
使用Conda配置
python==3.8
pytorch==1.8.2
torchvision==0.9.2
pytorch-lightning==1.6.0

具体步骤如下：
```
# 创建新的 conda 环境，指定 Python 版本为 3.8
conda create -n violence-detector python=3.8
# 激活虚拟环境
conda activate violence-detector
# 使用pytorch官网[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)中的指令下载pytorch，torchvision和torchaudio
# 此处下载的是cpu版，如果有GPU可以依据官网下载对应版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts
# 下载pytorch-lighting
conda install pytorch-lightning==1.6.0 torchmetrics==0.4.1 -c conda-forge
# 下载pillow(PIL)
conda install pillow
```

## 文件组织
* 将所有py程序放在名为Violence-Detector的文件夹下；同时，建立data文件夹放在Violence-Detector中，将下载的数据放在data文件夹里；代码运行时将自动创建train_logs文件夹
* 注意：有两个可运行版本，带old前缀的为我们的第一版，run_classify.py使用方式与最新版本略有区别，详情见old_README.md

## 地址修改
* 将dataset.py中的data_root之后的地址修改为自己数据所在地址；
* 将test.py中的ckpt_root和ckpt_path修改为train_logs对应地址

## 测试脚本运行
* 注意: 将config.txt中信息修改为自己对应的data,log,device,training_settings信息；
* 注意：使用--classify 参数时，需先在run_classify.py所在目录下创建image_paths.txt文件用于存放图片绝对地址，一行一个地址。
* 注意：使用--classify2 参数时，需要在config.txt的data-pred中填入预测图片文件夹路径，或是使用--img参数直接进行输入。

## 训练模型

要训练模型，运行：
```sh
python run_classify.py --train
```

## 测试模型

要测试模型，运行：
```sh
python run_classify.py --test
```

## 图像分类

要分类单个图像，运行：
```sh
python run_classify.py --classify /path/to/image.jpg
```

分类图像路径文件中所包含的所有图像，运行：
```sh
python run_classify.py --classify /path/to/imagepath.txt
```

分类一个文件夹的多个图像，运行
```sh
python run_classify.py --classify2  --img /path/to/imagefolder
```
## 示例

假设有一个图像位于 `/path/to/image.jpg`，可以通过以下命令对其进行分类：

```sh
python run_classify.py --classify /path/to/image.jpg
```

假设所有需要分类的图像路径保存在`imagepath.txt`中，运行：
```sh
python run_classify.py --classify imagepath.txt
```

假设需要分类的图像保存在与同级的`data/test/`文件夹里：
可以运行：
```sh
python run_classify.py --classify2  --img ./data/test/
```
或者在`config.txt`中保存`pred=./data/test/`，随后直接运行：
```sh
python run_classify.py --classify2
```


输出将指示图像被分类为 `Violent`（暴力）或 `Non-Violent`（非暴力）。
