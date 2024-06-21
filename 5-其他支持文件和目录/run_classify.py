import argparse
from classify import ViolenceClass

def main(args):
    # 初始化模型
    vc = ViolenceClass()
    
    if args.train:
        # 训练模型
        vc.train_model()
    
    if args.test:
        # 测试模型
        vc.test()
    
    if args.classify:
        # 这里假设用户提供了一个包含图像路径的文件
        import os
        from PIL import Image
        
        # 读取图像路径
        with open(args.classify, 'r') as f:
            image_paths = f.read().splitlines()
        
        # 加载图像
        images = [Image.open(image_path) for image_path in image_paths]

        # 将图像转换为3*224*224的tensor
        images = [vc.transform(image) for image in images]
        
        # 分类图像
        results = vc.classify(images)
        
        # 输出结果
        for image_path, result in zip(image_paths, results):
            print(f"{image_path}: {'Violent' if result == 1 else 'Non-Violent'}")
    
    
    if args.classify2:
        import os

        #获取图像路径
        if args.img:
            data_root=args.img
        else:
            data_root=vc.pred_root
        data = [os.path.join(data_root, i) for i in os.listdir(data_root)]

        #使用classify2函数获取预测结果
        results = vc.classify2(args.img)

        #结果输出
        for image_path, result in zip(data, results):
            print(f"{image_path}: {'Violent' if result == 1 else 'Non-Violent'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Violence Detection Script')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--classify', type=str, help='Classify images. Provide a file with image paths, one per line.')
    parser.add_argument('--classify2', action='store_true', help='Classify images. use "--img" or edit config.txt to provide the path to the image folder')
    parser.add_argument('--img', type=str, help='Provide the path to the image folder')

    args = parser.parse_args()
    main(args)
