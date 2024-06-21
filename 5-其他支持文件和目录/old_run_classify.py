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
        
        # 分类图像
        results = vc.classify(images)
        
        # 输出结果
        for image_path, result in zip(image_paths, results):
            print(f"{image_path}: {'Violent' if result == 1 else 'Non-Violent'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Violence Detection Script')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--classify', type=str, help='Classify images. Provide a file with image paths, one per line.')
    
    args = parser.parse_args()
    main(args)
