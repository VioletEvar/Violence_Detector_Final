# 老版本区别
run_classify不支持文件夹方式的图像识别指令，其余部分与新版相同

# 训练
python run_classify.py --train
# 测试
python run_classify.py --test
# 调用classify.py对图像进行分类
python run_classify.py --classify image_paths.txt
