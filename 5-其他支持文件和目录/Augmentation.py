from PIL import Image
import os
import numpy as np

#添加椒盐噪声
def sp_noise(image):
      output = np.zeros(image.shape,np.uint8)
      max_noise = 0.001
      min_noise = 0.0005
      prob = (max_noise - min_noise) * np.random.random_sample() + min_noise
      thres = 1 - prob 
      for i in range(image.shape[0]):
          for j in range(image.shape[1]):
              rdn = np.random.random()
              if rdn < prob:
                output[i][j] = 0
              elif rdn > thres:
                output[i][j] = 255
              else:
                output[i][j] = image[i][j]
      return output

#添加高斯噪声
def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

#split = 'train'
split = 'val'
data_root = "./violence_224/"
save_root_sp = "./Augmentation/sp"
save_root_gasuss = "./Augmentation/gasuss"
data = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]
path_sp_noise = [os.path.join(save_root_sp, split, i) for i in os.listdir(data_root + split)]
path_gasuss_noise = [os.path.join(save_root_gasuss, split, i) for i in os.listdir(data_root + split)]

i=0
for i in range(len(data)):
    img_path = data[i]
    noise_path = path_sp_noise[i]
    img = np.array(Image.open(img_path))
    img_sp_noise = sp_noise(img)
    img_noise = Image.fromarray(img_sp_noise)
    img_noise.save(noise_path)
    i=i+1

i=0
for i in range(len(data)):
    img_path = data[i]
    noise_path = path_gasuss_noise[i]
    img = np.array(Image.open(img_path))
    img_gasuss_noise = gasuss_noise(img)
    img_noise = Image.fromarray(img_gasuss_noise)
    img_noise.save(noise_path)
    i=i+1
