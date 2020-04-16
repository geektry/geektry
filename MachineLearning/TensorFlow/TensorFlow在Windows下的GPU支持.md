# TensorFlow在Windows下的GPU支持

## 前言

在Windows下做一些训练工作的时候，会看到这样的提示：

![image-20200416174053351](images\image-20200416174053351.png)

然后执行以下代码

```
import tensorflow as tf
print("Is GPU Available: ", tf.test.is_gpu_available())

# Is GPU Available:  False
```

果然并没有使用GPU在计算，而TensorFlow从2开始CPU/GPU是均支持的

![image-20200416174440424](images\image-20200416174440424.png)

那么怎么才能充分利用起GPU的优势呢？

## 开始

看了TensorFlow官网中对GPU支持的页面[GPU支持](https://tensorflow.google.cn/install/gpu)

需要按照NVIDIA给出的[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)

![image-20200416180654352](images\image-20200416180654352.png)

系统依赖：

- 支持CUDA的GPU（[支持CUDA的GPU列表](https://developer.nvidia.com/cuda-gpus)）
- 被支持的版本的Windows
- 被支持的版本的Microsoft Visual Studio
- NVIDIA CUDA Toolkit

笔者的GPU是GeForce GTX 960M算力5.0勉强及格哈哈，Windows 10现成的，需要额外下载VS和CUDA

[Visual Studio 2019下载地址](https://visualstudio.microsoft.com/zh-hans/downloads/)

[CUDA Toolkit下载地址](https://developer.nvidia.com/cuda-downloads)

![image-20200416185116781](images\image-20200416185116781.png)