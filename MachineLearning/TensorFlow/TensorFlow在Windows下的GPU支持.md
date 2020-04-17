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

那么怎么才能充分利用起GPU来计算呢？

## 开始

开始之前，笔者先假定读者已经安装好Python、Pip、virtualenv（可选）

我们先看下TensorFlow2的官方要求：[GPU支持](https://tensorflow.google.cn/install/gpu)

![image-20200417160358328](images\image-20200417160358328.png)

- [x] 显卡：GTX 960M，算力5.0，>=3.5（查询自己的显卡：[支持CUDA的GPU列表](https://developer.nvidia.com/cuda-gpus)）

- [x] 驱动：版本号441.22，>418.x
- [ ] CUDA：待安装（TensorFlow 2.1.0+ 目前只支持 CUDA 10.1）
- [ ] CUPTI：跟随CUDA附带
- [ ] cuDNN：待安装

下面来跟随笔者一个个解决吧

## 安装CUDA 10.1（for TensorFlow 2.1.0+）

![image-20200417162324375](images\image-20200417162324375.png)

[适用于Windows的CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)

![image-20200417162610128](images\image-20200417162610128.png)

简单来说，这一步之前我们已经确认了GPU，Windows 10，需要做的是安装VS和CUDA

先安装VS 2019：[Visual Studio 2019下载地址](https://visualstudio.microsoft.com/zh-hans/downloads/)

再安装CUDA 10.1：[CUDA全版本下载地址](https://developer.nvidia.com/cuda-toolkit-archive)

这里我们选择`CUDA Toolkit 10.1 update2`，Windows、x86_64、10、exe（network）方式下载安装

![image-20200417163210087](images\image-20200417163210087.png)

我们选择精简安装

![image-20200417163845618](images\image-20200417163845618.png)

如果看到以下界面说明安装顺利完成，我们接下来编译验证一下

![image-20200417170442855](images\image-20200417170442855.png)

到这一步，VS和CUDA就安装好了，接下来我们按照步骤验证一下是否ok

![image-20200417172259915](images\image-20200417172259915.png)

进入`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1`

![image-20200417170822526](images\image-20200417170822526.png)

我们双击运行`Samples_vs2019.sln`，可能会提示需要安装额外的工作负载

![image-20200417101731977](images\image-20200417101731977.png)

![image-20200417101812607](images\image-20200417101812607.png)

再次打开`Samples_vs2019.sln`

![image-20200417171452166](images\image-20200417171452166.png)

右击`1_Utilities`目录单击`生成`，看到以下输出说明成功了

![image-20200417171728431](images\image-20200417171728431.png)

进入`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\bin\win64\Debug`，我们看到需要的`deviceQuery`和`bandwidthTest`均已生成

![image-20200417171847075](images\image-20200417171847075.png)

我们用`cmd`运行以下3个测试命令

![image-20200417172039667](images\image-20200417172039667.png)

![image-20200417172122383](images\image-20200417172122383.png)

![image-20200417172141188](images\image-20200417172141188.png)

均看到`Result = PASS`，至此验证完成

这里提示一下，如果我们发现了下图中出现的报错，提示找不到`CUDA 10.2.props`

![image-20200417110228261](images\image-20200417110228261.png)

可以查阅官网发现VS 2019中`CUDA 10.2.props`等文件的位置在此

![image-20200417110436244](images\image-20200417110436244.png)

我们直接进入这个目录，把所有内容复制到目标位置即可解决

![image-20200417113102440](images\image-20200417113102440.png)

全部复制粘贴过去

![image-20200417113122019](images\image-20200417113122019.png)

## 安装cuDNN 7.6.5（for CUDA 10.1）

[cuDNN下载地址](https://developer.nvidia.com/cudnn)

选择下载`cuDNN Library for Windows 10`，这里需要免费注册并填写一些信息

![image-20200417172942508](images\image-20200417172942508.png)

下载完成并解压到`C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN`后，我们配置环境变量来完成最后的安装工作

![image-20200417174026869](images\image-20200417174026869.png)

运行一下

```
import tensorflow as tf
print("Is GPU Available: ", tf.test.is_gpu_available())
```

即可验证我们已经是使用GPU来运算了

![image-20200417174502374](images\image-20200417174502374.png)

小结：期间，因为要跑的ISR项目依赖TensorFlow 2.0.0，开始安装的CUDA 10.2根本不支持，运行项目一直各种无反应导致浪费大量时间。这里谨记，一定要严格按照官网文档来，最新的不一定是最好的。