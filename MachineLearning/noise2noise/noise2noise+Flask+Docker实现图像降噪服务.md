# noise2noise+Flask+Docker实现图像降噪服务

## 前言

最近了解到一个基于noise2noise来实现图像降噪的库

链接：https://github.com/yu4u/noise2noise

这是一个非官方的部分的使用Keras实现的`Noise2Noise: Learning Image Restoration without Clean Data`

与最初的论文有几点不同（但看noise2噪声训练框架如何工作并不是一个致命的问题）：

- 训练数据集（原论文：ImageNet，本实现：[2]）

- 模型（原论文：RED30 [3]，本实现：SRResNet [4]或UNet [5]）

[1] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data," in Proc. of ICML, 2018.

[2] J. Kim, J. K. Lee, and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," in Proc. of CVPR, 2016.

[3] X.-J. Mao, C. Shen, and Y.-B. Yang, "Image Restoration Using Convolutional Auto-Encoders with Symmetric Skip Connections," in Proc. of NIPS, 2016.

[4] C. Ledig, et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," in Proc. of CVPR, 2017.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in MICCAI, 2015.

这次我们把它用Flask封装成HTTP服务的形式，并且用Docker来部署

## 开始

它对外提供了一个`test_model.py`和一些配置参数以供调用`predict`方法

![image-20200429101144560](http://q9h8o4ouf.bkt.clouddn.com/image-20200429101144560.png)

![image-20200429101225947](http://q9h8o4ouf.bkt.clouddn.com/image-20200429101225947.png)

但是要做成服务形式暴露出来，我们先要对主流程代码进行一些小的重写，以下是项目总体结构

`src/`之外的是一些项目构建所需要的内容，`venv`是Python做环境隔离之用的`virtualenv`生成的目录，里边是本项目自己的Python运行依赖库以及一些其它内容，`.dockerignore`是个纯文本文件，里边是不需要被`docker build`时加入上下文的内容，可以理解为被加入了上下文后，才能在被`docker build`时进行`COPY`，`ADD`等操作，`DockerFile`是在做`docker build`时脚本文件，Docker会按照它来逐步解释执行

`src/`目录下是所有源码以及资源文件，也是一会需要打包进Docker的部分，`src/weights/`里包含的是本仓库已经训练好的网络，`model.py`和`noiese_model.py`都是自有依赖，`noise2noise.py`是主入口，`requirements.txt`是Python项目的依赖包管理文件

![image-20200429103106941](http://q9h8o4ouf.bkt.clouddn.com/image-20200429103106941.png)

`.dockerignore`：这里先ignore所有文件，再排除掉`src/`下的

```
*
!src/
```

`DockerFile`：`FROM`指定了一个`image`作为起始镜像来构建，往后每一行（使用`\`换行算同一行）都是一层`layer`，可以理解为git的每一次commit，这块会在以后的文章中详解，`LABEL`指定了一些声明信息，`WORKDIR`指定了在`image`内部的工作空间，在这之后的每一层`layer`的操作都会在这个目录下进行，`COPY`从刚才说的上下文中，复制文件到你基于`WORKDIR`的一个相对路径，`RUN`用来在`image`中执行一些命令，`EXPOSE`指定了我们需要让`image`生成的`container`对外提供服务暴露的端口号，最后的`CMD`指定了我们启动`container`的时候执行的默认命令

```dockerfile
FROM tensorflow/tensorflow:2.1.0-gpu-py3

LABEL maintainer="Chaohang Fu <fuchaohang@migu.cn>"

WORKDIR /opt/noise2noise/

COPY src/ ./

# RUN apt-get update \
RUN apt-get install -y libsm6 libxrender1 libxext6 \
 && python -m pip install -i https://pypi.doubanio.com/simple/ --upgrade pip \
 && python -m pip install -i https://pypi.doubanio.com/simple/ -r requirements.txt

EXPOSE 5000

CMD ["python", "noise2noise.py"]

```

`noise2noise.py`：一个最简单又繁重的主入口了，包含一个`route`，`errorhandler`以及一些其它公用部分

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" noise2noise """

__author__ = 'Chaohang Fu'

import cv2
from model import get_model
from noise_model import get_noise_model
import os
import json
import base64
import numpy as np
from flask import Flask
from flask import request
from werkzeug.exceptions import HTTPException


os.environ['FLASK_APP'] = 'noise2noise.py'
val_noise_model = get_noise_model('gaussian,25,25')
model = get_model('srresnet')
model.load_weights('weights/weights.056-66.803-30.57923_gauss_clean.hdf5')
app = Flask(__name__)


@app.route('/', methods=['POST'])
def denoise_image():
    str_body = request.get_data()
    str_body = get_or_else_null(str_body)
    if str_body is None:
        raise Exception("missing request body")
    json_body = json.loads(str_body)
    base64_str_img = json_body.get('base64Image')
    base64_str_img = get_or_else_null(base64_str_img)
    if base64_str_img is None:
        raise Exception("missing param: 'base64Image' in request body")
    # {str} ''
    bytes_img = base64.b64decode(base64_str_img)
    # -> {bytes: 60501} b''
    np_array_1 = np.frombuffer(bytes_img, np.uint8)
    # -> {ndarray: (60501,)} [], dtype: uint8, size: 60501
    np_array_3 = cv2.imdecode(np_array_1, cv2.IMREAD_COLOR)
    # -> {ndarray: (720, 1154, 3)} [[[]]], dtype: uint8, size: 2492640
    np_array_3 = val_noise_model(np_array_3)
    # -> {ndarray: (720, 1154, 3)} [[[]]], dtype: uint8, size: 2492640
    np_array_4 = np.expand_dims(np_array_3, 0)
    # -> {ndarray: (1, 720, 1154, 3)} [[[[]]]], dtype: uint8, size: 2492640
    np_array_4 = model.predict(np_array_4)
    # -> {ndarray: (1, 720, 1154, 3)} [[[[]]]], dtype: float32, size: 2492640
    np_array_3 = np_array_4[0]
    # -> {ndarray: (720, 1154, 3)} [[[]]], dtype: float32, size: 2492640
    np_array_3 = np.clip(np_array_3, 0, 255)
    # -> {ndarray: (720, 1154, 3)} [[[]]], dtype: float32, size: 2492640
    np_array_3 = np_array_3.astype(np.uint8)
    # -> {ndarray: (720, 1154, 3)} [[[]]], dtype: uint8, size: 2492640
    np_array_2 = cv2.imencode('.png', np_array_3)[1]
    # -> {ndarray: (1067105, 1)} [[]], dtype: uint8, size: 1067105
    bytes_img = base64.b64encode(np_array_2.tostring())
    # -> {bytes: 1418600} b''
    base64_str_img = str(bytes_img, 'utf-8')
    # -> {str} ''
    return ResponseBody(0, 'SUCCESS', {
        'base64Image': base64_str_img
    }).__dict__


@app.errorhandler(HTTPException)
def handle_exception(e):
    return ResponseBody(-1, str(e.original_exception), None).__dict__


def get_or_else_null(s):
    if s is None:
        return None
    if type(s) == bytes:
        s = str(s, 'utf-8')
    s = s.strip()
    if s == '':
        return None
    return s


class ResponseBody:
    def __init__(self, code, msg, data):
        self.code = code
        self.msg = msg
        self.data = data


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)

```

`requirements.txt`：在`DockerFile`中需要用`pip`安装的依赖包

```
keras==2.3.1
opencv-python==4.2.0.34
flask==1.1.2

```

## 安装Docker

关于Docker的信息，相信网上有非常多的资料了，这里不做赘述，它最大的好处莫过于可以`一次构建，到处运行`，这和当年Java的口号有异曲同工之妙

地址：https://hub.docker.com/

我们这边安装的是`Docker Desktop`，这里有个坑，那就是在Windows端并不支持Docker应用使用GPU资源，不过并不影响我们制作好`image`并部署到其它Linux系统上使用GPU来计算，详见：

- https://github.com/NVIDIA/nvidia-docker

- https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support

![image-20200429110737851](http://q9h8o4ouf.bkt.clouddn.com/image-20200429110737851.png)

在安装完成Docker后，由于众所周知的网络原因，推荐使用其它镜像源来代替官网镜像源，具体步骤可以搜索到，这里不做赘述，我们用的是阿里云的镜像源加速，效果很好

![image-20200429111055147](http://q9h8o4ouf.bkt.clouddn.com/image-20200429111055147.png)

## 构建镜像

先进入我们项目根目录

```bash
cd /path/to/noise2noise
```

开始构建：这里最后一个`.`不要遗漏，表明以此作为`docker build`的上下文路径

```
docker build --network host -t noise2noise:1.0.0 .
```

在此之后你会看到`Sending build context to Docker daemon   91.7MB`，以及一步步执行`DockerFile`的记录

![image-20200429134146588](http://q9h8o4ouf.bkt.clouddn.com/image-20200429134146588.png)

在`docker build`结束后，可以用`docker image ls`查看到所有的`image`

![image-20200429134401382](http://q9h8o4ouf.bkt.clouddn.com/image-20200429134401382.png)

## 运行容器

当我们运行一个`image`之后，它会按照该镜像来启一个进程，也就是`container`

下面的`-d`表示后台运行`container`并打印`container id`，`-p`可以允许你绑定`host`和`container`的端口映射，那么末尾可以看到我们没有写命令，那么默认就是执行我们`DockerFile`中由`CMD`指定的命令了

```
docker run -d -p 5000:5000 noise2noise:1.0.0
```

还可以有以下方式：`-it`能保持`STDIN`打开并申请一个`pseudo-TTY`连接，`--rm`让容器退出时自动被清理，最后的`bash`是我们要执行的命令，那么连起来就是我们可以以`bash`方式进入容器，把它当一个Linux系统来操作

```
docker run -it --rm -p 5000:5000 noise2noise:1.0.0 bash
```

## 测试

最后我们用Postman或者其它方式发起一个HTTP请求，就可以等待结果返回了

```
POST http://localhost:5000/

{
    "base64Image": "..."
}
```

以上，我们就完成了一个noise2noise服务的开发/构建/运行了