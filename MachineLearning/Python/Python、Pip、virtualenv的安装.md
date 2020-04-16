#  Python、Pip、virtualenv的安装

## Python、Pip的安装

我们可以在Python的官网找到下载链接

https://www.python.org/downloads/

选择好自己需要安装的版本后，会进入该版本的不同形式的安装包下载页面

如果是Windows 64位系统用户，一般选择Windows x86-64 executable installer即可

安装完成后，Pip也会被作为默认提供的包管理内置在其中

## virtualenv的安装

先说下为什么要安装virtualenv

我们在开发Python项目的时候，所有第三方包都会被Pip安装到Python的site-packages目录下

如果我们不同Python项目使用的依赖包版本号不一样，就会导致一些问题

为了解决它，virtualenv允许每个Python项目都拥有自己独立的运行环境

安装virtualenv：

```
pip install virtualenv
```

创建venv：

```
virtualenv --system-site-packages -p python3 ./venv
```

激活venv：

```
.\venv\Scripts\activate
```

之后我们会看到命令行的开头出现了`(venv)`字样，说明我们已在虚拟环境中

那么，在当前目录下，我们就可以相互独立地执行Python脚本了

退出venv：

```
deactivate
```