
## 安装conda

按照官网安装指示安装即可

## 安装TensorFlow

服务器cuda版本是`10.0`，使用conda命令安装`1.13.1`版本的TensorFlow-gpu，会自动安装`cudatoolkit 10.0`
```shell
#创建python版本为3.6的虚拟环境
conda create --name pucycgan python=3.6
# 切换环境
conda activate pucycgan

# 安装tensorflow
conda install tensorflow-gpu=1.13.1

# 使用 numpy<1.17 可以避免一些警告
conda install numpy=1.16.6

```

## 编译TensorFlow operators

1. 检查tf_ops目录下每个子文件内的`tf_XXX_compile.sh`脚本，将shell脚本中关于python解释器、CUDA Runtime API、TensorFlow的位置修改成和本机对应。

2. 安装cuda toolkit

   ```shell
   sudo apt install nvidia-cuda-toolkit
   ```

3. 依次执行每个`tf_XXX_compile.sh`，或者从Docker目录复制compile.sh到tf_ops目录下执行；然后执行一下`compile_render_balls_so.sh`脚本

    ```shell
    # 复制批量编译脚本
    cp Docker/compile.sh tf_ops/
    # 执行编译
    sh tf_ops/compile.sh
    sh tf_ops/compile_render_balls_so.sh
    ```

4. 如果报错`/usr/bin/ld: cannot find "-ltensorflow_framework"`，到控制台输出的$TF_LIB的目录下，创建软连接

   ```shell
   # 注意不同版本可能是 so.2
   ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
   ```

   其他`cannot find -lXXX` 则是相关库文件(libXXX.so)没有找到，查看具体的库文件是否存在于脚本中配置的上下文中。
5. 如果没有错误，则编译成功。

## 训练模型

1. 下载 [数据文件](https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ) 放在`data/train/`下
2. 安装依赖
   ```shell
    conda install matplotlib scikit-learn 
    conda install -c conda-forge tqdm pulp
    conda install -c open3d-admin open3d
    pip install plyfile
   ```
2. 执行
    ```shell
    python pu-gan.py phase=train
    ```
3. 如果报错 `symbol: _ZTIN10tensorflow8OpKernelE` 将sh文件中的`-D_GLIBCXX_USE_CXX11_ABI=0` 改为 `1`。
   >https://github.com/google/sentencepiece/issues/293#issuecomment-510806920

4. 如果报错 `ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running` ，将`visu_utils.py中的plt.switch_backend('TkAgg')`修改为`plt.switch_backend('Agg')`

## 测试模型

1. 按照`README`的说明下载的训练好的模型包含四个文件
   1. model-100.index
   2. model-100.meta
   3. model-100.data-00000-of-00001
   4. checkpoint
   
2. 将上面四个文件放入model文件夹内，执行命令
```shell
# 可以指定模型所在路径 和 数据所在路径 和 预测结果的输出路径
python pu_gan.py --phase=test --log_dir=model/ --data_dir=data/test --out_folder=evlat/s08-u/output
```

## 评估

1. 安装CGAL

下面的方法适用于ubuntu，centos参见[CentOS 安装 CGAL](https://www.jianshu.com/p/7781a9c29f37)
```shell
apt update
apt install libcgal-dev
# 如果提示缺少qt5
apt install libcgal-qt5-dev 
```
如果提示`/root/data/earor/project/upugan/evaluation_code/evaluation.cpp:19:50: fatal error: CGAL/Polygon_mesh_processing/measure.h: No such file or directory`,大概是因为你源中的cgal版本太低,更新源或者使用清华源再更新

2. 评估

```shell
cd evaluation_code 
cmake .
make
./evaluation Icosahedron.off Icosahedron.xyz

python evaluation.py --pred evlat/s07-u/output --gt evlat/poisson
```

## 数据集

### PU-GAN

* 测试
  
   从 `https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC` 下载数据集，打开test文件夹下的27个测试模型，使用 `https://github.com/yulequan/PU-Net/tree/master/prepare_data` 中提供的程序采到8192个点，再用随机采样到2048个点。

   ```shell
   cd Poisson_sample
   cmake .
   make
   ./PdSampling 8192 ../MeshSegmentation/egea_1.off ./output/egea_1.xyz
   ```

### kitti

* 官网 <http://www.cvlibs.net/datasets/kitti/>
* 读取 <https://github.com/utiasSTARS/pykitti>

### semantic3d

* 官网 http://www.semantic3d.net

