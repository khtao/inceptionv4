**安装依赖库**

在当前目录下，在终端执行

```bash
pip install -r requirements.txt
```

默认数据集文件夹结构：

```
./dataset/
    train/
        neg/
        pos/
    test/
        neg/
        pos/
```

在**utils/config.py**中将**data_dir**改为自己数据集的路径