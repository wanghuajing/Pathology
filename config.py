import pandas as pd

lr = 0.01

num_works = 0
# 数据表
table = pd.read_csv("cbis.csv")
# 数据路径
path = "C:/Users/whj/OneDrive/文件/cbis_ddsm_original_compressed/Calc-test/"
# 数据集的类别
num_class = 2

# 训练时batch的大小
batch_size = 8

val_percent = 0.1
# 训练轮数
epoch = 25

##预训练模型的存放位置
# 下载地址：https://download.pytorch.org/models/resnet50-19c8e357.pth
save_path=''

##训练完成，权重文件的保存路径,默认保存在trained_models下
TRAINED_MODEL = 'trained_models/vehicle-10_record.pth'
