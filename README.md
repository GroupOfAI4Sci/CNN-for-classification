# 注意点：
1. 该容器内脚本和文件用于测试基于CNN的分类问题。
2. 后缀为fasta的文件为多序列比对文件，不同文件名对应包含的毒株类型。多序列比对文件对应的label信息在对应的包含label的文件中。feature文件在feature_files文件夹中。
3. 脚本使用命令行: python cnn_conv4_pl4.py <alignment_file_train> <alignment_file_validation> <feature_file> <label_file_train> <label_file_validation>
4. cnn_conv4_pl4.py设计的CNNC架构包含4层卷积层、4层最大池化层，两层全连接层，激活函数为Relu，损失函数为交叉熵函数。
5. 模型的架构可根据自己的需求重新设计，每一层卷积层的超参数见脚本，可更改。batch_size默认16，运行次数epoch默认为100，均可修改。
6. 模型设计采用Early Stopping机制防止过拟合，可通过修改脚本中的patience参数来调整模型性能。
7. 模型没有加入dropout层，有需要可在定义forward层时加入该层。
8. 模型设计采用CUDA训练模型，当设备没有GPU时会改用CPU。
9. 模型训练会给出训练损失和验证损失的动态变化曲线便于调整模型。
10. 未完待续
