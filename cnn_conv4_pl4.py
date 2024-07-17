import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取序列文件
def read_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        sequences = []
        sequence_ids = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                sequence_ids.append(line[1:])
                sequences.append('')
            else:
                sequences[-1] += line
    return sequence_ids, sequences

#生成特征字典
def read_feature(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    feature_dict = {}
    global feature_length
    feature_length = None

    for _, row in df.iterrows():
        char = row['char']
        features = row[1:].tolist()

        #判断每个字符对应的特征维度是否相同
        if feature_length is None:
            feature_length = len(features)
        elif len(features) != feature_length:
            raise ValueError(f"Character {char} has a different feature length: {len(features)}")
        
        feature_dict[char] = features
    
    return feature_dict

# 读取标签文件
def read_labels(labels_file):
    labels = []
    labels_df = pd.read_csv(labels_file, sep='\t')

    for _, row in labels_df.iterrows():
        labels.append(row['label'])
    labels_array = np.array(labels)
    num_classes = len(set(labels))
    return labels_array, num_classes

#生成序列每个位点的特征张量
def generate_tensor(sequences, feature_dict):
    
    #判断序列是否为比对序列，即序列是否等长
    seq_lengths = [len(seq) for seq in sequences]
    if len(set(seq_lengths)) == 1:
        global equal_length
        equal_length = seq_lengths[0]
        print(f"All sequences are of equal length: {equal_length}")
    else:
        raise ValueError("Sequences are not of equal length.")

    input_features = len(next(iter(feature_dict.values())))
    tensor = np.zeros((len(sequences), equal_length, input_features))  #3-D tensor的维度分别为序列序号，位点，特征

    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            if char in feature_dict:
                tensor[i, j, :] = feature_dict[char]
            else:
                raise ValueError(f"Character {char} not found in feature dictionary.")
    
    return tensor

#定义CNN模型
class CNN(nn.Module):
    def __init__(self, feature_length, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_length, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * (equal_length // 16), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#定义训练函数
def train(model, train_loader, cirterion, optimizer, num_epochs, device, patience=8):
    
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移至GPU设备
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1)) #调整输入的张量形状为(batch_size, feature_length, seq_length)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 准确性评估
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.permute(0, 2, 1))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        train_accuracies.append(avg_val_accuracy)  # 显示训练准确率
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # 检查模型是否有改进
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最优模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping triggered.')
                break
    
    # 绘制损失和准确率图
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

#定义评估函数
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移至GPU设备
            outputs = model(inputs.permute(0, 2, 1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 定义绘图函数
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python all.py <alignment_file_train> <alignment_file_validation> <feature_file> <label_file_train> <label_file_validation>")
        sys.exit(1)

#读取序列文件、特征文件以标签文件，并将序列转化为3D的特征张量
sequence_ids_train, sequences_train = read_fasta(sys.argv[1])
sequence_ids_val, sequences_val = read_fasta(sys.argv[2])

feature_dict = read_feature(sys.argv[3])

tensor_data_train = torch.tensor(generate_tensor(sequences_train, feature_dict))
tensor_data_val = torch.tensor(generate_tensor(sequences_val, feature_dict))

label_array_train, num_classes_train = read_labels(sys.argv[4])
label_array_val, num_classes_val = read_labels(sys.argv[5])
num_classes = None
if num_classes_train == num_classes_val:
    num_classes = num_classes_train
else:
    raise ValueError("The numbers of classes in training set and testing set are different.")

labels_data_train = torch.tensor(label_array_train)
labels_data_val = torch.tensor(label_array_val)

#labels_data_train = torch.tensor(read_labels(sys.argv[4]))
#labels_data_val = torch.tensor(read_labels(sys.argv[5]))

tensor_data_train = tensor_data_train.float()
tensor_data_val = tensor_data_val.float()
print(tensor_data_train.dtype, tensor_data_val.dtype)

#CNN模型超参数设定
learning_rate = 0.001   #学习率
batch_size = 16 #批大小
num_epochs = 100 #训练轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU设备
print(f"Using device: {device}")

#加载输入数据
train_dataset = TensorDataset(tensor_data_train, labels_data_train)
val_dataset = TensorDataset(tensor_data_val, labels_data_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#初始化模型，损失函数及优化器，训练、评估模型
model = CNN(feature_length, num_classes).to(device) # 将数据移至GPU设备
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_loader, criterion, optimizer, num_epochs, device)  # 传递device参数
model.load_state_dict(torch.load('best_model.pth'))
accuracy = evaluate(model, val_loader, device)  # 传递device参数

print(f'Validation Accuracy: {accuracy:.2f}%')


#将特征张量打印至文本文件
#with open('tensor_output_list.txt', 'w') as f:
#    f.write(f"Shape: {tensor_data.shape}\n")
#    for i in range(tensor_data.shape[0]):
#        f.write(f">{sequence_ids[i]}\n")
#        for j in range(tensor.shape_data[1]):
#            f.write(f"site {j + 1}: {tensor_data[i, j].tolist()}\n")
#        f.write("\n")


