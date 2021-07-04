import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda,Compose
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    #这里并没有使用transforms.Compose说明我们这里并不需要进行多个图片操作。
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

batch_size = 64
#将Dataset传入DataLoader中。
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#z注意如果是自己的数据需要封装，那么要先写一个类，该类继承Dataset，然后在该类中要
#重写__len__,__getitem__项。并且在该类的__init__方法中，除了self，还有文件的路径。
#由torchvision中获得的数据已经是Dataset格式，不需要经过Dataset,直接传入DataLoader中。
for X, y in test_dataloader:
    print("shape of X [N, C, H, W]:", X.shape)
    print("Shape of y:", y.shape, y.dtype)
    break
#这里循环中加了break,所以只需要打印一批的数据的信息。

#Creating Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print('using {} device'.format(device))

#define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#接下来是优化模型参数
#定义损失函数
loss_fn = nn.CrossEntropyLoss()#你可以这样理解，nn.xx，是一个类，torch.nn.functional.xx是函数。
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
#w为了训练我们的模型，我们需要损失函数和优化器
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #pred.argmax(1)在指定维度上求最大值。
            #指定类型改变。
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: /n Accuracy:{(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}\n")


#训练过程是在好几次迭代情况下实施的。
if __name__ == "__main__":
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------")
        train(train_dataloader,model,loss_fn,optimizer)
        test(test_dataloader,model,loss_fn)
    print("Done!!")