import torch
import numpy as np

# -----------------------------------------------------------------
# Tensor            
# -----------------------------------------------------------------

# initialize a tensor for given data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# from anothor tensor
x_ones = torch.ones_like(x_data)
x_rand = troch.rand_like(x_data)

# from constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# attriutes
tensor = torch.rand(2,3)
tensor_shape = tensor.shape
tensor_dtype = tensor.dtype
tensor_device = tensor.device

# move the tensor to the GPU
if tensor.cuda.is_available():
    tensor = tensor.to("cuda")

# standard numpy-like indexing and slicing operations
tensor = torch.ones(4,4)
first_row = tesnor[0]
first_col = tensor[:,0]
last_col = tensor[...,-1] # use ellipsis to make it clean
tensor[:,1] = 0 # this will make the second column 0

# join tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1) 
# dim = 0 -> stack on rows, while dim = 1 -> stack on columns

# arithmetic operations
# -> matrix multiplication
y1 = tensor @ tensor.T # @ sign represents matrix multiplication
y2 = torch.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

# -> element-wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

# -> single-element tensors
agg = tensor.sum()
agg_item = agg.item() # convert it to a python numerical value, type(agg_item) is 'float'

# -> inplace operations, underscore
tensor.add_(5)

# -----------------------------------------------------------------
# datasets and dataloaders        
# -----------------------------------------------------------------

from torch.util.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False,
    transform = ToTensor()
)

# prepare data
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# iterate through the DataLoader
train_features, train_labels = next(iter(train_dataloader)) # returns an iterator for the given argument
feature_batch_shape = train_features.size()
label_batch_shape = train_labels.size()

img = train_features[0].squeeze() # squeeze before plot
label = train_labels[0]
plt.imshow(img, cmap = "gray")
plt.show()


# -----------------------------------------------------------------
# transforms  
# -----------------------------------------------------------------

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor(),
    # normalized tensors, and the labels as ONE-HOT encoded tensors
    # scatter_(dim, index, src) 
    # https://blog.csdn.net/weixin_45547563/article/details/105311543?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-105311543-blog-105751765.235%5Ev38%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-105311543-blog-105751765.235%5Ev38%5Epc_relevant_default_base&utm_relevant_index=2
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), values=1))
)

# -----------------------------------------------------------------
# build the neural network
# -----------------------------------------------------------------
import os
# get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# define the Class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )

z = torch.matmul(x, w) + b
print(z.requires_grad) # True
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad) # False

# -----------------------------------------------------------------
# optimizing model parameters
# -----------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # set the model to evalution mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluatin the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

# END OF CODE