# Configuration
CPU: 32 cores
Memory: 50 GB
GPU: A100

# Structure of ConvNet
```Python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1000, kernel_size=5)
        self.conv2 = nn.Conv2d(1000, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

# Comparison of memory usage
Unit: GB
| Method of create sub-process | if share memory | device of model | 1 subprocess | 2 subprocess | 4 subprocess | 8 subprocess | 16 subprocess | 
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | 
| Spawn  | 
| Fork  | 
