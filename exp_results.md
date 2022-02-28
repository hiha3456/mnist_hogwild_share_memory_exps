## Configuration
CPU: 32 cores
Memory: 50 GB
GPU: A100

## Structure of ConvNet
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

## Comparison of memory usage
Unit: GB

<table>
	<tr>
	    <th>Method of create sub-process</th>
	    <th>if share memory</th>
	    <th>device of model</th>  
        <th>1 subprocess</th>
        <th>2 subprocess</th>
        <th>4 subprocess</th>
        <th>8 subprocess</th>
        <th>16 subprocess</th>
	</tr>
	<tr>
	    <td rowspan="4">spawn</td>
	    <td rowspan="2">Yes</td>
        <td>CPU</td>	
        <td>2.07</td>	
        <td>3.05</td>	
        <td>5.52</td>	
        <td>9.96</td>	
        <td>18.64</td>
	</tr>
	<tr>
	    <td>GPU</td>	
        <td>6.69</td>	
        <td>10.80</td>	
        <td>18.90</td>	
        <td>35.40</td>	
        <td>68.27</td>
	</tr>
	<tr>
	    <td rowspan="2">No</td>
        <td>CPU</td>	
        <td>1.89</td>	
        <td>3.03</td>	
        <td>5.16</td>	
        <td>9.82</td>	
        <td>18.73</td>
	</tr>
	<tr>
	    <td>GPU</td>	
        <td>5</td>	
        <td>9.14</td>	
        <td>17.44</td>	
        <td>33.89</td>	
        <td>67.17</td>
	</tr>
	<tr>
        <td rowspan="4">Fork</td>
	    <td rowspan="2">Yes</td>
        <td>CPU</td>	
        <td>1.53</td>	
        <td>1.92</td>	
        <td>3.11</td>	
        <td>4.89</td>	
        <td>10</td>
	</tr>
	<tr>
	    <td>GPU</td>	
        <td>-</td>	
        <td>-</td>	
        <td>-</td>	
        <td>-</td>	
        <td>-</td>
	</tr>
	<tr>
	    <td rowspan="2">No</td>
        <td>CPU</td>	
        <td>1.45</td>	
        <td>2</td>	
        <td>2.73</td>	
        <td>5.21</td>	
        <td>9.83</td>
	</tr>
	<tr>
	    <td>GPU</td>	
        <td>5.91</td>	
        <td>11.04</td>	
        <td>18.08</td>	
        <td>36.76</td>	
        <td>72.71</td>
	</tr>
</table>

## Comparison of GPU memory usage

<table>
	<tr>
	    <th>Method of create sub-process</th>
	    <th>if share memory</th>  
        <th>1 subprocess</th>
        <th>2 subprocess</th>
        <th>4 subprocess</th>
        <th>8 subprocess</th>
        <th>16 subprocess</th>
	</tr>
    <tr>
        <td rowspan="2">spawn</td>
	    <td>Yes</td>	
        <td>3.50</td>	
        <td>5.80</td>	
        <td>10.42</td>	
        <td>19.64</td>	
        <td>38.09</td>
    </tr>
    <tr>
	    <td>No</td>	
        <td>2.06</td>	
        <td>4.12</td>	
        <td>8.24</td>	
        <td>16.48</td>	
        <td>32.96</td>
    </tr>
        <td>Fork</td>
	    <td rowspan="2">No</td>	
        <td>2.06</td>	
        <td>4.12</td>	
        <td>8.24</td>	
        <td>16.48</td>	
        <td>32.96</td>
</table>

## Comparison of Volatile GPU-Util
<table>
	<tr>
	    <th>Method of create sub-process</th>
	    <th>if share memory</th>  
        <th>1 subprocess</th>
        <th>2 subprocess</th>
        <th>4 subprocess</th>
        <th>8 subprocess</th>
        <th>16 subprocess</th>
	</tr>
    <tr>
        <td rowspan="2">spawn</td>
	    <td>Yes</td>	
        <td>41%</td>	
        <td>82%</td>	
        <td>99%</td>	
        <td>99%</td>	
        <td>99%</td>
    </tr>
    <tr>
	    <td>No</td>	
        <td>39%</td>	
        <td>88%</td>	
        <td>99%</td>	
        <td>99%</td>	
        <td>99%</td>
    </tr>
        <td>Fork</td>
	    <td rowspan="2">No</td>	
        <td>39%</td>	
        <td>89%</td>	
        <td>99%</td>	
        <td>99%</td>	
        <td>99%</td>
</table>
