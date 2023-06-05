# %%
#======Libraries======
import numpy as np
import torch as pt

# %%
#======Sets the device======
if pt.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Using {device} device')

# Five parts to the machine learning process:
# 1) Gather and read data
# 2) Cleaning the data
# 3) Creating a machine learning model (training)
# 4) Evaluation (testing)

# %%
#======PART 1======
x = np.random.uniform(-1, 1, size = (100, 1))
y = 4.3 * x + 5

# %%
#======PART 2======
x_train = x[:int(len(x)/(5/4))]
x_val = x[int(len(x)/(5/4)):]

y_train = y[:int(len(y)/(5/4))]
y_val = y[int(len(y)/(5/4)):]

# %%
#======PART 3======
#Hyperparameters
lr = 0.001 #Learning rate
n_epochs = 1000 #Number of iterations

# pt.nn.Linear(a, b) - PyTorch linear neural network
# a - number of inputs (features)
# b - number of outputs (labels)
model = pt.nn.Sequential(pt.nn.Linear(1, 1)).to(device) #Creates a model and sends it to the device
loss_fn = pt.nn.MSELoss() #Defines a MSE loss function
optimizer = pt.optim.SGD(model.parameters(), lr) #Defines a stochastic gradient descent (SGD) optimizer to update the parameters

#Training
model.train() #Sets model to training mode

#Send the training data to PyTorch
x_train = pt.from_numpy(x_train).float().to(device)
y_train = pt.from_numpy(y_train).float().to(device)

for epochs in range(n_epochs):
    yPred = model(x_train)
    loss = loss_fn(y_train, yPred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#Weight - 4.3
#Bias - 5
print(model.state_dict())

# %%
#======PART 4======
model.eval() #Sets model to evaluation mode
x_val = pt.from_numpy(x_val).float().to(device)
yPred = model(x_val).cpu().detach().numpy() #Detaches the tensor from the GPU and converts it to a numpy array
comparison = abs(yPred - y_val).flatten()

#=====
a=20
x = np.array([a])
x = pt.from_numpy(x).float().to(device)
yPred = model(x).cpu().detach().numpy()
print(yPred)
# %%
