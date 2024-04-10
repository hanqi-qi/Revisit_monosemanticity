import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.n_task = 2
        self.u = torch.rand(size=(2, 1),requires_grad=True,device="cuda")
        self.mse_loss = nn.MSELoss()
        self.v = torch.rand(size=(10, 1),requires_grad=True,device="cuda")
        self.shareM = torch.rand(size=(2,10),requires_grad=True,device="cuda")
                  # Generate random label y with shape (10, 3)
        nn.init.xavier_uniform_(self.u)            # Initialize u using Xavier initialization
        nn.init.xavier_uniform_(self.v)            # Initialize v using Xavier initialization

    def forward(self,y):
        w = torch.matmul(self.u, self.v.T) 
        new_icv = self.shareM + w
        # new_y = w+y.cuda()# Compute w as the matrix product of u and v
        loss = self.mse_loss(new_icv, y)            # Compute the Mean Squared Error (MSE) loss between w and y
        return new_icv,loss

# Instantiate the network
net = MyNetwork()

# Define Mean Squared Error (MSE) loss function
# mse_loss = nn.MSELoss()

# Forward pass
y = torch.randn(2, 10).cuda()
optim = torch.optim.Adam([net.u,net.v,net.shareM],lr=1e-1)
for i in range(10):
    new_y,loss = net(y)
    optim.zero_grad()
    loss.backward(retain_graph=True)
    y_value = new_y.detach().cpu().numpy()
    y = torch.tensor(y_value).cuda()
    optim.step()
    print(loss,net.u,net.v)
# Backward pass

# Now you can access gradients for u and v as follows:
u_gradient = net.u.grad
v_gradient = net.v.grad
