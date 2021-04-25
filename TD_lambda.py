import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
plt.style.use('ggplot')

# System Parameters
p1, p2, p3, fd1, fd2 = 3.473, 0.196, 0.242, 5.3, 1.1

# Define NN
'''
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,1)
        
    def forward(self,x):
        x1 = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(x1))
'''

class Net():
    def __init__(self):
        self.w1 = torch.rand(2,10,requires_grad=True)
        self.w2 = torch.rand(10,2,requires_grad=True)
        
    def forward(self,x):
        x1 = torch.sigmoid(torch.matmul(x,self.w1))
        return torch.sigmoid(torch.matmul(x1,self.w2))

def policy(x):
    if np.abs(x[0]+x[1])>=.1:
        u = [[-3],[-2]]
    else:
        u=[[0],[0]]
    return u

def distrubance(t):
    return [[0.5*np.cos(0.5*t)],[np.sin(t)]]

def model(x,t,policy,disturbance):
    c1=np.cos(x[0])
    c2=np.cos(x[1])
    s1=np.sin(x[0])
    s2=np.sin(x[1])
    
    fs1=1
    fs2 =1
    u=policy(x)
    M=np.array([[p1+2*p2*c2,p2+p3*c2],
                [p2+p3*c2,p2]])
    C=np.array([[-p3*s2*x[3],-p3*s2*(x[2]+x[3])],
                [p3*s2*x[2],0]])
    D=np.array([[fd1,0],
                [0,fd2]])
    S = np.array([[fs1,0],
                  [0,fs2]])
        
    x3x4dot = np.linalg.inv(M)@(-(C+D)@[[x[2]],[x[3]]]-S@distrubance(t)+u)
 
    return [x[2],x[3],x3x4dot[0][0],x3x4dot[1][0]]

def Q_approximation(net_x,u):
    
    
    return torch.norm(u) + torch.dot(u,net_x) 

def Temperal_difference(x,k,Q_net):
    # Define desired trajectory
    x_d = torch.tensor([math.cos(0.5*k),2*math.cos(k)])
    
    # Cost function
    c = x - x_d
    
    # Pass x through the net
    x_net = Q_net.forward(x)
    print(x_net)
    
    # Define loss function for nn
    Loss = 0.5*Q_approximation(x_net,-x_net)
    print(Loss)
    
    Loss.backward()
    print(Q_net.w1.grad)
    return Q_net.w1.grad

'''
xinit=[0,0,0,0]
t=np.linspace(0,100,1000)


x = odeint(model,xinit,t,args=(policy,distrubance,))

fig, axs =plt.subplots(2,1)
axs[0].plot(t,x[:,0],t,x[:,1])
axs[0].legend(['$q_1$','$q_2$'])
axs[1].plot(t,x[:,2],t,x[:,3])
axs[1].set_xlabel('$t$')
axs[1].legend(['$\dot{q}_1$','$\dot{q}_2$'])
'''

# random seed
torch.manual_seed(200)
net = Net()
x = torch.tensor([1.0,2.0])
Temperal_difference(x,1.0,net)

