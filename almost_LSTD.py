from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# System Parameters
p1, p2, p3, fd1, fd2 = 3.473, 0.196, 0.242, 5.3, 1.1

def cost(x,u,reference):
    S = np.identity(2)
    R = np.identity(2)
    error = np.array([reference[0]-x[0],reference[1]-x[1]])
    return error.T.dot(S).dot(error)+u.T.dot(R).dot(u)

def basis(x,u):
    return np.array([x[0]*x[0],
            x[1]*x[1],
            x[2]*x[2],
            x[3]*x[3],
            u[0]*u[0],
            u[1]*u[1],
            x[0]*x[1],
            x[0]*x[2],
            x[0]*x[3],
            x[0]*u[0],
            x[0]*u[1],
            x[1]*x[2],
            x[1]*x[3],
            x[1]*u[0],
            x[1]*u[1],
            x[2]*x[3],
            x[2]*u[0],
            x[2]*u[1],
            1])
    
def policy(x):
    if np.abs(x[0]+x[1])>=.1:
        u = np.array([[-3],[-2]])
    else:
        u=np.array([[0],[0]])
    return u

def distrubance(t):
    return [[0.5*np.cos(0.5*t)],[np.sin(t)]]

def compute_temporal_difference(x, u, cost, theta):
    TD = theta.T
    
def model(x,t,policy,disturbance,reference):
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

    
    
xinit=[0,0,0,0]
t=np.linspace(0,100,1000)

reference = np.array([[0],[0]])
x = odeint(model,xinit,t,args=(policy,distrubance,reference,))

fig, axs =plt.subplots(2,1)
axs[0].plot(t,x[:,0],t,x[:,1])
axs[0].legend(['$q_1$','$q_2$'])
axs[1].plot(t,x[:,2],t,x[:,3])
axs[1].set_xlabel('$t$')
axs[1].legend(['$\dot{q}_1$','$\dot{q}_2$'])

#print(x[0,0:2])
costs = []
u = []
#print(x.shape[0])
for i in range(x.shape[0]):
    u.append(policy(x[i,:]))
    #print(x[i,:])
    costs.append(cost(x[i,:], policy(x[i,:]), reference))
    
costs=np.array(costs).reshape(x.shape[0],)

fig = plt.subplots()
plt.plot(costs)
plt.xlabel('$t$')
plt.ylabel('$c(x_t,u_t)$')
plt.title('Cost Over Time')

# Compute temporal difference sequence
sigma = 1

u = np.array(u).reshape([x.shape[0],2])


theta = np.random.rand(19)
TD_kplus = np.zeros([x.shape[0]-1])
upsilon_kplus = np.zeros([x.shape[0]-1,19])
for k in range(x.shape[0]-1):
    TD_kplus[k] = -theta.dot(basis(x[k,:], u[k,:]))+cost(x[k,:], u[k,:], reference)+theta.dot(basis(x[k+1,:], u[k+1,:]))
    upsilon_kplus[k,:] = basis(x[k,:], u[k,:])-basis(x[k+1,:], u[k+1,:])
gamma_k = costs


print(upsilon_kplus)
fig = plt.subplots()
plt.plot(TD_kplus)
plt.xlabel('$t$')
plt.ylabel('$\mathcal{D}_k(\theta)$')
plt.title('Temporal Difference Sequence')
#print(basis(x[1,:], u[1,:]).shape)
#print(np.random.rand(6))

