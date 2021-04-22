from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# System Parameters
p1, p2, p3, fd1, fd2 = 3.473, 0.196, 0.242, 5.3, 1.1

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

xinit=[0,0,0,0]
t=np.linspace(0,100,1000)


x = odeint(model,xinit,t,args=(policy,distrubance,))

fig, axs =plt.subplots(2,1)
axs[0].plot(t,x[:,0],t,x[:,1])
axs[0].legend(['$q_1$','$q_2$'])
axs[1].plot(t,x[:,2],t,x[:,3])
axs[1].set_xlabel('$t$')
axs[1].legend(['$\dot{q}_1$','$\dot{q}_2$'])


