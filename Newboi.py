#
#
#
#
#Heat Diffusion code for PHYS129L final project
#
#By Yusuf Al-Nawakhtha and Shahryar Mooraj
#
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as la
import numpy as np
import math
import matplotlib.pyplot as plt
from pynput.mouse import Button, Controller, Listener


#start and finishing time
t_start = 0
t_final = 1000

t = t_start
t_click = np.inf
x_click = 0
y_click = 0

#time-step
dt = 1

def on_press(event):
    global t
    global t_click
    global x_click
    global y_click
    global dt
    t_click = t + 2*dt
    x_click = event.xdata
    y_click = event.ydata


#def press(x, y, button, pressed):
#    global t


#with Listener(on_click=press) as listener:
#    listener.join()
    
#Setting up domain of problem

xL = -2
xR = 2
yT = 2.5
yB = -2.5

t = 0

#diffusion coefficient
DiffCo = 0.0015

#Boundary Cofficients
def Tboundary(t):

    #T_bc = min([20+(4/3)*t, 100])
    T_bc = 20
    return(T_bc)


#initial temp
T_initial = 20


#setting up co-ordinates
Nx = 80
Ny = 80
x = np.linspace(xL, xR, Nx)
y = np.linspace(yB, yT, Ny)[::-1]
dx = (xR-xL)/(Nx-1)
dy = (yT-yB)/(Ny-1)



#source term
def source(x,y,t):
    global t_click
    global x_click
    global y_click
    f = 0
    if(abs(x-x_click)<= 0.1 and abs(y-y_click) <= 0.1 and t == t_click):
        #print(y)
        f = 50
    return(f)

T_old = np.full((Nx*Ny,1), T_initial)
T_new = np.full((Nx*Ny,1), T_initial)

t = t_start

def point(x,y):
    return (y*Nx + x)

RHS = np.zeros(Nx*Ny)

values = []
nonzero_row = []
nonzero_col = []

for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        p = point(i,j)
        for k in range(5):
            nonzero_row.append(p)

        nonzero_col.append(p)
        nonzero_col.append(p-1)
        nonzero_col.append(p+1)
        nonzero_col.append(p-Nx)
        nonzero_col.append(p+Nx)
        
        values.append(1 + (2*DiffCo*dt)/(dx**2) + (2*DiffCo*dt)/(dy**2))
        values.append(-DiffCo*(dt)/(dx**2))
        values.append(-DiffCo*(dt)/(dx**2))
        values.append(-DiffCo*(dt)/(dy**2))
        values.append(-DiffCo*(dt)/(dy**2))


for i in range(Nx):
    for j in range(Ny):
        if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
            p = point(i,j)
            nonzero_row.append(p)
            nonzero_col.append(p)
            values.append(1)

A = csc_matrix((values, (nonzero_row, nonzero_col)), shape = (Nx*Ny,Nx*Ny))
fig, ax = plt.subplots(1,1)

cid = fig.canvas.mpl_connect('button_press_event', on_press)

while t < t_final:
    for i in range(1,Nx-1):
        for j in range(1, Ny-1):
            RHS[point(i,j)] = T_old[point(i,j)] + dt*source(x[i],y[j],t+dt)

            
    for i in range(Nx):
        for j in range(Ny):
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                RHS[point(i,j)] = Tboundary(t)

    #Solving system equations

    T_new = la.spsolve(A, RHS)

    T_matrix = np.full((Nx,Ny), 0.)
    X,Y = np.meshgrid(x,y)
    for i in range(Nx):
        for j in range(Ny):
            T_matrix[i][j] = T_old[point(i,j)]
    T_mat = np.transpose(T_matrix)
    graph = plt.imshow(T_mat, cmap='jet', extent = [xL, xR, yB, yT])
    if t == t_start:
        bar = fig.colorbar(graph, ax=ax)
    else:
        #bar = plt.colorbar(graph)
        if(T_mat.max()-T_mat.min() > 5):
            graph.autoscale()
            bar.update_bruteforce(graph)
        else:
            plt.clim(0,100)
    plt.pause(0.0001)
    T_old = T_new  
    t = t+dt
    
    
plt.show()
    


