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

    T_bc = min([20+(4/3)*t, 100])
    return(T_bc)

#source term
def source(x,y,t):
    f = 0
    return(f)

#initial temp
T_initial = 20

#start and finishing time
t_start = 0
t_final = 1000


#setting up co-ordinates
Nx = 80
Ny = 100
x = np.linspace(xL, xR, Nx)
y = np.linspace(yB, yT, Ny)
dx = (xR-xL)/(Nx-1)
dy = (yT-yB)/(Ny-1)

#time-step

dt = 5

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
while t < t_final:
    for i in range(1,Nx-1):
        for j in range(1, Ny-1):
            RHS[point(i,j)] = T_old[point(i,j)] + dt*source(x[i],y[i],t+dt)

            
    for i in range(Nx):
        for j in range(Ny):
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                RHS[point(i,j)] = Tboundary(t)

    #Solving system equations

    T_new = la.spsolve(A, RHS)

    T_matrix = np.full((Nx,Ny), 0)
    X,Y = np.meshgrid(x,y)
    for i in range(Nx):
        for j in range(Ny):
            T_matrix[i][j] = T_old[point(i,j)]
    T_mat = np.transpose(T_matrix)

    graph = ax.imshow(T_mat, cmap='jet')
    if t == t_start:
        bar = fig.colorbar(graph, ax=ax)
    else:
        #bar = plt.colorbar(graph)
        graph.autoscale()
        bar.update_bruteforce(graph)
    plt.pause(0.001)
    T_old = T_new
    t = t+dt

plt.show()
    


