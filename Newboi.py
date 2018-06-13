#
#
#
#
#Heat Diffusion code for PHYS129L final project
#
#By Yusuf Al-Nawakhtha and Shahryar Mooraj
#
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la
import numpy as np
import math
import matplotlib.pyplot as plt
import threading
import sys
#from pynput.mouse import Button, Controller, Listener

#used to calculate indicies for arrays and matricies
def point(x,y):
    return (y*gridX + x)

#function that asks the user for a float between minimum and maximum
def floatInput(prompt,minimum, maximum):
    #keeps asking for input until it receives a valid input
    while(True):
        result = input(prompt + ":\n") #ask for input

        #check that the input is a float
        try:
            result = float(result)

        except:
            print("Invalid input. Please provide a real number:")

        #if the input is valid, set invalidInput to False
        else:
            if(result > minimum and result < maximum):
                return result
            else:
                print("Invalid input. Please provide a number between " + str(minimum) + " and " + str(maximum) + ":")

#function that asks the user for an integer between minimum and maximum
def intInput(prompt,minimum, maximum):
    #keeps asking for input until it receives a valid input
    while(True):
        result = input(prompt + ":\n") #ask for input

        #check that the input is an int
        try:
            result = int(result)

        except:
            print("Invalid input. Please provide an integer:")

        #if the input is valid, set invalidInput to False
        else:
            if(result > minimum and result < maximum):
                return result
            else:
                print("Invalid input. Please provide an integer between " + str(minimum) + " and " + str(maximum) + ":")

#source term
def source(x,y,t, temperature):
    f = 0

    #check the location and time of the click
    if(abs(x-x_click)<= 0.025*width and abs(y-y_click) <= 0.025*length and t == t_click):
        f = temperature
    for i in range(len(perm_heat_x)):
        if(abs(x-perm_heat_x[i])<= 0.025*width and abs(y-perm_heat_y[i]) <= 0.025*length):
            f += perm_heat_value[i]
    return(f)

def sourceInput():
    global T_source
    while(True):
        T_source = floatInput("Please input a temperature between 0 and 200 Kelvin to add when pressing on the plot", 0, 200)

#Boundary Cofficients
def Tboundary(t, m):
    T_boundary = T_initial + m*t
    if(min_temp > t):
        T_boundary = min_temp
    elif(max_temp < t):
        T_boundary = max_temp
    return(T_boundary)

def on_press(event):
    global t_click
    global x_click
    global y_click
    global click_type

    #the click time is set to be two iterations after the current time to insure that it is not incremented before the source function is called
    if(event.button == 1):
        t_click = t + 2*dt
        x_click = event.xdata
        y_click = event.ydata

    elif(event.button == 3):
        perm_heat_x.append(event.xdata)
        perm_heat_y.append(event.ydata)
        perm_heat_value.append(T_source)

#start time
t_start = 0
t = t_start

#the time the click occured
t_click = np.inf

#the x position of the click
x_click = 0

#the y position of the click
y_click = 0

#click_type is changed to 1 or 3 if a left or right click occurs respectively
click_type = 0

#time-step in seconds
dt = floatInput("Please input the desired time step at each iteration in seconds", 0, np.inf)

perm_heat_x = []
perm_heat_y = []
perm_heat_value = []



#Taking user input for the width and length of the plate
width = floatInput("Please input the desired width of the plate in meters", 0, np.inf)
length = floatInput("Please input the desired length of the plate in meters", 0, np.inf)

xL = -(width/2)
xR = (width/2)
yT = (length/2)
yB = -(length/2)

#diffusion coefficient
thermalDiff = floatInput("Please input the desired thermal diffusivity constant in (m^2)/s", 0, 0.2)

max_temp = 200
min_temp = 0

#initial temp
T_initial = floatInput("Please input the initial boundary temperature in Kelvin, between 0 and 200 Kelvin", 0, 200)

#rate of change of the boundary temperature
rate = floatInput("Please input the rate of change of the boundary temperature in Kelvin/s (0 for constant temperature)", -np.inf, np.inf)

if(rate > 0):
    max_temp = floatInput("Please input the max temperature for the boundary in Kelvin. The value must be greater than the initial temperature of the boundary and less than 200 Kelvin", T_initial, 200)

elif(rate < 0):
    min_temp = floatInput("Please input the minimum temperature for the boundary in Kelvin. The value must be less than the initial temperature of the boundary and greater than 0 Kelvin", 0, T_initial)


#initial temperature of the plate
T_plate = floatInput("Please input the initial plate temperature in Kelvin, between 0 and 200 Kelvin", 0, 200)

#setting up the number of points
gridX = intInput("Please input the desired number of points along horizontal component of the grid", 0, np.inf)
gridY = intInput("Please input the desired number of points along vertical component of the grid", 0, np.inf)

#create an array of x-points
x = np.linspace(xL, xR, gridX)

#create an array of y-points. Reverse it because the plotting method used lists the y-axis in reverse
y = np.linspace(yB, yT, gridY)[::-1]

#find the distances between the points
dx = (xR-xL)/(gridX-1)
dy = (yT-yB)/(gridY-1)

T_source = T_plate

#set up the vectors for the old and new temperatures
T_old = np.full((gridX*gridY,1), T_plate)
T_new = np.full((gridX*gridY,1), T_plate)

#the right hand side of the equation
RHS = np.zeros(gridX*gridY)

thr = threading.Thread(target = sourceInput)
thr.daemon = True
thr.start()

#parameter vectors for creating the sparse matrix
values = []
nonzero_row = []
nonzero_col = []

#loop over the internal points of the 2D grid
for i in range(1, gridX-1):
    for j in range(1, gridY-1):
        #coorelations with the sparse matrix: the point itself is at (p,p). The point to its left and right are at (p,p-1) and (p,p+1) respectively. The point to its top and bottom are at (p,p+gridX) and (p,p-gridY) respectively.
        p = point(i,j)

        #append the non-zero coordinates and the values at the indicies to the appropriate vectors.
        for k in range(5):
            nonzero_row.append(p)

        nonzero_col.append(p)
        nonzero_col.append(p-1)
        nonzero_col.append(p+1)
        nonzero_col.append(p-gridX)
        nonzero_col.append(p+gridX)
        
        values.append(1 + (2*thermalDiff*dt)/(dx**2) + (2*thermalDiff*dt)/(dy**2))
        values.append(-thermalDiff*(dt)/(dx**2))
        values.append(-thermalDiff*(dt)/(dx**2))
        values.append(-thermalDiff*(dt)/(dy**2))
        values.append(-thermalDiff*(dt)/(dy**2))

for i in range(gridX):
    for j in range(gridY):
        if i == 0 or i == gridX - 1 or j == 0 or j == gridY - 1:
            #append 1 to the matrix indicies coorelated with the boundary points and nothing else in that row, so the boundary points would remain constant
            p = point(i,j)
            nonzero_row.append(p)
            nonzero_col.append(p)
            values.append(1)

            #set the boundary temperatures
            T_old[p] = T_initial
            T_new[p] = T_initial

#create the sparse matrix
A = csc_matrix((values, (nonzero_row, nonzero_col)), shape = (gridX*gridY,gridX*gridY))

fig, ax = plt.subplots(1,1)

cid = fig.canvas.mpl_connect('button_press_event', on_press)

while True:
    for i in range(1,gridX-1):
        for j in range(1, gridY-1):
            RHS[point(i,j)] = T_old[point(i,j)] + dt*source(x[i],y[j],t+dt, T_source)

            
    for i in range(gridX):
        for j in range(gridY):
            if i == 0 or i == gridX-1 or j == 0 or j == gridY-1:
                RHS[point(i,j)] = Tboundary(t, rate)

    #Solving system equations

    T_new = la.spsolve(A, RHS)

    T_matrix = np.full((gridX,gridY), 0.)
    for i in range(gridX):
        for j in range(gridY):
            T_matrix[i][j] = T_new[point(i,j)]
    T_mat = np.transpose(T_matrix)
    if(not plt.get_fignums()):
        exit()
    graph = plt.imshow(T_mat, cmap='jet', extent = [xL, xR, yB, yT], vmin =0, vmax = 200)
    if t == t_start :
        bar = fig.colorbar(graph, ax=ax)
    
    plt.pause(0.00001)
    T_old = T_new  
    t = t+dt
    
plt.show()
