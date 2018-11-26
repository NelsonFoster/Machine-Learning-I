# %%%%%%%%%%%%% Python %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 Jan - 04 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Math Python %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# =============================================================
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------Probablity ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

#Nelson Foster Homework 1

# Q-1: Import packages that you can generate random numbers.
import matplotlib.pyplot as plt
import numpy as np
import math
import random

matrix1 = np.array([[1,2,3], [1,2,3],[1,2,3]])
matrix2 = np.array([[1],[1],[1]])
matrixmult = np.dot(matrix1, matrix2)
logsig=1/(1+np.exp(-matrixmult))
print(matrix2)
print(logsig)


# ----------------------------------------------------------------------------------------------------------------------

# Q-2: a) Generate 100 samples points from a uniform distribution. Plot your results and verify it.


UD =np.random.uniform(0.0, 1.0, 100)

print (UD)
# ----------------------------------------------------------------------------------------------------------------------

# Q-2: b)Generate 1000 samples points from normal distribution (with mean zero and std 1). Plot your results and verify it.

mu, sigma = 0, 0.1
ND =np.random.normal(mu, sigma, 1000)
abs(mu - np.mean(ND)) < 0.01
abs(sigma - np.std(ND, ddof=1)) < 0.01
count, bins, ignored = plt.hist(ND, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
linewidth=2, color='r')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Q-3: a)Generate a list of numbers (random or arbitary) and convert it to a numpy array.

RL =np.random.rand(5,10)

print(RL)

np.array(RL)

print(RL)
# ----------------------------------------------------------------------------------------------------------------------

# Q-3: b)Create 1x3, 3x3 and 5x5 size matrices. Use different python methods to generates these matrices that you know of (It can be random or manually enter numbers).


M1 = np.random.rand(1, 3)

##M1 = np.arrange(8, 48).reshape(3,3)

M2 = np.matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

rows = 5
cols = 5
M3 = np.matrix(np.random.randint(25,50, size=(rows, cols)))

print(M1, M2, M3)


# ----------------------------------------------------------------------------------------------------------------------
# Q-3: c) Multiply the first two matrix by the other elementwise and matrix multiplication if it is possible. Check your results by hand.

    
#matrix multiplication

np.matmul(M1, M2)
    
    
#elementwise


np.multiply(M1, M2)









# ----------------------------------------------------------------------------------------------------------------------
# Q-4: a) Create a matrix size of 3x3. Create a new varibale and name it var then add the first and second column of matrix then sum the the vector. Save the results in var.

rows = 3
cols = 3

M4 = np.matrix(np.random.randint(10,30, size=(rows, cols)))

var = sum((M4[:,:2]))

print(var)



# ----------------------------------------------------------------------------------------------------------------------
# Q-4: b) Create a vector of 100 samples (with any python method that you have in your mind). Plot the vector. Calcuate the mean and std.

V1 = random.sample(range(1, 100), 3)

print(V1)

np.mean(V1,axis=0)
np.std(V1)



# ----------------------------------------------------------------------------------------------------------------------
# Q-4: c) Create a matrix 3x3 and 3x1. Multiply the first matrix by the second matrix. 
#         Define a logsig transfer function.
#         Pass the results of the the multiplication to the transfer function (vector vize). Check does it make sense. Explain your results

rows = 3
cols = 3

M5 = np.matrix(np.random.randint(10,30, size=(rows, cols)))

rows = 3
cols = 1

M6 = np.matrix(np.random.randint(10,30, size=(rows, cols)))

LS = M5*M6                                                                             

def sigmoid_array(LS):                                        
    return 1 / (1 + np.exp(-LS))


print(LS)
print(sigmoid_array)                                             
                                                                              
# ----------------------------------------------------------------------------------------------------------------------
# Q-5: a) Expand (x+y)^6.

import sympy

from sympy import init_printing, Symbol, expand, simplify
init_printing()
x = Symbol('x')
y = Symbol('y')
e = (x + y)**6
e
(x + y) 
e.expand()
   
   

# ----------------------------------------------------------------------------------------------------------------------
# Q-5: b) Simplify the trigonometric expression sin(x) / cos(x)



init_printing()
x = Symbol('x')
y = Symbol('y')


simplify(sin(x) / cos(x))



# ----------------------------------------------------------------------------------------------------------------------
# Q-5: c) Calulate the derivative of log(x) for x

from sympy import symbol, Derivative

Derivative(log(x), x)


# ----------------------------------------------------------------------------------------------------------------------
# Q-5: d) Solve the system of equations x + y = 2, 2x + y = 0

#defining matrices

A = np.matrix([[1,1],[2,1]])
B = np.matrix([[2],[0]])

A_inverse = np.linalg.inv(A)

#solve for x

X = A_inverse * B

print(X)
# ----------------------------------------------------------------------------------------------------------------------
# Q-6:  Estimating Pi using the Monte Carlo Method
#      1- To estimate a value of Pi using the Monte Carlo method - generate a large number of random points and see
#         how many fall in the circle enclosed by the unit square.
#      2- Check the following link for instruction
#      3- There are variety of codes available in the net please write your own code.


#shots made
inside = 78
# Total number of basketballs to shoot
total = 100

# Iterate for the number of basketballs
for i in range(0, total):
  
  x2 = random.random()**2
  y2 = random.random()**2
    
if math.sqrt(x2 + y2) < 1.0:
     inside += 1


pi = (float(inside) / total) * 4


print(pi)










