# Project 02  

1. Generate the training set:  
  a.   𝑁_train = 10  
  b.   𝑋_train contains samples from a uniform distribution U(0,1).  
  c.   𝑡_train = sin(2pi*X_train) + 𝜀, where 𝜀 contains samples from a Gaussian distribution N(0, 𝜎 =0.3).  
2. Generate the test set:
  a.   𝑁_test = 10  
  b.   𝑋_test contains samples from a uniform distribution U(0,1).  
  c.   𝑡_test = sin(2pi*X_test) + 𝜀, where 𝜀 contains samples from a Gaussian distribution N(0, 𝜎 =0.3).  
3. Use the method of “linear regression with non-linear models” to fit a polynomial of degree 𝑀 = 0,1,2, … ,9 to the training set. 
4.  Record the training and testing errors for each of the 10 cases.
5.  Produce the plot as shown below, where
𝐸𝐸𝑅𝑅𝑅𝑅𝑅𝑅  = �𝐽𝐽�𝑤𝑤�/𝑁𝑁
6. Repeat the exercise for 𝑁𝑁𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇𝑇 = 100.
Upload your m-file to Blackboard before midnight Saturday, Feb 10.

