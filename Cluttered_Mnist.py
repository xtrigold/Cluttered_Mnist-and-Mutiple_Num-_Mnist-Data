#The Cluttered_Mnist generation which is based on the Mnist dataset.
#This Dataset contains the random nums at diff. random locations.

import cPickle
import gzip

import numpy as np
import scipy


# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_x, train_y = train_set


    

#define a function for New Mnist dataset generation

def Data_generation(data, index):
   
    #the data size is a Mnist data: 28*28
    if data.shape != (28, 28):
        data = data.reshape(28, 28)
    
    #define a blackboard
    #define the blackboard whose size is 100*100
    board = np.zeros((100,100))
    
    
    #define the diff. locations, here the (x,y) coordinations 
    l1_x = np.random.randint(0,72)
    l1_y = np.random.randint(0,72)
    
    
    #inserted the Mnist data, which is the same num. but located in the diff. places
    board[l1_x:l1_x+28,l1_y:l1_y+28] = data
    
    
    #add noise: here contains 3 diff. noise
    noise1 = np.random.randint(0,2,(8,8))
    loc1_x = np.random.randint(0,91)
    loc1_y = np.random.randint(0,91)
    
    noise2 = np.random.randint(0,2,(4,4))
    loc2_x = np.random.randint(0,95)
    loc2_y = np.random.randint(0,95)
    
    noise3 = np.random.randint(0,2,(6,6))
    loc3_x = np.random.randint(0,93)
    loc3_y = np.random.randint(0,93)
    
    board[loc1_x:loc1_x+8,loc1_y:loc1_y+8] = noise1
    board[loc2_x:loc2_x+4,loc2_y:loc2_y+4] = noise2
    board[loc3_x:loc3_x+6,loc3_y:loc3_y+6] = noise3
    
    return board, index

#New_data, New_index = Data_generation(random_data, random_index) 
#scipy.misc.imsave('test.jpg', New_data)
N = 100*1000
New_Mnist_x = np.zeros((N, 100*100))
New_Mnist_y = np.zeros((N,))
for k in range(0,N):
    print("The turn is:" + str(k))
    # random a data which form the Mnist data train dataset
    i = np.random.randint(0, 50000)
    random_data = train_x[i]
    random_index = train_y[i]
    New_data, New_index = Data_generation(random_data, random_index) 
    New_Mnist_x[k,:] = New_data.reshape(1,100*100)
    New_Mnist_y[k] = New_index
    
#print(New_Mnist_x.shape)
#print(New_Mnist_y.shape)

#save the new data and label into "New_Mnist_data.npy" & "New_Mnist_label.npy"
np.save("Cluttered_Mnist_data.npy", New_Mnist_x)
np.save("Cluttered_Mnist_label.npy", New_Mnist_y)
