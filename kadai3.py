import numpy as np
import random
import math
import matplotlib.pyplot as plt

def weight(w, i, x_mem, memsize, k):
	for j in range(k):
		if i == j:
			w[i][j] = 0
		else:
			x_mem_T = x_mem[:,j:j+1].reshape(1, memsize)
			w[i][j] = np.dot(x_mem_T,x_mem[:,i:i+1])
	return w
	
def dynamics(x, k):
	u = 0
	for i in range(k):
		for t in range(19):
			u = np.dot(x[t:t+1,:],w[i:i+1,:].reshape(k,1))
			if u > 0:
				x[t+1][i] = 1
			else:
				x[t+1][i] = -1
	return x

#Associative memory
NEURONS = 1000
MEMORIESITEM = 80
T = 20
FIPPED = [i for i in range(451) if i % 25 == 0]
for a in FIPPED:
	x_memory = np.zeros((MEMORIESITEM, NEURONS)) #np.zeros((列, 行))
	x = np.zeros((T, NEURONS))
	w = np.zeros((NEURONS, NEURONS))
	s = np.zeros(T)
	
	#ステップ1
	for alpha in range(MEMORIESITEM):
		x_memory[alpha] = random.choices([-1, 1], k=NEURONS, weights=[1, 1])
	
	for i in range(NEURONS):
		weight(w, i, x_memory, MEMORIESITEM, NEURONS) #ステップ2
	
	#ステップ3
	x[0] = x_memory[0]
	if a != 0:
		for i in range(a):
			if x[0][i] == 1:
				x[0][i] = -1
			else:
				x[0][i] = 1
	else:
		pass
	
	dynamics(x, NEURONS) #ステップ4
	
	#ステップ5
	x_memory_0T = x_memory[0].reshape(NEURONS, 1)
	for t in range(T):
		s[t] = np.dot(x[t], x_memory_0T)
	s /= NEURONS
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], s)
plt.show()