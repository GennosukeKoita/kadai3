import numpy as np
import random
import math
import matplotlib.pyplot as plt

def generate_memories(m, n):
	x_mem = np.zeros((m, n))
	for alpha in range(m):
		x_mem[alpha] = random.choices([-1, 1], k=n, weights=[1, 1])
	return x_mem
	
def generate_weight(x_mem, m, n):
	w = np.zeros((n, n)) #重み付け
	for i in range(n):
		for j in range(n):
			if i == j:
				w[i][j] = 0
			else:
				x_mem_T = x_mem[:,j:j+1].reshape(1, m)
				w[i][j] = np.dot(x_mem_T,x_mem[:,i:i+1])
	w /= n
	return w
	
def memories_fipped(n, a, x, x_mem):
	for i in range(n):
		if i < a:
			x[0][i] = -x_mem[0][i]
		else:
			x[0][i] = x_mem[0][i]
	return x
	
def update_status(t, x, w, n):
	for i in range(n):
		u = 0
		u = np.dot(x[t:t+1,:],w[i:i+1,:].reshape(n,1))
		if u > 0:
			x[t+1][i] = 1
		else:
			x[t+1][i] = -1
	return x
	
def compute_s(n, xt, x_mem):
	u = np.dot(xt, x_mem[0].reshape(n, 1))
	u /= n
	return u
	
def main():
	NEURONS = 1000
	MEMORIESITEM = 80
	T = 20
	FIPPED = [i for i in range(451) if i % 50 == 0]
	x_memory = generate_memories(MEMORIESITEM, NEURONS)
	w = generate_weight(x_memory, MEMORIESITEM, NEURONS)
	for A in FIPPED:
		x = np.zeros((T, NEURONS))
		s = np.zeros(T)
		x = memories_fipped(NEURONS, A, x, x_memory)
		#print("FIPPED:",A)
		for t in range(T):
			s[t] = compute_s(NEURONS, x[t], x_memory)
			print(t,"/",s[t])
			if t == 19:
				pass
			else:
				update_status(t, x, w, NEURONS)
		plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], s)
	plt.show()
	
if __name__ == "__main__":
	main()

