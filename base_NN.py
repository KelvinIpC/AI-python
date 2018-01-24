#'cause i use macbook, this one is to eliminate some warnings
#if dont use it, the result would not be affected

import numpy as np
def relu(input_x, derv = False):
	if derv == True:
		input_x[input_x>0] = 1
		input_x[input_x<=0] = 0
			
	input_x[input_x>0] = 1
	input_x[input_x<=0] = 0
	return input_x
def sigmoid(input_x, derv = False):
	if(derv == True):
		return input_x*(1-input_x)
	else:
		return 1/(1+np.exp(-input_x))

class base_NN:

	def __init__(self,ins_size, 
		outs_size, layers_size, layers_activation_function):
		self.activation_function = layers_activation_function
		self.ins_size = ins_size
		self.outs_size = outs_size
		self.layers_size = layers_size
		self.weight = []
		for i in range(layers_size-1):
			self.weight.append(np.random.random((ins_size[i], outs_size[i]))+1)

	def train(self, input_x, input_y = None, locked = False):
		layer = [input_x]
		for i in range(self.layers_size-1):
			layer.append(self.activation_function(np.dot(layer[i],self.weight[i])))
			

		if not locked:
			delta = None
			for i in range(self.layers_size-1):
				index = self.layers_size - i - 2
				
				if index == self.layers_size-2:
					delta = (input_y - layer[index+1])*self.activation_function(layer[index+1], True)
					self.weight[index] += layer[index].T.dot(delta)
				else:
					delta = delta.dot(self.weight[index+1].T)*self.activation_function(layer[index+1], True)
					self.weight[index] += layer[index].T.dot(delta)


		return layer[self.layers_size-1]

	def predict(self, input_x):
		return self.train(input_x, locked = True)

def MSE_Error( layer, input_y):
	return (np.mean(np.abs(input_y - layer)))

nn = base_NN(ins_size = [3, 4, 5, 4],
	outs_size = [4, 5,4,1],
	layers_size = 5,
	layers_activation_function = sigmoid)

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0, 1, 1, 0]]).T


for i in range(600000):
	temp = nn.train(input_x = X,input_y = y)
	if i%10000 == 0:
		print(i/ 10000, MSE_Error(temp, y))
		
	if MSE_Error(temp, y) < 0.01:
		print("train is done with the error rate at: ",MSE_Error(temp, y), temp)
		break
print(temp)

X1 = np.array([ [0,0,0],[0,1,0],[1,0,1],[1,1,0] ])
print("\n the answer\n",nn.predict(X1))