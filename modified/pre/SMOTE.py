from sklearn.neighbors import NearestNeighbors
import random
def smote(X,radio,k):
	# X:input feature vectors,Y:output feature vectors,radio:must be integer,the number of Y's vectors /X's, k: k nearest neighbors
 	n = len(X)
	#sum_num = int(n*radio)
	neigh = NearestNeighbors(k, 1,'kd_tree')
	neigh.fit(X)
	Y = []
	for line in X:
		neighborsIDbysklearn = neigh.kneighbors(line, 2, return_distance=False)
		#print neighborsIDbysklearn 
		neighborsID=neighborsIDbysklearn[0]
		#print neighborsID
		new_vector = synthetic_vector(line,neighborsID,X,radio)
		Y+=new_vector
		Y.append(line)
	return Y 


def synthetic_vector(vector,neighborsID,X,radio):
	# X feature space, vector: based on this vector sythetize new vector
	k = len(neighborsID)
	Y = []
	for i in range(radio):
		new_vector= []
		length = len(vector)
		nn=random.randint(0, k-1)
		for j in range(length):
			diff = X[neighborsID[nn]][j]-vector[j]
			gap = random.random()
			new_vector.append(vector[j]+gap*diff)

		Y.append(new_vector)

	return Y	


def downsample(X,radio):
	# X: input vectors, radio: sampling radio,precision reach 0.001
	Y=[]
	number = len(X)
	r=int(radio*1000)
	cont =0
	while len(Y)<int(radio*number) and cont<10:
		for item in X:
			randnum = random.randint(0, 999) 
			if randnum in range(r) and item.count(0)<=3:
				Y.append(item)
				if len(Y)>=int(radio*number):
					break
		cont+=1
	return Y	

		



