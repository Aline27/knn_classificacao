import sys
import numpy as np
from math import sqrt
from operator import itemgetter
from scipy import stats
import time

def read_file (arq, N):
	cont=0
	for row in arq:
	    values = row.split()
	    if (cont==0):
			matriz=np.zeros((N,int(values[1])))
			label=np.zeros((N,1))
	    elif (cont <= N):
			matriz[cont-1,:]=values[0:132]
			label[cont-1]=values[132]
	    else:
			return matriz,label
			break
	    cont+=1

def KNN_Classification(matriz_train,matriz_test,label_train,label_test,N_train,N_test,k):

	error_num=0
	label_result_array = np.zeros((N_test, 1))
	for i_test in range(0,N_test):
		flag=0
		aux=np.ones((k,2))
		l=0
		print("Teste "+str(i_test))
		vet_distance=np.zeros((N_train,1))
		matriz_distance=np.zeros((N_train,132))
		matriz_distance=(matriz_train[:,:]-matriz_test[i_test,:])**2
		vet_distance=matriz_distance.sum(axis=1)
		vet_distance=np.sqrt(vet_distance)

		for i_train in range(0,N_train):
			if (i_train < k):
				#print vet_distance[i_train]
				aux[l, 0] = vet_distance[i_train]
				aux[l, 1] = label_train[i_train]
				l+=1
			if (i_train==k):
				aux = sorted(aux, key=itemgetter(0))
				aux = np.array(aux)
			if (i_train >=k):
				for q in range(0, k):
					if (vet_distance[i_train] < aux[q, 0]):
						for cont in range(k - 1, q):
							aux[cont, 0] = aux[cont - 1, 0]
							aux[cont, 1] = aux[cont - 1, 1]
						aux[q, 0] = vet_distance[i_train]
						aux[q, 1] = label_train[i_train]
						break

		#print (label_test[i_test])
		#print (aux)
		label_result=stats.mode(aux)
		label_result = label_result[0]
		#print (int(label_result[0,1]))
		label_result_array[i_test]=label_result[0,1]
		if(label_result[0,1]!=label_test[i_test]):
			#print("Errou")
			error_num+=1

	return label_result_array, error_num


def calc_rates(errors, N_test):

	hit_rate=(float(N_test-errors)/float(N_test))*100

	print("A taxa de acertos e de:"+str(hit_rate)+"%")
	print("A taxa de erros e de:"+str(100-hit_rate)+"%")


def Create_ConfusionMat(label1,label2,n):

	confusion_mat=np.zeros((10,10))
	for i in range(0,n):
		confusion_mat[int(label1[i]),int(label2[i])]+=1

	print confusion_mat
	return confusion_mat


text1=str(sys.argv[1])
text2=str(sys.argv[2])
k=int(str(sys.argv[3]))
archive = open("150k/"+text1, 'r')
archive2 = open("150k/"+text2, 'r')
N_train=1000#149999
N_test=100#58645#60088

matriz_train,label_train=read_file(archive,N_train)
matriz_test,label_test=read_file(archive2,N_test)
print("Calculando...")
start = time.time()
label_array,errors=KNN_Classification(matriz_train,matriz_test,label_train,label_test,N_train,N_test,k)
rates=calc_rates(errors, N_test)
confusion_mat=Create_ConfusionMat(label_test,label_array,N_test)
end = time.time()
print("Tempo de execucao: ", end - start)

archive.close()
archive2.close()









