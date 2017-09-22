#ESSE SCRIPT ESTÁ FUNCIONANDO
#Copyright (c) Jesús Manuel Mager Hois 2016
#https://gist.github.com/pywirrarika/6a8a05e9911cab260fe9de045a99b44e


import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self, data):
        self.ciclosx = []
        self.ciclosy = []
        
        self.d = data
        self.data = np.insert(self.d, len(self.d[0])+2, 1, axis=1)
        self.wn = []

    def plotD(self):
        nd = np.transpose(self.d)

        x3 = nd[0].tolist()
        y3 = nd[1].tolist()
        z3 = nd[2].tolist()

        x3 = x3[0]
        y3 = y3[0]
        z3 = z3[0]

        print(nd[0][0])
        print(nd[1][0])
        print(nd[2][0])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(nd[0].tolist(), nd[1].tolist(), nd[2].tolist(), c='r', marker='o')

        ax.set_xlabel('X1') #nomeando eixo x1
        ax.set_ylabel('x2') #nomeando eixo x2
        ax.set_zlabel('x3') #nomeando eixo x3
        plt.title("Dados a classificar")

        plt.show()

    def train(self,convergence = 0.3, output = 1):
        r = convergence
        wn = np.array([1,1,1,r])
        newvec = 0
        classe = 0
        itr = 0
        while True:
            itr = itr + 1
            errors = 0
            j = 0
            if output:
                print("-"*75)
            for x in self.data:
                classereal = 0
                xnwn = 0
                 
                x = np.array(x.tolist()[0])
                for i in range(len(x)):
                    xnwn= xnwn + x[i] * wn[i]
                if j < 5:
                    classereal = 1
                else:
                    classereal = 2
                # Des rule
                if xnwn < 0: 
                    classe = 1
                else:
                    classe = 2
             
                pen = "0"
                if classe == 2 and classe != classereal:
                    newvec = wn - r*x
                    errors = errors + 1
                    pen = "-x"
                if classe == 1 and classe != classereal:
                    newvec = wn + r*x
                    errors = errors + 1
                    pen = "+x"

                if output:
                    print(str(x), "\t|", str(wn), "\t|", str(xnwn), "\t|", str(clase), "\t|", str(clasereal), "\t|", pen)

                wn = newvec
                
                j = j + 1
            if errors == 0:
                self.wn = np.array(wn)
                break

        print("Número de interações:", str(itr)) 
        self.ciclosx.append(r)
        self.ciclosy.append(itr)

    def classify(self, vector):
        xnwn = 0
        vector.append(1)
        x=np.array(vector)

        for i in range(len(x)):
            xnwn= xnwn + x[i] * self.wn[i]
        if xnwn < 0: 
            classe = 1
        else:
            classe = 2
        print("Classe:", str(classe), str(xnwn), str(vector))
        return classe
    def eval(self):
        C1,C2 = np.split(self.d, 2)
        C=[C1,C2]
        i = 0
        self.confmatrix = np.zeros((len(C), len(C)))
        for Classe in C:
            i += 1
            for vector in Classe:
                vector = vector.tolist()[0]
                disc = self.classify(vector)
                self.confmatrix[i-1][disc-1] += 1
        print("Matriz de confusão por substituição") 
        print(self.confmatrix)
    

        self.data= []
        probe = []
        for Classe in C:
            Classe = np.matrix(Classe)
            A,B = np.split(Classe, 2)
            self.data.append(A)
            probe.append(B)

        self.data = np.insert(self.d, len(self.d[0])+2, 1, axis=1)
        
        self.train(output=0)
        i=0
        self.confmatrix = np.zeros((len(probe), len(probe)))
        for b in probe:
            i += 1
            for vector in b:
                vector = vector.tolist()[0]
                disc = self.classify(vector)
                self.confmatrix[i-1][disc-1] += 1
        
        print("Matriz de confusão por Cross Validation") 
        print(self.confmatrix)

if __name__ == "__main__":
    data = np.matrix([[0,0,0],
               [1,0,1],
               [1,0,0],
               [1,1,0],
               [0,0,1],
               [0,1,1],
               [0,1,0],
               [1,1,1]])
    neuron = Perceptron(data=data)

    neuron.plotD()
    neuron.train(output=0)
    neuron.eval()
