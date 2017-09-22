#Inspiração https://gist.github.com/e135754/163ee64d915a1f7ada6271043f48011d
#a plotagem ainda não está funcionando
import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self):
        pass  
    
    def get_error(self,predict,data):
        return  data[-1] - predict

    def get_prediction(self,predict,data):
        total = np.dot(predict,data[:-1])
        return (total > 0).astype(np.int)

    def modify(self,diff,data,para):
        ALPHA = 0.1
        return para + ALPHA*diff*data[:-1]

    def learn(self,data):
        epoch = 1000
        modify_count = 0
        times = epoch

        para = np.random.random(size=[data.shape[1]-1])
        print("Valor inicial")
        print(para)
        
        for a in range(epoch):
            for element in data:
                prediction = self.get_prediction(para, element)
                error = self.get_error(prediction, element)
                if 0 != error:
                    para = self.modify(error, element, para)
                    modify_count += 1
            if 0 == modify_count:
                print("Número de tentativas")
                print(a)
                return para
            modify_count = 0
        print("Número máximo de testes alcançados")
        return para        

    def predict(self,parameteres,data):
        array = []
#        for x in np.greater(np.dot(parameteres,data[:,:3].T),np.zeros(4)):
#            array.append(1 if x else 0)
        array = [1 if x else 0 for x in np.greater(np.dot(parameteres,data[:,:3].T),np.zeros(4))]
        return array

if __name__ == "__main__":
    perceptron = Perceptron()
    data = np.array(
            [
                [0.2,0.1,1,1],
                [0.4,0.6,1,1],
                [0.5,0.2,1,1],
                [0.7,0.9,1,0]
            ]
    )
    parameters = perceptron.learn(data)
    print("Parâmetro")
    print(parameters)
    


data = np.array(
                 [
                    [ 0.2,  0.1,  1. ,  1. ],
                    [ 0.4,  0.6,  1. ,  1. ],
                    [ 0.5,  0.2,  1. ,  1. ],
                    [ 0.7,  0.9,  1. ,  0. ]
                 ]
               )
W = [ 0.20859692, -0.30753284,  0.12567978]
#W = [-0.30845968, -0.01897712,  0.22308366]
#W = [-0.29900141,  0.01123346,  0.15981181]
x=data[:,:1]
y=data[:,1:2]
x_fig = np.array(range(-10,10))
y_fig = -(W[0]/W[1])*x_fig - (W[2]/W[1])

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.plot(x[:-1],y[:-1],'o',markersize=10)
plt.plot(x[-1],y[-1],'ro',markersize=10)
plt.plot(x_fig,y_fig)
#plt.show()
plt.savefig('figure_1.png')
