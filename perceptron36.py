#concertei mais ainda n funciona
#fonte https://gist.githubusercontent.com/TrisZaska/21a9930d20c6cb98c429dbb180468efa/raw/00c9397c7638b4a534e0d185661fe34c7f29901f/perceptron_snippet_code_3.py
#Train Perceptron with initial X and y
pct = Perceptron()
pct.train(X, y)

#Show the weight of Perceptron after learning
print ('The weight of Perceptron after learning is: ')
print (pct.weight_)
print ('The output of Perceptron after learning is: ')
print (pct.predict(X))


#Plot the error every epoch
plt.plot(range(len(pct.error_)), pct.error_)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
