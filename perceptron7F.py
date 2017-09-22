# Original https://gist.github.com/theSage21/1112ee74d3b3f75e85560e21bb7aebed AUTHOR: Arjoonn Sharma
#Código modificado Raul Campos Nascimento
def dot_product(vector1, vector2):
    "Returns the dot product between two vectors"
    # As afirmações nos permitem garantir que algumas coisas sejam verdadeiras antes
    # we proceede. Here dot product is only defined for vectors
    # of the same dimension
    assert len(vector1) == len(vector2), (vector1, vector2)

    total = 0
    # zip let's us iterate over two lists together
    # see https://docs.python.org/3.3/library/functions.html#zip
    for v1, v2 in zip(vector1, vector2):
        total = total + (v1 * v2)
    return total

class Perceptron:
    def __init__(self, weights, bias):
        # see https://pythontips.com/2013/08/07/the-self-variable-in-python-explained/
        # to know what 'self' is. It's similar to 'this' in Java and C
        self.weights = weights
        self.weights.append(bias)

    def get_output(self, inp_received):
        "Get output for given input"
        inp = list(inp_received)  # necessário por causa da passagem de Python pelo comportamento de referência
        inp.append(1) # BIAS TERM
        dot = dot_product(self.weights, inp)
        #return_value = 1 if (dot > 0) else 0
        # the above line is the same as the code below but more readable
        if dot > 0:
            return_value = 1
        else:
            return_value = 0
        return return_value

    def learn_single_datapoint(self, inp, out_expected):
        "Train on single training example"
        out = self.get_output(inp)
        err = out_expected - out
        # modificar os pesos
        # For list comprehensions, see http://www.secnetix.de/olli/Python/list_comprehensions.hawk
        inp_with_bias = list(inp) + [1]
        changes_to_be_made = [err * i for i in inp_with_bias]
        new_weights = [weight + i for i, weight in zip(changes_to_be_made, self.weights)]
        self.weights = new_weights

    def train(self, input_set, output_set):
        "Treine um conjunto de exemplos"
        for inp, out in zip(input_set, output_set):
            self.learn_single_datapoint(inp, out)
        return ("progresso")

# ------------------------------------------------------------------------------------------
# Agora testamos nosso perceptron
p = Perceptron([0, 0], 1)
inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
# vemos se o perceptron pode aprender a função AND
outputs = [(i[0] * i[1]) for i in inputs]

# We see the outputs when the perceptron is not trained
for inp, out in zip(inputs, outputs):
    inpts = list(inp)
    pred = p.get_output(inpts)
    print('{} como entrada, {} esperado, obteve {}'.format(inp, out, pred))

print('pesos são: {}'.format(p.weights))
print('-'*10)

# Nós treinamos agora
# Uma vez que 4 exemplos é pouco demais para aprender, vamos
# duplicá-los 100 vezes para obter um total de 400 exemplos
# see basic list repetition at www.tutorialspoint.com/python/python_lists.htm
repeated_inputs = inputs * 200  # repetimos os elementos 200 vezes para criar uma lista de 800 elementos
repeated_outputs = outputs * 200  # o mesmo que acima


progress = p.train(repeated_inputs, repeated_outputs)

print('pesos são: {}'.format(p.weights))
for inp, out in zip(inputs, outputs):
    inpts = list(inp)
    pred = p.get_output(inp)
    print('{} como entrada, {} esperado, obteve {}'.format(inpts, out, pred))
print('O percptron aprendeu nossa função AND')
