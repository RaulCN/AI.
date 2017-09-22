#Código não funciona
#http://codegist.net/snippet/python/perceptronpy_damca_python
import numpy as np
 
w = (0, 0)
 
xs = ((-1, 1), 
       (0, 1),
       (10, 1))
 
ys = (1, -1, 1)
 
w = np.array(w)
xs = np.array(xs)
ys = np.array(ys)
 
def step(x):
    return 1 - 2*int(x <= 0)
 
i = 0
error_free = 0
idx = -1
while True:
    i += 1
    idx += 1
    idx = idx % len(xs)
    x, y = xs[idx], ys[idx]
    b = sum(x * y)
    b = step(b)
    dy = y - b
    dy = dy * 0.5
    if dy:
        w = w + dy*x
        error_free = 0
        print(w)
    else:
        error_free += 1
    if error_free == len(xs):
        break
    elif i > 20:
        raise OverflowError("Calculation exceeding limit")
         
print("Converged weight values are: ", w)
