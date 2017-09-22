# -*- coding: utf-8 -*-
#fonte https://gist.github.com/YusukeKanai/cfa78b9ce4aacab31f66839ca6295edf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Coletar dados de amostra 1

    p1 = np.random.rand(100)*10
    r1 = np.random.rand(100)*12
    q1 = 2*p1+r1

    # Coletar dados de amostra 1
    p2 = np.random.rand(100)*10
    r2 = np.random.rand(100)*12
    q2 = 3*p2+r2+15

    data = np.c_[p1, q1, p2, q2]

    # Visualização de gráfico

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # add data points on graph
    ax.scatter(p1,q1, c='vermelho')
    ax.scatter(p2,q2, c='azul')

    # Configuração do gráfico
    ax.set_title('segundo gráfico de dispersão')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim([0,12])
    plt.ylim([0,50])

    # Set initial hyperplane (linear)
    w0 = 1.
    w1 = -1.
    x = np.linspace(0,10,2)
    y = -1*w0/w1 * x # normal vector is [w0, w1]

    plt.plot(x, y, "r-")

    fig.show()

    import time
    time.sleep(1)
    
    for dp1, dq1, dp2, dq2 in data:
        f = (w0 * dp1 + w1 * dq1)
        if f <= 0:
            w1 = w1 + dp1
            w0 = w0 + dq1

            x = np.linspace(0,10,2)
            y = -1 * w0/w1 * x
            plt.plot(x, y, "r-")

            fig.canvas.draw()
            time.sleep(1)

        f = (w0 * dp2 + w1 * dq2) * (-1)
        if f <= 0:
            w1 = w1 - dp2
            w0 = w0 - dq2

            x = np.linspace(0,10,2)
            y = -1 * w0/w1 * x
            plt.plot(x, y, "r-")

            fig.canvas.draw()
            time.sleep(1)
