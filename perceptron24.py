# -*- coding: utf-8 -*-
#não está funcionando
# fonte : https://gist.githubusercontent.com/peace098beat/fb4087039374ba8eef62/raw/73fa630bce0bef57b2a7838f92d215ec7b47329e/NeuralNetwork2.py
"""
kawa1128の日記 Pythonでニューラルネット
http://d.hatena.ne.jp/kawa1128/20090920/1253456109

"""
import math
import random
import sys


class NeuralNetwork(object):
    """バックプロパゲーションを使ったパーセプトロン"""

    def __init__(self, inLayer=0, hiddenLayer=0, outlayer=0, seed=0):
        """
        ネットワークの作成
        :param inLayer: 入力層のユニット数
        :param hiddenLayer: 隠れ層のユニット数
        :param outlayer: 出力層のユニット数
        :param seed: 初期ネットワークの重みの乱数の種類
        """
        if inLayer != 0 and hiddenLayer != 0 and outlayer != 0:
            self.inLayer = inLayer
            self.hiddenLayer = hiddenLayer
            self.outLayer = outlayer
            self.seed = seed
            self.createNetwork()
        else:
            print '初期化に失敗しました'

    def createNetwork(self):
        """初期ネットワークの作成"""
        random.seed(self.seed)

        # 入力層から隠れ層の重みの初期値
        self._w1 = [[random.uniform(-0.1, 0.1) for x in range(self.inLayer)] for y in range(self.hiddenLayer)]
        # self.W = [['name'],['obj']]
        # self.W[0][0] = [1,2,3]
        # print self.W

        # 隠れ層から出力層への重みの初期値
        self._w2 = [[random.uniform(-0.1, 0.1) for x in range(self.hiddenLayer)] for y in range(self.outLayer)]
        # self.W[i][i+1] = [[random.uniform() for x in range(self.wu.forwardn)] for y in range(self.backworkdn)]
        # WaitUnit.fowardLayerNumber
        # WaitUnit.backfowardLayerNumber

        # 各レイヤのユニットを初期化
        self.inVals = [0.0] * self.inLayer
        self.hiddenVals = [0.0 for x in range(self.hiddenLayer)]
        self.outVals = [0.0 for x in range(self.outLayer)]

        pass

    def sigmoid(self, x):
        """活性化関数"""
        return 1.0 / (1.0 + math.exp(-1 * x))

    def compute(self, inVals):
        """ NNに値を入力し結果を返す
        """
        self.inVals = inVals
        # 隠れ層の計算
        for i in range(self.hiddenLayer):
            total = 0.0
            for j in range(self.inLayer):
                total += self._w1[i][j] * inVals[j]
            self.hiddenVals[i] = self.sigmoid(total)

        # 出力層の計算
        for i in range(self.outLayer):
            total = 0.0
            for j in range(self.hiddenLayer):
                total += self._w2[i][j] * self.hiddenVals[j]
            self.outVals[i] = self.sigmoid(total)

        return self.outVals

    def backPropagation(self, teachVals, alpha=0.1):
        # 隠れ層から出力層の重みの更新
        for i in range(self.outLayer):
            for j in range(self.hiddenLayer):
                delta = -1.0 * alpha * (
                    -(teachVals[i] - self.outVals[i]) * self.outVals[i] * (1.0 - self.outVals[i]) * self.hiddenVals[j])
                self._w2[i][j] = self._w2[i][j] + delta

        # 入力層から隠れ層の重いみの更新
        for i in range(self.hiddenLayer):
            total = 0.0
            for k in range(self.outLayer):
                total += self._w2[k][i] * (teachVals[k] - self.outVals[k]) * self.outVals[k] * (1.0 - self.outVals[k])

            for j in range(self.inLayer):
                delta = alpha * self.hiddenVals[i] * (1.0 - self.hiddenVals[i]) * self.inVals[j] * total
                self._w1[i][j] = self._w1[i][j] + delta
        pass

    def calcError(self, teachVals):
        """ 教師信号と出力結果の二乗誤差を求める"""
        error = 0.0
        for i in range(self.outLayer):
            error += math.pow((teachVals[i] - self.outVals[i]), 2.0)
        error /= 2.0
        return error


def runPerceptron():
    inVal = [[1, 1], [1, 0], [0, 1], [0, 0]]
    outVal = [[0.0, ], [1.0, ], [1.0, ], [0.0, ]]

    NN = NeuralNetwork(inLayer=2, hiddenLayer=5, outlayer=1, seed=10)

    l = 0
    while 1:
        err = 0.0

        for (i, v) in zip(inVal, outVal):
            # 一度計算させる(入力値をセットしている)
            o = NN.compute(i)
            # 重みを更新
            NN.backPropagation(v)
            # 二乗誤差の総和を計算
            err += NN.calcError(v)

        l += 1
        print l, ":Error->", err

        if (err < 0.001):
            print "Error < 0.001"
            break

    for i, v in zip(inVal, outVal):
        o = NN.compute(i)
        print "Input->", i, "Output->", o, "Answer->", v


import numpy as np


class NeuralNetWork2(object):
    def __init__(self, layerLen=3, unisPropaty=[5, 3, 1]):
        self.W = []
        self.S = []
        L = layerLen
        self.layerLen = layerLen
        print unisPropaty
        # 各レイヤのユニットを作成
        for l in range(L):
            Sl = np.random.randn(unisPropaty[l])
            self.S.append(Sl)
        print self.S

        # 各重みを作成
        for l in range(L - 1):
            Wl = np.random.randn(unisPropaty[l + 1], unisPropaty[l])

            self.W.append(Wl)
        print self.W

    def runNeuralNetworks(self, X):
        """NNを計算し返却"""
        L = self.layerLen
        for l in range(L):
            if l == 0:
                self.S[l] = X
            else:
                self.S[l] = np.dot(self.W[l-1], self.S[l-1])

        return self.S[-1]

    def backPropagation(self):
        pass

    def calcError(self):
        pass


if __name__ == '__main__':
    # runPerceptron()
    NN = NeuralNetWork2(layerLen=3, unisPropaty=[5,3,1])
    X = np.array((1, 2, 3, 4, 5))
    res = NN.runNeuralNetworks(X)
    print 'result:', res
