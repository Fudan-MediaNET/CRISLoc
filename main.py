# User: Ruby
# Date: 2019/7/23 7:35
# Version: PyCharm 3.7.3

import numpy as np
import itertools
import sklearn.neighbors
from scipy.special import comb
from sklearn.cluster import DBSCAN, AffinityPropagation
import scipy.io
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import sklearn.neighbors
import math

class AP_detect:
    def __init__(self, _train_x, _train_y, _test_x, _test_y, _altered_ap, _dim, _num_subset, _neighboursLOF, _method,
                 _ifToDB, _breakPointWeight, _neighboursDBSCAN, _filePath):
        if _ifToDB:
            self.train_x = self.toDb(_train_x)
            self.test_x = self.toDb(_test_x)
        else:
            self.train_x = _train_x
            self.test_x = _test_x
        self.test_y = _test_y
        self.train_y = _train_y
        self.altered_ap = _altered_ap
        self.dim = int(_dim)
        num_test, m = self.test_x.shape
        print(num_test)
        self.ap_num = m // self.dim
        self.num_subset = _num_subset
        num_combs = 0
        for i in _num_subset:
            num_combs += comb(self.ap_num, i)
        self.num_combs = int(num_combs)
        self.neighboursLOF = _neighboursLOF
        self.method = _method
        self.breakPointWeight = _breakPointWeight
        self.neighboursDBSCAN = _neighboursDBSCAN
        self.filePath = _filePath

    def toDb(self, x):
        n, m = x.shape
        dbUnit = np.zeros((n, m))
        for i in range(0, n):
            for j in range(0, m):
                if x[i, j] == 0:
                    dbUnit[i, j] = -100
                else:
                    dbUnit[i, j] = 10 * math.log10(x[i, j])
        return dbUnit

    def generate_ave_2dim(self, X, Y):
        # X is a n*m array
        n, m = X.shape
        Y_coded = np.floor(Y[:, 0] * 1000000 + Y[:, 1] * 100)
        Y_new = np.unique(Y_coded)
        pos_num = len(Y_new)
        X_new = np.zeros((pos_num, m))
        X_counter = np.zeros(pos_num)
        Y_new_2dim = np.zeros((pos_num, 2))
        for i in range(0, n):
            ii = int(np.argwhere(Y_new == int(Y_coded[i])))
            X_new[ii, :] += X[i, :]
            X_counter[ii] += 1
        for i in range(0, pos_num):
            X_new[i,:] /= X_counter[i]
            ii = np.where(Y_coded == Y_new[i])
            Y_new_2dim[i,:] = Y[ii[0][0],:]
        return X_new, Y_new_2dim

    def myWKNN(self, Xs, Xt):
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean').fit(
            Xs)
        distances, indices = nbrs.kneighbors(Xt)
        num_test, m = Xt.shape
        predicts = np.zeros((num_test, 2))
        for i in range(0, num_test):
            weights = 1 / distances[i, :]
            weights = weights / np.sum(weights)
            index_nei = indices[i, :]
            labels_twodim = self.train_y_ave[index_nei, :]
            predicts[i, 0], predicts[i, 1] = np.sum(np.multiply(labels_twodim[:, 0], weights)), \
                                             np.sum(np.multiply(labels_twodim[:, 1], weights))
        return predicts

    def subsetMatrix(self, item, num_test):
        subsetArray = np.zeros(self.ap_num, dtype=np.int)
        for tempIndex in item:
            subsetArray[tempIndex] = 1
        subsetMatrix = np.tile(subsetArray, [num_test, 1])
        return subsetMatrix

    def subsetResults(self):
        num_test, m = self.test_x.shape
        x = range(0, self.ap_num)
        counter = 0
        predicts = np.zeros((num_test * self.num_combs, 2))
        subsets = np.zeros((num_test * self.num_combs, self.ap_num))
        self.train_x_ave, self.train_y_ave = self.generate_ave_2dim(self.train_x, self.train_y)
        for r in self.num_subset:
            for item in itertools.combinations(x, r):
                index = [list(range(i * self.dim, (i + 1) * self.dim)) for i in item]
                index = np.ravel(np.array(index))
                Xt_subset = self.test_x[:, index]
                Xs_subset = self.train_x_ave[:, index]
                predict_subset = self.myWKNN(Xs=Xs_subset, Xt=Xt_subset)
                index_subset = self.subsetMatrix(item=item, num_test=num_test)
                predicts[counter * num_test : (counter + 1) * num_test, :] = predict_subset
                subsets[counter * num_test : (counter + 1) * num_test, :] = index_subset
                counter += 1
        np.save('predicts.npy', predicts)
        np.save('subsets.npy', subsets)

    def dispersion_baseline(self):
        predicts = np.load('predicts.npy')
        num_test, m = self.test_x.shape
        meanDistArray = np.zeros(num_test)
        for i in range(0, num_test):
            index = np.array([x for x in range(i, num_test * self.num_combs, num_test)])
            predict = predicts[index, :]
            meanDistArray[i] = np.mean(scipy.spatial.distance.pdist(predict))
        print(np.mean(meanDistArray), np.std(meanDistArray))

    def chooseParameter_oneSample(self, predict, ifPlot):
        kDistance = np.zeros(len(predict))
        Y = pdist(predict, 'euclidean')
        v = squareform(Y)
        for j in range(0, len(predict)):
            kDistance[j] = np.sort(v[j, :])[self.neighboursDBSCAN - 1]
        kDisUnsorted = kDistance
        kDistance = np.sort(kDistance)
        # if ifPlot == True:
        #     plt.plot(kDistance)
        #     plt.show()
        # scipy.io.savemat('kDistance.mat', {'kDistance': kDistance})
        return kDistance[int(0.8*len(predict))], kDisUnsorted

    def clusterDBSCAN(self, _eps, subset, predict, groundTruth, ifPlot):
        if _eps < 0:
            _eps, kDisUnsorted = self.chooseParameter_oneSample(predict, ifPlot)
        clustering = DBSCAN(eps=_eps, min_samples=self.neighboursDBSCAN).fit_predict(predict)
        temp = np.where(clustering >= 0)[0]
        if temp.__len__() != 0:
            clusters = clustering[temp]
            counts = np.bincount(clusters)
            densest = np.argmax(counts)
            sampleIndex = np.where(clustering == densest)[0]
            frequence = np.sum(subset[sampleIndex, :], axis=0)
            if ifPlot == True:
                self.plotResults(subset=subset, predict=predict, groundTruth=groundTruth, y=sampleIndex)
        else:
            frequence = np.zeros((1, self.ap_num))
        return frequence

    def clusterAP(self, subset, predict):
        clustering = AffinityPropagation().fit_predict(predict)
        if all(clustering >= 0):
            counts = np.bincount(clustering)
            densest = np.argmax(counts)
            sampleIndex = np.where(clustering == densest)[0]
            frequence = np.sum(subset[sampleIndex, :], axis=0)
        else:
            frequence = np.zeros((1, self.ap_num))
        return frequence

    def cAndoDBSCAN(self, subset, predict, ifPlot, sizeRatio, groundTruth):
        _eps, kDis = self.chooseParameter_oneSample(predict, ifPlot=ifPlot)
        clustering = DBSCAN(eps=_eps, min_samples=4)
        y = clustering.fit_predict(predict)
        temp = np.where(y >= 0)[0]
        methodChange = False
        if len(temp) != 0:
            clusters = y[temp]
            counts = np.bincount(clusters)
            counts.sort()
            counts = counts[::-1]
            if (len(counts) > 1 and counts[0] / counts[1] <= sizeRatio and len(np.where(clustering.labels_ == -1)[0]) > 1) or \
                    counts[0] > 0.5 * self.num_combs:
                methodChange = True
                """
                outlierIndex = np.where(clustering.labels_ == -1)[0]
                weights = kDis[outlierIndex]
                weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
                weights = np.reshape(weights, (len(weights), 1))
                weights_matrix = np.dot(weights, np.ones((1, self.ap_num)))
                frequence = np.sum(np.multiply(subset[y == -1, :], weights_matrix), axis=0)
                if ifPlot == True:
                    self.plotResults(subset=subset, predict=predict, groundTruth=groundTruth, y=outlierIndex)
                """
                detector = sklearn.neighbors.LocalOutlierFactor(n_neighbors=self.neighboursLOF, metric='euclidean')
                y = detector.fit_predict(predict)
                if ifPlot:
                    self.plotResults(subset=subset, predict=predict, groundTruth=groundTruth, y=y)
                if any(y == -1):
                    weights = - detector.negative_outlier_factor_[y == -1]
                    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
                    weights = np.reshape(weights, (len(weights), 1))
                    weights_matrix = np.dot(weights, np.ones((1, self.ap_num)))
                    frequence = np.sum(np.multiply(subset[y == -1, :], weights_matrix), axis=0)
                else:
                    frequence = np.sum(subset[y == -1, :], axis=0)
            else:
                counts = np.bincount(clusters)
                densest = np.argmax(counts)
                sampleIndex = np.where(y == densest)[0]
                frequence = np.sum(subset[sampleIndex, :], axis=0)
                if ifPlot == True:
                    self.plotResults(subset=subset, predict=predict, groundTruth=groundTruth, y=sampleIndex)
        else:
            frequence = np.zeros((1, self.ap_num))
        return methodChange, frequence

    def outlierLOF(self, subset, predict, ifPlot, ifWeight, groundTruth):
        detector = sklearn.neighbors.LocalOutlierFactor(n_neighbors=self.neighboursLOF, metric='euclidean')
        y = detector.fit_predict(predict)
        if ifPlot == True:
            self.plotResults(subset=subset, predict=predict, groundTruth=groundTruth, y=y)
        if ifWeight == True and any(y == -1) == True:
            weights = - detector.negative_outlier_factor_[y == -1]
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            weights = np.reshape(weights, (len(weights), 1))
            weights_matrix = np.dot(weights, np.ones((1, self.ap_num)))
            frequence = np.sum(np.multiply(subset[y == -1, :], weights_matrix), axis=0)
        else:
            frequence = np.sum(subset[y == -1, :], axis=0)
        return frequence

    def plotResults(self, subset, predict, groundTruth, y):
        index_wrong = np.zeros(len(subset))
        for k in range(0, len(subset)):
            if (subset[k, self.altered_ap] == 1).any():
                index_wrong[k] = 1
        plt.scatter(predict[index_wrong == 1, 0], predict[index_wrong == 1, 1], c='b')
        plt.scatter(predict[index_wrong == 0, 0], predict[index_wrong == 0, 1], c='r')
        plt.scatter(groundTruth[0], groundTruth[1], c='g')
        plt.show()
        scipy.io.savemat('right&wrong.mat', {'right': predict[index_wrong == 0, :],
                                             'wrong': predict[index_wrong == 1, :],
                                             'goundTruth': groundTruth})
        """
        if self.method == 1:
            plt.scatter(predict[y == -1, 0], predict[y == -1, 1], s=10, c='y', marker='*')
        else:
            plt.scatter(predict[y,0], predict[y, 1], s=10, c='y', marker='*')
        """
        # plt.scatter(predict[y == -1, 0], predict[y == -1, 1], s=10, c='y', marker='*')

    def computebreakWeight(self, predict):
        a = np.mean(scipy.spatial.distance.pdist(predict))
        weight = math.exp(-(a - 2 * 0.265044)) + 1
        return weight

    def multitest(self, test_group, _eps, ifPlot, ifWeight, sizeRatio):
        predicts = np.load('predicts.npy')
        subsets = np.load('subsets.npy')
        num_test, m = self.test_x.shape
        frequenceList = np.zeros((num_test, self.ap_num))
        methodArray = np.zeros(num_test, dtype=bool)
        breakpointList = np.zeros(num_test)
        kKeepUp = np.zeros((num_test, 2), dtype=np.int32)
        iList = np.zeros(num_test)
        with open("results.txt", "w") as f:
            i = 0
            while i < num_test:
                k = 0
                index = np.array([], dtype=np.int32)
                while k < test_group and i + k < num_test - 1 and self.test_y[i + k][0] == self.test_y[i + k + 1][0] and \
                        self.test_y[i + k][1] == self.test_y[i + k + 1][1]:
                    temp = np.array([x for x in range(i + k, num_test * self.num_combs, num_test)])
                    index = np.hstack((index, temp))
                    k += 1
                if len(index) == 0:
                    i = i + k + 1
                    continue
                subset = subsets[index, :]
                predict = predicts[index, :]
                groundTruth = self.test_y[i,:]
                if self.method == 1:
                    frequence = self.outlierLOF(subset=subset, predict=predict, ifPlot=ifPlot, ifWeight=ifWeight, groundTruth=groundTruth)
                elif self.method == 2:
                    frequence = self.clusterDBSCAN(_eps=_eps, subset=subset, predict=predict, ifPlot=ifPlot, groundTruth=groundTruth)
                elif self.method == 3:
                    frequence = self.clusterAP(subset=subset, predict=predict)
                else:
                    methodChange, frequence = self.cAndoDBSCAN(subset=subset, predict=predict, ifPlot=ifPlot, sizeRatio=sizeRatio, groundTruth=groundTruth)
                    methodArray[i] = methodChange
                # breakpointList[i] = self.computebreakWeight(predict=predict)
                breakpointList[i] = self.breakPointWeight
                frequenceList[i, :] = frequence
                kKeepUp[i, 0] = i
                kKeepUp[i, 1] = i + k
                f.write("location" + str(i) + str(i + k) + ":" + str(frequenceList[i, :]) + "\n")
                iList[i] = 1
                i = i + k
            frequenceList = frequenceList[iList == 1, :]
            methodArray = methodArray[iList == 1]
            breakpointList = breakpointList[iList == 1]
            kKeepUp = kKeepUp[iList == 1]
            np.save('frequenceList.npy', frequenceList)
            np.save('methodArray.npy', methodArray)
            np.save('breakpointList.npy',breakpointList)
            tempPath = self.filePath + 'kKeepUp'
            scipy.io.savemat(tempPath, {'kKeepUp':kKeepUp})

    def breakPointFun(self, item, ifOutlier, breakWeight):
        item_sort = np.array(item)
        item_sort.sort()
        min_var = np.inf
        min_var_index = 0
        if ifOutlier:
            # k = self.breakPointWeight
            k = breakWeight
        else:
            # k = 1 / self.breakPointWeight
            k = 1 / breakWeight
        for j in range(1, self.ap_num):
            var_l = np.var(item_sort[0:j])
            var_h = np.var(item_sort[j:self.ap_num])
            var_sum = k * var_h + var_l
            if min_var > var_sum:
                min_var = var_sum
                min_var_index = j
        breaks = item_sort[min_var_index]
        return breaks

    def majorityWeighting(self, _min_num, _max_num, _certainty):
        frequence = np.load('frequenceList.npy')
        methodArray = np.load('methodArray.npy')
        breakpointList = np.load('breakpointList.npy')
        if self.method != 4:
            print(sum(frequence))
        else:
            print(sum(frequence[methodArray==True, :]), sum(frequence[methodArray == False, :]))
        n, m = frequence.shape
        moved_ap = np.zeros((n, self.ap_num))
        for i in range(0, n):
            item = frequence[i, :]
            if (self.method == 4 and methodArray[i]) or self.method == 1:
                breaks = self.breakPointFun(item=item, ifOutlier=True, breakWeight=breakpointList[i])
                ap_index = np.where(item >= breaks)[0]
            else:
                breaks = self.breakPointFun(item=item, ifOutlier=False, breakWeight=breakpointList[i])
                ap_index = np.where(item < breaks)[0]
            for j in ap_index:
                moved_ap[i, j] = 1
        # Majority Weighting
        alteredAP = np.zeros((n, self.ap_num), dtype=np.int32)
        a = list(range(n))
        random.shuffle(a)
        moved_ap = moved_ap[a, :]
        ans = np.zeros(self.ap_num)
        i = 0
        counter = 0
        j = 0
        undetected_counter = 0
        while (i < n) & (j < n):
            j = i + _min_num
            while j < n:
                temp = moved_ap[i:j, :]
                certainty = np.sum(temp, axis=0) / (j - i)
                if np.where((certainty < _certainty) & (certainty > (1 - _certainty)))[0].__len__() <= max(
                        [self.ap_num / 4, 1]) or j - i >= _max_num:
                    if j - i >= _max_num:
                        altered = np.where(certainty >= 0.5)[0]
                    else:
                        altered = np.where(certainty >= _certainty)[0]
                    if len(altered) == 0:
                        undetected_counter += 1
                    for item in altered:
                        ans[item] += 1
                        alteredAP[a[i:j], item] = 1
                    i = j
                    counter += 1
                    break
                else:
                    j = j + 1
        print("Undetected Counter:" + str(undetected_counter))
        tempPath = self.filePath + "detect"
        scipy.io.savemat(tempPath, {'alteredAP': alteredAP})
        ans, counter = self.computeAcc(ans=ans, counter=counter)
        return ans, counter

    def computeAcc(self, ans, counter):
        num_recall = 0
        for item in self.altered_ap:
            num_recall += ans[item]
        acc = num_recall / sum(ans)
        recall = num_recall / (counter * len(self.altered_ap))
        print(ans, counter)
        print("accuracy", acc, "recall", recall)
        return ans, counter

def main():
    filepath = '../../Nexmon/frame_number_check/crisscross_floor3/repeat_scale_dB_319_No12toNo9altered_9only'
    data = scipy.io.loadmat(filepath)
    _train_x = data['train_x']
    _train_y = data['train_y']
    _test_x = data['test_x']
    _test_y = data['test_y']
    _altered_ap = np.array([1])  # follow the rule of Python
    _dim = 49
    _row_num = 10
    _num_subset = np.array([3])
    _test_group = 3
    _neighboursLOF = 168
    _neighboursDBSCAN = 4
    _method = 4  # 1: LOF, 2: DBSCAN, 3: AP
    _min_num, _max_num, _certainty = 5, 30, 0.62
    _eps = -5  # if _eps<0, then automatic.
    _breakPointWeight = 1
    _sizeRatio = 2
    _ifWeight, _ifPlot, _ifToDB = True, False, False
    if _method == 1:
        print('outlierLOF')
    elif _method == 2:
        print('clusterDBSCAN')
    elif _method == 3:
        print('clusterAP')
    elif _method == 4:
        print('Combined')
    else:
        print('Wrong Index!')
        return
    myDetect = AP_detect(_train_x=_train_x, _train_y=_train_y, _test_x=_test_x, _test_y=_test_y,
                         _altered_ap=_altered_ap, _dim=_dim, _num_subset=_num_subset, _neighboursLOF=_neighboursLOF,
                         _method=_method, _ifToDB = _ifToDB, _breakPointWeight=_breakPointWeight,
                         _neighboursDBSCAN=_neighboursDBSCAN, _filePath = filepath)
    myDetect.subsetResults()
    myDetect.multitest(test_group=_test_group, ifWeight=_ifWeight, ifPlot=_ifPlot, _eps=_eps, sizeRatio=_sizeRatio)
    myDetect.majorityWeighting(_min_num, _max_num, _certainty)
    # myDetect.dispersion_baseline()
    # Floor 6: subset Variance Baseline: Mean = 0.26504412913853403, StdVar = 0.19728690323433523
    # Floor 3: subset Variance Baseline: Mean = 0.8744270833766314, StdVar = 0.5903505219891764


def serverVersion():
    with open('allResults.txt','w') as f:
        roomNum = np.array([319, 625])
        for room_i in range(0, len(roomNum)):
            if room_i == 0:
                ap_all_num = 10
            else:
                ap_all_num = 9
            for ap_i in range(1, ap_all_num):
                filePath = "./repeat_scale_%d_No%dtoNo%daltered_%donly" % (roomNum[room_i], ap_i, ap_all_num, ap_all_num)
                data = scipy.io.loadmat(filePath)
                _train_x = data['train_x']
                _train_y = data['train_y']
                _test_x = data['test_x']
                _test_y = data['test_y']
                _altered_ap = np.array([2])  # follow the rule of Python
                _dim = 49
                _row_num = 10
                _num_subset = np.array([3,4,5])
                _test_group = 3
                _neighboursLOF = 91
                _neighboursDBSCAN = 4
                _method = 1  # 1: LOF, 2: DBSCAN, 3: AP
                _min_num, _max_num, _certainty = 5, 30, 0.9
                _eps = -5  # if _eps<0, then automatic.
                _breakPointWeight = 2.5
                _ifWeight, _ifPlot, _ifToDB = True, True, True
                myDetect = AP_detect(_train_x=_train_x, _train_y=_train_y, _test_x=_test_x, _test_y=_test_y,
                                     _altered_ap=_altered_ap, _dim=_dim, _num_subset=_num_subset,
                                     _neighboursLOF=_neighboursLOF, _method=_method, _ifToDB=_ifToDB,
                                     _breakPointWeight=_breakPointWeight, _neighboursDBSCAN=_neighboursDBSCAN)

                myDetect.subsetResults()
                myDetect.multitest(test_group=_test_group, ifWeight=_ifWeight, ifPlot=_ifPlot, _eps=_eps)
                ans, counter = myDetect.majorityWeighting(_min_num, _max_num, _certainty)
                f.write(filePath + "\nMethod:" + str(myDetect.method) + ", ans:" + str(ans) +
                        ", counter:" + str(counter) + "\n")
                myDetect.method = 2
                myDetect.subsetResults()
                myDetect.multitest(test_group=_test_group, ifWeight=_ifWeight, ifPlot=_ifPlot, _eps=_eps)
                ans, counter = myDetect.majorityWeighting(_min_num, _max_num, _certainty)
                f.write(filePath + "\nMethod:" + str(myDetect.method) + ", ans:" + str(ans) +
                        ", counter:" + str(counter) + "\n")
                myDetect.method = 3
                myDetect.subsetResults()
                myDetect.multitest(test_group=_test_group, ifWeight=_ifWeight, ifPlot=_ifPlot, _eps=_eps)
                ans, counter = myDetect.majorityWeighting(_min_num, _max_num, _certainty)
                f.write(filePath + "\nMethod:" + str(myDetect.method) + ", ans:" + str(ans) +
                        ", counter:" + str(counter) + "\n")

# serverVersion()


if __name__ == '__main__':
    main()
