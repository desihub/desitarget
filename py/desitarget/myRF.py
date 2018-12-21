"""
desitarget.myRF
===============

This module computes the Random Forest probability
and it stores the RF with our own persistency.
"""
import numpy as np
import sys


class myRF(object):
    """ Class for I/O operations and probability calculation for Random Forest
    """
    def __init__(self, data, modelDir, numberOfTrees=200, version=2):
        # loads the data once and initializes arrays
        self.data = data.copy()
        self.proba = np.zeros(len(data))
        self.bdtOutput = np.zeros(len(data))
        self.modelDir = modelDir
        self.version = version
        self.nTrees = numberOfTrees
        if self.version in [1, 2]:
            # print ("version is :",self.version)
            self.filesPerTree = 4  # for models-decals-dr3, (was 5 for models-decals)
        else:
            print("unsupported version=", self.version)
            sys.exit()

    def loadTree(self, treeFile, answerFile):
        # loads one tree and checks that the recursion limit is enough
        self.treeInfo = np.load(treeFile)
        self.treeAnswer = np.load(answerFile)
        if len(self.treeInfo) > sys.getrecursionlimit():
            sys.setrecursionlimit(int(len(self.treeInfo)*1.2))
            # print "WARNING recursion limit set to length(tree)*1.2 :",sys.getrecursionlimit()

    def unloadTree(self):
        # delete the current tree information to avoid memory leaks
        del self.treeInfo
        if self.version == 1:
            del self.treeAnswer

    def searchNodes(self, indices, nodeId=0):
        # recursively navigates in the tree and calculate the tree response
        nodeInfo = self.treeInfo[nodeId]

        # version without probability per leaf
#        if nodeInfo[0]==-1 :
#            if self.treeAnswer[nodeId,0,0]<self.treeAnswer[nodeId,0,1] :
#                score=1.
#            else :
#                score=0.
#            self.proba[indices]=score
#            return

        if nodeInfo[0] == -1:
            if self.version == 1:
                self.proba[indices] = self.treeAnswer[nodeId, 0, 1]*1./(self.treeAnswer[nodeId, 0, 0]+self.treeAnswer[nodeId, 0, 1])
            else:
                self.proba[indices] = nodeInfo[4]

            return

        leftChildId = nodeInfo[0]
        rightChildId = nodeInfo[1]
        feature = nodeInfo[2]
        threshold = nodeInfo[3]

        leftCond = (self.data[indices, feature] <= threshold)
        leftChildIndices = indices[leftCond]
#        rightCond = (self.data[indices,feature] > threshold)
        rightChildIndices = indices[leftCond is False]

        self.searchNodes(leftChildIndices, nodeId=leftChildId)
        self.searchNodes(rightChildIndices, nodeId=rightChildId)
        return

    def predict_proba(self):
        # calculate the forest response using the average response of the trees in the forest

        for iTree in np.arange(self.nTrees):
            # if iTree%10 == 0 : print ("tree=",iTree)
            self.loadTreeFromForest(iTree)
            self.searchNodes(np.arange(len(self.data)))
            self.bdtOutput += self.proba

        self.bdtOutput /= self.nTrees
        return self.bdtOutput

    def loadForest(self, forestFileName):
        # loads forest
        t = np.load(forestFileName, encoding='bytes')
        self.forest = t['arr_0']
        return

    def loadTreeFromForest(self, iTree):
        # loads one tree from the forest file and checks that the recursion limit is enough
        if self.version == 1:
            self.treeInfo = self.forest[iTree*2]
            self.treeAnswer = self.forest[iTree*2+1]
        elif self.version == 2:
            self.treeInfo = self.forest[iTree]
        else:
            print("unsupported version=", self.version)
            sys.exit()

        if len(self.treeInfo) > sys.getrecursionlimit():
            sys.setrecursionlimit(int(len(self.treeInfo)*1.2))
            # print "WARNING recursion limit set to length(tree)*1.2 :",sys.getrecursionlimit()

    def saveForest(self, forestFileName):
        # reads trees useful information and stores them in forestFileName

        def getFilledNumber(iFile):
            # just because fileNumber <10 have been padded with one 0 in scikit-learn
            if iFile < 10:
                return str(iFile).zfill(2)
            else:
                return str(iFile)

        forest = []

        for iTree in np.arange(self.nTrees):
            if iTree % 10 == 0:
                print("tree=", iTree)
            fileNumber = (iTree*self.filesPerTree+4)
            treeFile = self.modelDir+"bdt.pkl_"+getFilledNumber(fileNumber)+".npy"
            answerFile = self.modelDir+"bdt.pkl_"+getFilledNumber(fileNumber-1)+".npy"

            # Store only useful information
            newt = None
            t = np.load(treeFile)
            a = np.load(answerFile)
            if self.version == 1:
                newt = np.zeros(len(t), dtype='int16, int16, int8, float32')
            elif self.version == 2:
                newt = np.zeros(len(t), dtype='int16, int16, int8, float32, float32')
            else:
                pass

            for i in np.arange(len(t)):
                temp_t = t[i]
                if self.version == 1:
                    tup = (temp_t[0], temp_t[1], temp_t[2], temp_t[3])
                elif self.version == 2:
                    temp_a = a[i]
                    proba = temp_a[0, 1]/(temp_a[0, 0]+temp_a[0, 1])
                    tup = (temp_t[0], temp_t[1], temp_t[2], temp_t[3], proba)
                newt[i] = tup

            if self.version == 1:
                forest.append(newt)
                forest.append(np.load(answerFile))
            elif self.version == 2:
                forest.append(newt)
            else:
                pass

            del newt

        np.savez_compressed(forestFileName, forest)
        return
