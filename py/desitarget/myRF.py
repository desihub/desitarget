"""
desitarget.myRF
===============

This module does something.
"""
import numpy as np
import sys

class myRF(object):
    """Great, this class has no documentation whatsoever.
    """
    def __init__(self,data,modelDir,version=1):
    # loads the data once and initializes arrays
        self.data  = data.copy()
        self.proba = np.zeros(len(data))
        self.bdtOutput = np.zeros(len(data))
        self.modelDir = modelDir
        self.version = version

    def loadTree(self,treeFile,answerFile):
    # loads one tree and checks that the recursion limit is enough
        self.treeInfo   = np.load(treeFile)
        self.treeAnswer = np.load(answerFile)
        if len(self.treeAnswer)>sys.getrecursionlimit() :
            sys.setrecursionlimit(int(len(self.treeAnswer)*1.2))
            #print "WARNING recursion limit set to length(tree)*1.2 :",sys.getrecursionlimit()

    def unloadTree(self):
    # delete the current tree information to avoid memory leaks
        del self.treeInfo
        del self.treeAnswer

    def searchNodes(self,indices,nodeId=0) :
    # recursively navigates in the tree and calculate the tree response
        nodeInfo=self.treeInfo[nodeId]
#        print nodeInfo
        if nodeInfo[0]==-1 :
            if self.treeAnswer[nodeId,0,0]<self.treeAnswer[nodeId,0,1] :
                score=1.
            else :
                score=0.
            self.proba[indices]=score
            return

        leftChildId  = nodeInfo[0]
        rightChildId = nodeInfo[1]
        feature      = nodeInfo[2]
        threshold    = nodeInfo[3]

        leftCond = (self.data[indices,feature] <= threshold)
        leftChildIndices  = indices[leftCond]
#        rightCond = (self.data[indices,feature] > threshold)
        rightChildIndices = indices[leftCond==False]

        self.searchNodes(leftChildIndices,nodeId=leftChildId)
        self.searchNodes(rightChildIndices,nodeId=rightChildId)
        return

    def predict_proba(self,nTrees=200) :
    # calculate the forest response using the average response of the trees in the forest

        for iTree in np.arange(nTrees) :
            #if iTree%10 == 0 : print ("tree=",iTree)
            self.loadTreeFromForest(iTree)
            self.searchNodes(np.arange(len(self.data)))
            self.bdtOutput+=self.proba

        self.bdtOutput/=nTrees
        return self.bdtOutput

    def loadForest(self,forestFileName,nTrees=200,version=1) :
    # loads forest
        t = np.load(forestFileName,encoding='bytes')
        self.forest = t['arr_0']
        return

    def loadTreeFromForest(self,iTree):
    # loads one tree from the forest file and checks that the recursion limit is enough
        self.treeInfo   = self.forest[iTree*2]
        self.treeAnswer = self.forest[iTree*2+1]
        if len(self.treeAnswer)>sys.getrecursionlimit() :
            sys.setrecursionlimit(int(len(self.treeAnswer)*1.2))
            #print "WARNING recursion limit set to length(tree)*1.2 :",sys.getrecursionlimit()

    def saveForest(self,forestFileName,nTrees=200,version=1) :
    # reads trees useful information and stores them in forestFileName

        if self.version==1:
            filesPerTree = 4 # for models-decals-dr3, (was 5 for models-decals)
        else :
            print ("unsupported version=",self.version)
            sys.exit()

        forest = []

        for iTree in np.arange(nTrees) :
            if iTree%10 == 0 : print ("tree=",iTree)
            fileNumber = (iTree*filesPerTree+4)
            if iTree<2 :
                treeFile   = self.modelDir+"bdt.pkl_"+str(fileNumber).zfill(2)+".npy"
                answerFile = self.modelDir+"bdt.pkl_"+str(fileNumber-1).zfill(2)+".npy"
            else :
                treeFile   = self.modelDir+"bdt.pkl_"+str(fileNumber)+".npy"
                answerFile = self.modelDir+"bdt.pkl_"+str(fileNumber-1)+".npy"

            forest.append(np.load(treeFile))
            forest.append(np.load(answerFile))
        np.savez_compressed(forestFileName,forest)
        return
