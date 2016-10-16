# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""


class HashtableLSH:
    maxBucketSize = None
    buckets = None
    total = 0
    
    def __init__(self, maxBucketSize):
        self.maxBucketSize = maxBucketSize
        self.buckets = {}
        self.total = 0
    
    def add(self, point, hashcode):
        self.total += 1
        
        self.buckets[hashcode].append( point )      
        
        print(self.maxBucketSize)
        return True
        
    def size(self):
        return self.total
        
class LSH:
    from numpy import dot
    """A simple example class"""
    hyperPlanesNumber = None
    numberTables = None
    hList = None

    
    def __init__(self, hyperPlanesNumber=40, maxBucketSize=1000):
        self.hyperPlanesNumber = hyperPlanesNumber
        self.hList = [HashtableLSH(maxBucketSize) for i in range(hyperPlanesNumber)]
        

    def generateHashCode(self, point):
        hashcode = ''
        for hyperplane in self.hyperplanes:
            if(self.dotProduct(point, hyperplane) < 0):
                hashcode.append('0')
            else:
                hashcode.append('1')
        return hashcode


    def check(v1,v2):
        if len(v1) != len(v2):
            raise ValueError, "Not maching lengths of "
        pass

    def dotProduct(self, v1, v2):
        #check(v1, v2)
        return dot(v1, v2)
        

    def add(self, points):
        print (len(self.hList) , ' hash tables ', self.hyperPlanesNumber, ' hyperplanes')
        self.hList[0].add(None)
        return 'hello world'

#    def init(self, hyperPlanesNumber=40):
#        hyperPlanesNumber = self.hyperPlanesNumber
#        print(self.hyperPlanesNumber)
#        return 'hello world'

ll = LSH(hyperPlanesNumber=30)
ll.add(None)
