# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""
import pprint

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
        
        item = {}
        item['point'] = point
        #item['code'] = hashcode
        b = self.buckets.get(hashcode, [])
        #search for closest item
        b.append( item ) 
        self.buckets[hashcode] = b
        
        print(self.maxBucketSize)
        return True
        
    def size(self):
        return self.total
        
    def print(self):
        print('total: ', self.total, 'max bucke size', self.maxBucketSize)
#        for b in self.buckets:
#            print(b)
        pprint.pprint(self.buckets)
     
import numpy as np

class LSH:
    #from numpy import dot
    """A simple example class"""
    dimSize = 3
    hyperPlanesNumber = None
    hyperPlanes = None
    numberTables = None
    hList = None


    def randomHyperPlanes(self, dimSize, hyperPlanesNumber):
        planes = []
        for i in range(hyperPlanesNumber):
            v = np.random.normal(size = dimSize)
            #v = v/np.sqrt(np.sum(v**2))
            planes.append(v)
        return planes
    
    def __init__(self, dimensionSize ,hyperPlanesNumber=40, numberTables=4, maxBucketSize=10):
        self.hyperPlanesNumber = hyperPlanesNumber
        self.dimSize = dimensionSize
        self.numberTables = numberTables
        self.hList = [HashtableLSH(maxBucketSize) for i in range(numberTables)]
        
#        planes = []
#        for i in range(hyperPlanesNumber):
#            v = np.random.normal(size = dimensionSize)
#            #v = v/np.sqrt(np.sum(v**2))
#            planes.append(v)  
            
        self.hyperPlanes = self.randomHyperPlanes( dimensionSize, hyperPlanesNumber)

    def generateHashCode(self, point):
        hashcode = ''
        for hyperplane in self.hyperPlanes:
            if(self.dotProduct(point, hyperplane) < 0):
                hashcode += '0'
            else:
                hashcode += '1'
        return hashcode


    def check(sf, v1,v2):
        if len(v1) != len(v2):
            raise ValueError( "Not maching lengths of ", v1, v2)
        pass

    def print(self):
        print('dimenion: ', self.dimSize, 'tabels:', self.numberTables, 'hyperPlanesNumber', self.hyperPlanesNumber)
        for h in self.hList:
            h.print()
            
    def dotProduct(self, v1, v2):
        self.check(v1, v2)
        return np.dot(v1, v2)
        

    def add(self, point):
        for table in self.hList:
            code = self.generateHashCode(point)
            #print (len(self.hList) , ' hash tables ', self.hyperPlanesNumber, ' hyperplanes')
            table.add(point, code)
        return 'hello world'

#    def init(self, hyperPlanesNumber=40):
#        hyperPlanesNumber = self.hyperPlanesNumber
#        print(self.hyperPlanesNumber)
#        return 'hello world'

ll = LSH(3, numberTables=10, hyperPlanesNumber=30)
ll.add([3, 3, 3])
ll.add([3.1, 3.2, 3.2])
ll.add([3.1, 3.2, -3.2])

ll.print()