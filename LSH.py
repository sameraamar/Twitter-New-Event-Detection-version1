# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""
#import pprint
import numpy as np
from scipy import spatial
import math

def check(v1,v2):
    if len(v1) != len(v2):
        raise ValueError( "Not maching lengths of ", v1, v2)
    pass
      
def dotProduct(v1, v2):
    check(v1, v2)
    return np.dot(v1, v2)

def distance_cosine(a,b):
    res = 1-spatial.distance.cosine(a, b)
    
    return res
    
# angular similarity using definitions
# http://en.wikipedia.org/wiki/Cosine_similarity
def angular_similarity(a,b):
    dot_prod = np.dot(a,b)
    sum_a = sum(a**2) **.5
    sum_b = sum(b**2) **.5
    cosine = dot_prod/sum_a/sum_b # cosine similarity
    theta = math.acos(cosine)
    return 1.0-(theta/math.pi)

class HashtableLSH:
    maxBucketSize = None
    hyperPlanesNumber = None
    hyperPlanes = None
    buckets = None
    total = 0
    
    d = 0
    
    def __init__(self, maxBucketSize, dimensionSize, hyperPlanesNumber):
        self.hyperPlanesNumber = hyperPlanesNumber
        self.hyperPlanes = self.randomHyperPlanes( dimensionSize, hyperPlanesNumber)
        self.maxBucketSize = maxBucketSize
        self.buckets = {}
        self.total = 0
        self.d = hyperPlanesNumber
    
    def add(self, ID, point):
        self.total += 1
        
        point = np.asarray(point)
        
        hashcode = self.generateHashCode(point)

        item = {}
        item['ID'] = ID
        item['point'] = point
        item['closest'] = None

        #item['similar_count'] = 0
        b = self.buckets.get(hashcode, [])
        #search for closest item
        minDist = 1
        for k in b:
            dist = distance_cosine(k['point'], point)
            if dist < minDist:
                #k['similar_count'] += 1
                #item['similar_count'] += 1
                minDist = dist
                item['closest'] = k['ID']
        item['dist'] = minDist
            
            
        b.append( item ) 
        
        if len(b) > self.maxBucketSize:
            b = b[1:]
        
        self.buckets[hashcode] = b
        
        #print(self.maxBucketSize)
        return item

    def generateHashCode(self, point):
        hashcode = ''
        for hyperplane in self.hyperPlanes:
            if(dotProduct(point, hyperplane) < 0):
                hashcode += '0'
            else:
                hashcode += '1'
        return hashcode


    def randomHyperPlanes(self, dimSize, hyperPlanesNumber):
        planes = []
        for i in range(hyperPlanesNumber):
            v = np.random.normal(size = dimSize)
            #v = v/np.sqrt(np.sum(v**2))
            planes.append(v)
            
        planes = np.random.randn(hyperPlanesNumber, dimSize)
        return planes
   
    def size(self):
        return self.total

    def distance_aprox(self, p1, p2):
        h1 = self.generateHashCode(p1)
        h2 = self.generateHashCode(p2)
        
        #print h1, h2, zip(h1, h2)
        count = 0
        for (c1,c2)  in zip(h1, h2):
            if c1 == c2:
                count += 1
                
        res = float(count) / len(h1)
        return 1-res
        
    def myprint(self):
        print('total: ', self.total, 'max bucke size', self.maxBucketSize, 'hyperPlanesNumber', self.hyperPlanesNumber)
        lengths = [len(self.buckets[b]) for b in self.buckets]
        print(lengths)
        #pprint.pprint(self.buckets)
     
class LSH:
    """LSH class"""
    dimSize = 3
    numberTables = None
    hList = None
    nearestNeighbor = {}

    def __init__(self, dimensionSize ,hyperPlanesNumber=40, numberTables=4, maxBucketSize=10):
        self.dimSize = dimensionSize
        self.numberTables = numberTables
        self.hList = [HashtableLSH(maxBucketSize, dimensionSize, hyperPlanesNumber) for i in range(numberTables)]

    def myprint(self):
        for h in self.hList:
            h.myprint()
        print('dimenion: ', self.dimSize, 'tabels:', self.numberTables, self.hList[0].size() )
            
    def add(self, ID, point):
        for table in self.hList:
            table.add(ID, point)
        

    def distance(self, p1, p2):
        i = np.random.randint( len(self.hList) )
        return self.hList[i].distance_aprox(p1, p2)
        
    def hash_similarity(self, p1, p2):
        return 1-self.distance(p1, p2)
            

    def evaluate(self, nruns):
        avg = 0
        
        for run in range(nruns):
            p1 = np.random.randn(self.dimSize)
            p2 = np.random.randn(self.dimSize)        
            
            hash_sim = self.hash_similarity(p1, p2) 
            true_sim = angular_similarity(p1, p2)
            diff = abs(hash_sim-true_sim)/true_sim
            avg += diff
            print ('true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) )
        print ('avg diff' , avg / nruns)
        
        
#%%
##examples...

nruns = 1000

n = 8
d = 2**n
dim = 3
ll = LSH(dimensionSize=dim, numberTables=10, hyperPlanesNumber=d, maxBucketSize=70)

ID = 1
for run in range(nruns):
    p = np.random.randn(dim)
    ll.add(ID, p)
    ID += 1

p1 = [3.15, 1, -3.2]
ll.add(ID, p1)
ID += 1

p1 = [3.16, 3.14, -3.2]
ll.add(ID, p1)
ID += 1

p1 = [3.1, 3.1, -3.21]
ll.add(ID, p1)
ID += 1



ll.myprint()

#%%
#import math

ll.evaluate(nruns)

#p1 = [3.15, 1, -3.2]
#p2 = [3.16, 3.14, -3.2]
#
#p1 = np.asarray(p1)
#p2 = np.asarray(p2)
#
#print p1, ', ', p2, ',', 
#
#dist = ll.distance(p1, p2) 
#print 'hash distance: %.10f, ' % dist, 
#
#dist = distance_cosine(p1, p2)
#print 'distance_cosine: %.10f, ' % dist, 
#
#dist = angular_similarity(p1, p2)
#print 'angular_similarity: %.10f' % dist


