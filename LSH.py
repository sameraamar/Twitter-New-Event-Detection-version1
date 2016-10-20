# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""
#import pprint
import numpy as np
from scipy import spatial
import math
import scipy.sparse as sparse

def check(v1,v2, is_sparse):
    if not is_sparse and len(v1) != len(v2):
        raise ValueError( "Not maching lengths of ", v1, v2)
      
        
        
def dotProduct(v1, v2, is_sparse):
    check(v1, v2, is_sparse)
    if is_sparse:
        s = 0
        if len(v1) > len(v2): # go over the smaller list
            t = v1
            v1 = v2
            v2 = t
            
        for k in v1:
            s += v1[k] * v2[k]

        return s
    else:
        return np.dot(v1, v2)

def distance_cosine(a,b, is_sparse):
    if is_sparse:
        m1 = max(a.values())
        m2 = max(b.values())
        m = max(m1, m2)
        tmp = [0]*m
        for k in a:
            tmp[k] = a[k]
        a = tmp

        tmp = [0]*m
        for k in b:
            tmp[k] = b[k]
        b = tmp
        
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
    is_sparse = False
    
    d = 0
    
    def __init__(self, maxBucketSize, dimensionSize, hyperPlanesNumber, is_sparse=False):
        self.hyperPlanesNumber = hyperPlanesNumber
        self.maxBucketSize = maxBucketSize
        self.buckets = {}
        self.total = 0
        self.is_sparse = is_sparse
        self.d = hyperPlanesNumber
        self.hyperPlanes = self.randomHyperPlanes( dimensionSize, hyperPlanesNumber)
        
    
    def add(self, ID, point):
        self.total += 1
        
        if not self.is_sparse:
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
            dist = distance_cosine(k['point'], point, self.is_sparse)
            if dist < minDist:
                #k['similar_count'] += 1
                #item['similar_count'] += 1
                minDist = dist
                item['closest'] = k['ID']
        item['dist'] = minDist
            
            
        b.append( item ) 
        
        if len(b) > self.maxBucketSize:
            b = b[1:]
            self.total -= 1
        
        self.buckets[hashcode] = b
        
        #print(self.maxBucketSize)
        return item

    def generateHashCode(self, point):
        hashcode = ''
        for hyperplane in self.hyperPlanes:
            if(dotProduct(point, hyperplane, self.is_sparse) < 0):
                hashcode += '0'
            else:
                hashcode += '1'
        return hashcode


    def randomHyperPlanes(self, dimSize, hyperPlanesNumber):
#        planes = []
#        for i in range(hyperPlanesNumber):
#            v = np.random.normal(size = dimSize)
#            #v = v/np.sqrt(np.sum(v**2))
#            planes.append(v)
            
        if self.is_sparse:
            m = min(dimSize/2, np.random.randint(100))
            xr = range(dimSize)
            planes = []
            for i in range(hyperPlanesNumber):
                tmp = np.random.choice(xr, size=m)
                plane = {}
                for t in tmp:
                    plane[t] = 1
                planes.append(plane)
        else:
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
        print('total: ', self.total, 'number of buckets:', len(self.buckets), 'max bucke size', self.maxBucketSize, 'hyperPlanesNumber', self.hyperPlanesNumber)
        lengths = [len(self.buckets[b]) for b in self.buckets]
        print('number of items in each bucket: ', lengths)
        #pprint.pprint(self.buckets)
     
class LSH:
    """LSH class"""
    dimSize = 3
    numberTables = None
    hList = None
    is_sparse = None
    nearestNeighbor = {}

    def __init__(self, dimensionSize ,hyperPlanesNumber=40, numberTables=4, maxBucketSize=10, is_sparse=False):
        self.is_sparse = is_sparse
        self.dimSize = dimensionSize
        self.numberTables = numberTables
        self.hList = [HashtableLSH(maxBucketSize, dimensionSize, hyperPlanesNumber, is_sparse=self.is_sparse) for i in range(numberTables)]

    def myprint(self):
        for h in self.hList:
            print('*******************************************')
            h.myprint()
        print('dimenion: ', self.dimSize, 'tables:', self.numberTables, self.hList[0].size() )
            
    
    def add(self, ID, point):
        """add a point to the hash table
        the format of the point is assumed to be parse so it will be in libSVM format
        json {word:count, word:cout}"""
        for table in self.hList:
            table.add(ID, point)
        

    def distance(self, p1, p2):
        dmin = None
        for h in self.hList:
            d = h.distance_aprox(p1, p2)
            if dmin == None or dmin>d:
                dmin = d
                
        return dmin
        
    def hash_similarity(self, p1, p2):
        return 1-self.distance(p1, p2)
            

    def evaluate(self, nruns):
        avg = 0
        
        for run in range(nruns):
            if not self.is_sparse:
                p1 = np.random.randn(self.dimSize)
                p2 = np.random.randn(self.dimSize)        
            
            hash_sim = self.hash_similarity(p1, p2) 
            true_sim = angular_similarity(p1, p2)
            diff = abs(hash_sim-true_sim)/true_sim
            avg += diff
            print ('true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) )
        print ('avg diff' , avg / nruns)
   
#%%        
        
def test1():
    n = 5
    d = 2**n
    dim = 3
    maxB = 50
    tables = 10
    ll = LSH(dimensionSize=dim, numberTables=tables, hyperPlanesNumber=d, maxBucketSize=maxB)
   
    ID = 1

    nruns = 100
    mu, sigma = 3, 0.1 # mean and standard deviation
    
    for run in range(nruns):
        p = np.random.randn(dim)
        #p = np.random.normal(mu, sigma, dim)
        ll.add(ID, p)
        ID += 1
        perc = 100.0*run/nruns
        if perc % 10 == 0:
            print '%.2f' % perc, '%'

            
    
    ll.myprint()

    
    sss = ll.hList[4]
    bbb = sss.buckets
    
    s = 0
    for b in bbb:
        s += len( bbb[b ])
    print s
    
    s = 0 
    for x in bbb.values():
        s += len(x)
    print s
    
    
    print('total: ', sss.total, 'number of buckets:', len(sss.buckets), ' max bucke size', sss.maxBucketSize, 'hyperPlanesNumber', sss.hyperPlanesNumber)
    lengths = [len(sss.buckets[b]) for b in sss.buckets]
    print('number of items in each bucket: ', lengths)
    
    
    ll.evaluate(nruns)

    #
    #p1 = [2.88844345,  2.895764  ,  2.84288212]
    #p2 = [3.21012159,  3.04687657,  2.62343952]
    #p3 = [2.89465061,  -2.0,  2.51265069]
    #
    #print ll.hList[0].distance_aprox(p1, p3)
    #print ll.hList[0].distance_aprox(p3, p1)
    #print ll.hList[0].distance_aprox(p1, p3)
    #print ll.hList[0].distance_aprox(p3, p1)

#%%
def test2():
    n = 5
    d = 2**n
    dim = 3
    maxB = 50
    tables = 10
    nruns = 7

    sparse.rand(5, 5, density=0.1)
    
    ll = LSH(dimensionSize=dim, numberTables=tables, hyperPlanesNumber=d, maxBucketSize=maxB, is_sparse=True)

    ll.evaluate(nruns)

    
    ll.myprint()
    
    

from scipy.sparse import *
from scipy import *
from scipy import stats

def randomPoint(dim):
    rvs = stats.randint(low = 1, high = 5).rvs #.norm(scale=2, loc=0).rvs
    #S = sparse.random(1, dim, density=0.25, data_rvs=rvs)
    S = sparse.random(1, dim, format='coo', density=0.25, data_rvs=rvs)
    #stats.poisson(10, loc=0).rvs
    #rvs = stats.randint.stats(0, 100, moments='mvsk')
    print S, '\n', S.getnnz()

if __name__ == '__main__':
    test1()
    #test2()
    #randomPoint(100)
#    S = dok_matrix((5,5), dtype=int32)
#    for i in range(5):
#        for j in range(5):
#            S[i,j] = i+j # Update element
    
    