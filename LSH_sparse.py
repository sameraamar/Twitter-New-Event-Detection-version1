# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""


import math
from scipy import sparse 
#from scipy import *
import time
from scipy import spatial
from scipy import stats
import numpy as np


#%%

DEBUG = 3
INFO = 2
WARN = 1
ERROR = 0

log_level = ERROR
logger=None

def log(level, function, message):
    global logger
    if level>log_level:
        return
    t = time.asctime()
    text = str(t) + ' (' + str(level) + ') ' + function + ': ' + str(message) + '\n'

    if logger==None:
        logger = open('c:/temp/LSH.log', 'w')

    logger.write(text)

def debug(function, message):
    log(DEBUG, function, message)
    
def setLogLevel(loglevel=WARN):
    global log_level 
    log_level = loglevel

setLogLevel(DEBUG)
    
#%%


def randomPoint(dim):
    rvs = stats.randint(low = 1, high = 5).rvs #.norm(scale=2, loc=0).rvs
    #rvs =  numpy.random.randn
    #S = sparse.random(1, dim, density=0.25, data_rvs=rvs)
    S = sparse.random(1, dim, format='csr', density=1.0, data_rvs=rvs)
    
    #stats.poisson(10, loc=0).rvs
    #rvs = stats.randint.stats(0, 100, moments='mvsk')
    return S
    
      
'''compute norm of a sparse vector'''
def norm(x):
    sum_sq=x.dot(x.T)[0,0]
    
    norm=np.sqrt(sum_sq)
    return(norm)
        
def randomVector(dim, normalize=True):
    rvs = stats.randint(low = -5, high = 5).rvs #.norm(scale=2, loc=0).rvs
    #S = sparse.random(1, dim, density=0.25, data_rvs=rvs)
    P = sparse.random(1, dim, format='csr', density=1.0, data_rvs=rvs)
    
    if normalize:
        P = P / norm(P)
    return P
    

#%%


def check(v1,v2, is_sparse):
    if not is_sparse and v1.size != v2.size:
        raise ValueError( "Not maching lengths of ", v1, v2)

def dotProduct(v1, v2, is_sparse):
    base = time.time()
    check(v1, v2, is_sparse)
    if is_sparse:
        res = v1.dot(v2.T)[0,0]
    else:
        res = np.dot(v1, v2)
    debug( 'dotProduct', 1000.0*(time.time()-base))
    
    return res

def distance_cosine(a,b): 
    base = time.time()
    print a.shape, b.shape
    res = spatial.distance.cosine(a, b)
    debug( 'distance_cosine', 1000.0*(time.time()-base))
    return res
    
# angular similarity using definitions
# http://en.wikipedia.org/wiki/Cosine_similarity
def angular_similarity(a,b,is_sparse):
    if is_sparse:
        dot_prod = a.dot(b.T)[0,0]
        sum_a = norm(a)
        sum_b = norm(b)
    else:
        dot_prod = np.dot(a,b)
        sum_a = sum(a**2) **.5
        sum_b = sum(b**2) **.5
    cosine = dot_prod/(sum_a*sum_b) # cosine similarity
    if cosine>1.0 and cosine<1.001:
        cosine = 1;
        
    theta = math.acos(cosine)
    dist = (theta/math.pi)
    return 1.0-dist

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
        base = time.time()
        
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
            self.total -= 1
        
        self.buckets[hashcode] = b
        
        debug( 'add', 1000*(time.time()-base))
        return item

    def generateHashCode(self, point):
        base = time.time()
        hashcode = ''
        for hyperplane in self.hyperPlanes:
            if(dotProduct(point, hyperplane, self.is_sparse) < 0):
                hashcode += '0'
            else:
                hashcode += '1'
        debug( 'generateHashCode', 1000*(time.time()-base))
        return hashcode


    def randomHyperPlanes(self, dimSize, hyperPlanesNumber):
        base = time.time()
        planes = []
#        for i in range(hyperPlanesNumber):
#            v = np.random.normal(size = dimSize)
#            #v = v/np.sqrt(np.sum(v**2))
#            planes.append(v)
            
        if self.is_sparse:
            for i in range(self.hyperPlanesNumber):
                plane = randomVector(dimSize)
                planes.append(plane)
        else:
            planes = np.random.randn(hyperPlanesNumber, dimSize)
        debug( 'randomHyperPlanes', 1000*(time.time()-base))
        return planes
   
    def size(self):
        return self.total

    def distance_aprox(self, p1, p2):
        h1 = self.generateHashCode(p1)
        h2 = self.generateHashCode(p2)
        
        #xor = h1^h2
        nnz_xor = 0
        for (c1,c2)  in zip(h1, h2):
            if c1!=c2 and (c1=='1' or c2=='1'):
                nnz_xor += 1
        
        d = self.hyperPlanesNumber
        res = (d-nnz_xor)/float(d)
        return 1.0 - res
        
#        #p1 [p1 != 0] = 1.0
#        
#        #print h1, h2, zip(h1, h2)
#        count = 0
#        for (c1,c2)  in zip(h1, h2):
#            if c1 == c2:
#                count += 1
#                
#        res = float(count) / len(h1)
#        return 1-res
        
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
        dist = None
        for h in self.hList:
            d = h.distance_aprox(p1, p2)
            if dist == None or dist>d:
                dist = d
            
                
#        h = self.hList[  np.random.randint( len(self.hList) ) ]
#        dmax = h.distance_aprox(p1, p2)
        
        return dist
        
    def hash_similarity(self, p1, p2):
        return 1-self.distance(p1, p2)
            

    def evaluate(self, nruns):
        avg = 0
        
        for run in range(nruns):
            if not self.is_sparse:
                p1 = np.random.randn(self.dimSize)
                p2 = np.random.randn(self.dimSize)        
            else:
                p1 = randomPoint(self.dimSize)        
                p2 = randomPoint(self.dimSize)   
            
            hash_sim = self.hash_similarity(p1, p2) 
            true_sim = angular_similarity(p1, p2,self.is_sparse)
            diff = abs(hash_sim-true_sim)/true_sim
            avg += diff
            print ('true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) )
        print ('avg diff' , avg / nruns)
   
#%%        
        
def test1():
    n = 10
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
    n = 3
    d = 2**n
    dim = 3
    maxB = 50
    tables = 20
    nruns = 70

    #sparse.rand(5, 5, density=0.1)
    before = time.time()
    print before
    ll = LSH(dimensionSize=dim, numberTables=tables, hyperPlanesNumber=d, maxBucketSize=maxB, is_sparse=True)
    after = time.time()

    print 'time: ', 1000*( after-before)
    
    ll.evaluate(nruns)

    ID = 0
    for run in range(nruns):
        p = randomPoint(dim)
        #p = np.random.normal(mu, sigma, dim)
        
        ll.add(ID, p)
        ID += 1
        perc = 100.0*run/nruns
        if perc % 10 == 0:
            print '%.2f' % perc, '%'

    #ll.myprint()
    
    


if __name__ == '__main__':
    test2()
    #test2()
    #p = randomPoint(100)
    #print p
    #S = dok_matrix((50,50), dtype=int32)
#    print 'Point:'
#    print randomPoint(50)
#    print 'Vector:'
#    print randomVector(50)
#    for i in range(5):
#        for j in range(5):
#            S[i,j] = i+j # Update element
#    print S, S.getnnz()
    