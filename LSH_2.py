# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""


import math
from scipy import sparse
from scipy import stats
import numpy as np
import multirun

def debug(logger, message):
    logger.debug(message)

        
def info(logger, message):
    logger.info(message)

#%%

#DEBUG = 3
#INFO = 2
#WARN = 1
#ERROR = 0
#
#log_level = ERROR
#logger=None
#
#def log(level, function, message):
#    global logger
#    if level>log_level:
#        return
#    t = time.asctime()
#    text = str(t) + ' (' + str(level) + ') ' + function + ': ' + str(message) + '\n'
#
#    if logger==None:
#        logger = open('c:/temp/LSH.log', 'w')
#
#    logger.write(text)
#
#def debug(function, message):
#    log(DEBUG, function, message)
#    
#def setLogLevel(loglevel=WARN):
#    global log_level 
#    log_level = loglevel
#
#setLogLevel(DEBUG)
    
#%%

cache_values = {}


class MathHelper:
    logger = None

    def __init__(self, logger):
        self.logger = logger

    def clear(self):
        global norm_values 
        norm_values = {}

    def randomPoint(self, features):
        return self.randomSamples(1, features)
        
    def randomSamples(self, samples, features):
        rvs = stats.randint(low = 1, high = 5).rvs #.norm(scale=2, loc=0).rvs
        #rvs =  numpy.random.randn
        #S = sparse.random(1, dim, density=0.25, data_rvs=rvs)
        den = min(20.0/(features*samples), 0.9)
        S = sparse.random(samples, features, format='lil', density=den, data_rvs=rvs)
        
        #stats.poisson(10, loc=0).rvs
        #rvs = stats.randint.stats(0, 100, moments='mvsk')
        return S
        
        
          
    '''compute norm of a sparse vector'''
    def norm(self, id1, x):
        global cache_values
        norm1 = norm_values.get(id1, None)
        
        #self.logger.debug('Calculating norm: {0}: --> {1}:\n{2}\n'.format(id1, norm1, x))
        
        if norm1!=None:
            return norm1
            
        sum_sq=x.dot(x.T)[0,0]
        norm2=np.sqrt(sum_sq)
        
        cache_values[id1] = norm2

#        if norm1==None:
#            norm1 = norm2
#            
#        assert(norm1 == norm2)
        
        return(norm2)
            
#    def randomVector(self, dim, normalize=True):
#        rvs = stats.randint(low = -5, high = 5).rvs #.norm(scale=2, loc=0).rvs
#        #S = sparse.random(1, dim, density=0.25, data_rvs=rvs)
#        P = sparse.random(1, dim, format='lil', density=0.80, data_rvs=rvs)
#        
#        if normalize:
#            P = P / self.norm(P)
#        return P
    
    def dotProduct(self, id1, id2, v1, v2):
        key1 = '{0}.{1}'.format(id1, id2)
        key2 = '{1}.{0}'.format(id1, id2)
        
        global cache_values
        
        dot = norm_values.get(key1, norm_values.get(key2, None))
        if dot != None:
            self.logger.debug('Pingo: {0}*{1} = {2}'.format(id1, id2, dot))
            return dot
        
        res = v1.dot(v2.T)[0,0]
        cache_values[key1] = res
        return res
    
    def distance_cosine(self, id1, id2, a,b):
        res = self.angular_distance(id1, id2, a, b)
        return res
        
    #    base = time.time()
    #    #a= a.toarray()
    #    #b = b.toarray()
    #    res = spatial.distance.cosine(a, b)
    #    return res
    
    # angular similarity using definitions
    # http://en.wikipedia.org/wiki/Cosine_similarity
    def angular_distance(self, id1, id2, a,b):
        key1 = '{0}-{1}'.format(id1, id2)
        key2 = '{1}-{0}'.format(id1, id2)
        
        global cache_values
        
        cache_dist = cache_values.get(key1, cache_values.get(key2, None))
        if cache_dist != None:
            self.logger.debug('Pingo: {0}-{1} = {2}'.format(id1, id2, cache_dist))
            return cache_dist
        
        m = 0
        nonzeros = np.nonzero(a)
        if nonzeros != None:
            for i in nonzeros[1]:
                m += a[0, i]*b[0, i]
    
        dot_prod = self.dotProduct(id1, id2, a, b) #a.dot(b)[0,0]
        
        sum_a = self.norm(id1, a)
        sum_b = self.norm(id2, b)
    
        cosine = dot_prod/(sum_a*sum_b) # cosine similarity

        #rounding issue!
        if cosine>1.0 and cosine<1.001:
            cosine = 1;
        try:
            theta = math.acos(cosine)
        except :
            print ('a: ', a)
            print ('id1: ', id1)
            print ('b: ', b)
            print ('id2: ', id2)
            print ('cosine: ', cosine)
            
            if cosine>1.0 and cosine<1.001:
                print ('within epsilon')
            else:
                print ('more than 1')
                
            print (norm_values)
            raise Exception("Error!")
        
        dist = (theta/math.pi)
        
        cache_values[key1] = dist

        return dist
        
    def angular_similarity(self, id1, id2, a,b):
        return 1.0-self.angular_distance(id1, id2, a,b)

class HashtableLSH:
    maxBucketSize = None
    hyperPlanesNumber = None
    hyperPlanes = None
    buckets = None
    total = 0
    zeros = ''
    d = 0
    logger = None
    hashcodes = None
    hashcodes2 = None
    helper = None
    doc_indices = {}
    
    def __init__(self, maxBucketSize, dimensionSize, hyperPlanesNumber, logger, helper):
        self.hyperPlanesNumber = hyperPlanesNumber
        self.maxBucketSize = maxBucketSize
        self.buckets = {}
        self.total = 0
        self.d = hyperPlanesNumber
        self.logger = logger
        self.helper = helper
        self.hyperPlanes = self.randomHyperPlanes( dimensionSize, hyperPlanesNumber)
        for i in range(self.hyperPlanesNumber):
            self.zeros += '0'
        
    '''item is in the same structure as returned by add function'''
    def findNearest(self, item):
        self.logger.entry('HashtableLSH.findNearest')
        bucket = self.buckets[item['hashcode']]

        nearest = None
        minDist = None
        bucket_size = len(bucket)
        for neighbor in bucket:
            if neighbor['ID'] == item['ID']:
                continue
            p1 = item['point']
            p2 = neighbor['point']

            dist = 1.0-self.helper.distance_cosine(item['ID'], neighbor['ID'], p1, p2)
            if minDist == None or dist < minDist:
                minDist = dist
                nearest = neighbor

        self.logger.exit('HashtableLSH.findNearest')
        return nearest, minDist, bucket_size

    def generateHashCode(self, point, id1):
        self.logger.entry('HashtableLSH.generateHashCode')
        
        if id1==None or len(self.doc_indices)==0:
            assert(False)
            hashcode = point.dot(self.hyperPlanes)
            hashcode[hashcode < 0] = 0
            hashcode[hashcode > 0] = 1
            
            hashcode.eliminate_zeros()

        else:
            index = self.doc_indices[id1]
            hashcode = self.hashcodes[index,:]

        
        asstr = hashcode.A.astype('S1').tostring()
        
#        nonzeros = np.nonzero(hashcode)
#        asstr = ['0']*self.hyperPlanesNumber
#        if nonzeros != None:
#            for i in nonzeros[1]:
#                asstr[i] = '1'
#        asstr = ''.join(asstr)

#        ll=np.empty(hashcode.shape[1], dtype='U1')
#        ll.fill('0')
#        print (hashcode)
#        ll[np.nonzero(hashcode)]='1'

        values = (hashcode, asstr)

        #asstr.join(hashcode)
        #debug(self.logger, 'generateHashCode: {}'.format (time.time()-base))
        self.logger.exit('HashtableLSH.generateHashCode')
        return values

    def helper_calc(self, point):
        result = point.dot(self.hyperPlanes)
        result[result < 0] = 0
        result[result > 0] = 1
            
        result.eliminate_zeros()
      
        return result


    def add_all(self, doc_indices, points):
        self.logger.entry('HashtableLSH.add_all')        
        self.doc_indices = doc_indices
        #old version: result1 = map( lambda x: x.dot(self.hyperPlanes) , points )
        #self.hashcodes2 = map( lambda x: self.helper_calc(x) , points )
        
        
        #for h in self.hashcodes2:
        #    print(h)

        self.hashcodes = points.dot(self.hyperPlanes)
        self.hashcodes[self.hashcodes < 0] = 0
        self.hashcodes[self.hashcodes > 0] = 1
            
        self.hashcodes.eliminate_zeros()
        
        self.logger.exit('HashtableLSH.add_all')        
    
    def add_all2(self, doc_indices, points, id_list):
        self.logger.entry('HashtableLSH.add_all2')        
        #reuse hashcodes table
        
        
        # calculate string per hashcode
        
        self.logger.exit('HashtableLSH.add_all2')    
        items = []
        for index in range(len(doc_indices)):
            point = points[index, :]
            ID = id_list[index]
            hashcode = self.hashcodes[index]
            
            item = {}
            item['ID'] = ID
            item['point'] = point
            item['hashcode'] = hashcode
            b = self.buckets.get(hashcode, [])
            b.append( item ) 
            if len(b) > self.maxBucketSize:
                b = b[1:]
                self.total -= 1
            
            self.buckets[hashcode] = b
            items.append(item)
            
        return items
        
    def add(self, ID, point):
        self.logger.entry('HashtableLSH.add')        
        self.total += 1

        
        hasharray, hashcode = self.generateHashCode(point, ID)

        item = {}
        item['ID'] = ID
        item['point'] = point
        item['hashcode'] = hashcode

        #item['similar_count'] = 0
        b = self.buckets.get(hashcode, [])
        #search for closest item
#        minDist = 1
#        for k in b:
#            dist = distance_cosine(k['point'], point)
#            if dist < minDist:
#                #k['similar_count'] += 1
#                #item['similar_count'] += 1
#                minDist = dist
#        item['dist'] = minDist
            
            
        b.append( item ) 
        
        if len(b) > self.maxBucketSize:
            b = b[1:]
            self.total -= 1
        
        self.buckets[hashcode] = b
        
        self.logger.exit('HashtableLSH.add')        
        return item

    def randomHyperPlanes(self, dimSize, hyperPlanesNumber):
        self.logger.entry('HashtableLSH.randomHyperPlanes')        
        rvs = stats.norm(loc=0).rvs  #scale=2, 
        planes = sparse.random(dimSize, hyperPlanesNumber, format='lil', density=1.0, data_rvs=rvs)
        
        
#        for k in range(planes.shape[1]):
#            debug("randomHyperPlanes", 'plane ' + str(k) + ':\n' + str(planes[:,k]))
#            for r in range(k):
#                if (planes[:,k] - planes[:,r]).nnz == 0:
#                    debug("randomHyperPlanes", 'duplicate ' + str(k) + ' and ' + str(r))
        
        self.logger.exit('HashtableLSH.randomHyperPlanes')        
        return planes
   
    def size(self):
        return self.total

    def distance_aprox(self, id1, id2, p1, p2):
        self.logger.entry('distance_aprox')        
        h1, st1 = self.generateHashCode(p1, id1)
        h2, st2 = self.generateHashCode(p2, id2)
        
        xor = (h1 + h2) 
        xor[xor == 2] = 0
        xor.eliminate_zeros()
        
        nnz_xor = xor.getnnz()
        #xor = h1^h2
#        nnz_xor = 0
#        for (c1,c2)  in zip(h1, h2):
#            if c1!=c2 and (c1=='1' or c2=='1'):
#                nnz_xor += 1
        
        d = self.hyperPlanesNumber
        res = (d-nnz_xor)/float(d)
        self.logger.exit('distance_aprox')        
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
        info(self.logger, 'total: {0}. number of buckets: {1}.  max bucke size: {2}. hyperplanes number: {3}'.format(self.total, len(self.buckets), self.maxBucketSize, self.hyperPlanesNumber))
        lengths = [len(self.buckets[b]) for b in self.buckets]
        info(self.logger, 'number of items in each bucket: {}'.format( lengths))

import time        
        
class LSH:
    """LSH class"""
    dimSize = 3
    numberTables = None
    hList = None
    logger = None
    nearestNeighbor = {}
    helper = None

    def __init__(self, dimensionSize, logger, hyperPlanesNumber=40, numberTables=4, maxBucketSize=10):
        self.logger = logger
        self.logger.entry('LSH.__init__')
        self.helper = MathHelper(logger)
        self.helper.clear()
        info(self.logger, 'LSH model being initialized')
        self.dimSize = dimensionSize
        self.numberTables = numberTables
        self.hList = [HashtableLSH(maxBucketSize, dimensionSize, hyperPlanesNumber, logger, self.helper) for i in range(numberTables)]
        info(self.logger, 'LSH model initialization done')
        self.logger.exit('LSH.__init__')

    def myprint(self):
        if self.logger == None:
            return
            
        for h in self.hList:
            debug(self.logger, '*******************************************')
            h.myprint()
        info(self.logger, 'dimenion: {0} tables {1} '.format(self.dimSize, self.numberTables))

    def add_all(self, doc_indices, points):
        self.logger.entry('LSH.add_all')        
        invokes = []
        for table in self.hList:
            invokes.append( (table.add_all, doc_indices, points) )
            #table.add_all(doc_indices, points)
        multirun.run_all(invokes)
        self.logger.entry('LSH.add_all')        

    def add_all2(self, doc_indices, points, id_list):
        results = []
        for table in self.hList:
            res = table.add_all2(doc_indices, points, id_list)
            results.append(res)
        return results

    def add_single(self, table, ID, point):
        self.logger.entry('LSH.add_single')        
        item = table.add(ID, point)
        
        if True:
            candidateNeighbor, dist, bucket_size = table.findNearest(item)            
        self.logger.exit('LSH.add_single')        
        return candidateNeighbor, dist, bucket_size 
        
    def add(self, ID, point):
        self.logger.entry('LSH.add-a')        
        """add a point to the hash table
        the format of the point is assumed to be parse so it will be in libSVM format
        json {word:count, word:cout}"""
        nearest = None
        nearestDist = None
        comparisons = 0
        invokes = []
        results = []
        for table in self.hList:
            #candidateNeighbor, dist, bucket_size  = self.add_single(table, ID, point)
            #results.append((candidateNeighbor, dist, bucket_size ))
            invokes.append( ( self.add_single, table, ID, point ) )
        results = multirun.run_all(invokes)
        
        self.logger.exit('LSH.add-a')        
        self.logger.entry('LSH.add-b')        
        if True:
            for candidateNeighbor, dist, bucket_size in results:
                comparisons += bucket_size-1
                if nearestDist==None or (dist != None and nearestDist>dist):
                    nearest = candidateNeighbor
                    nearestDist = dist
                    
        
        self.logger.exit('LSH.add-b')        
        return nearest, nearestDist, comparisons

    def add1(self, ID, point):
        self.logger.entry('LSH.add')        
        """add a point to the hash table
        the format of the point is assumed to be parse so it will be in libSVM format
        json {word:count, word:cout}"""
        nearest = None
        nearestDist = None
        comparisons = 0
        for table in self.hList:
            item = table.add(ID, point)
            
            if True:
                candidateNeighbor, dist, bucket_size = table.findNearest(item)            
                comparisons += bucket_size-1
                if nearestDist==None or (dist != None and nearestDist>dist):
                    nearest = candidateNeighbor
                    nearestDist = dist
                    
        
        self.logger.exit('LSH.add')        
        return nearest, nearestDist, comparisons


    def distance(self, id1, id2, p1, p2):
        dist = None
        for h in self.hList:
            d = h.distance_aprox(id1, id2, p1, p2)
            if dist == None or dist>d:
                dist = d
            
                
#        h = self.hList[  np.random.randint( len(self.hList) ) ]
#        dmax = h.distance_aprox(p1, p2)
        
        return dist
        
    def hash_similarity(self, id1, id2, p1, p2):
        return 1-self.distance(id1, id2, p1, p2)
            

    def evaluate(self, nruns):
        self.helper.clear()
        avg = 0
        
        id1 = 0
        id2 = 1
        for run in range(nruns):

            p1 = self.helper.randomPoint(self.dimSize)        
            p2 = self.helper.randomPoint(self.dimSize)   
            
            hash_sim = self.hash_similarity(id1, id2, p1, p2) 
            true_sim = self.helper.angular_similarity(id1, id2, p1, p2)
            id1 += 2
            id2 += 2
            diff = abs(hash_sim-true_sim)/true_sim
            avg += diff
            info(self.logger, '%d: true %.4f, hash %.4f, diff %.4f' % (run, true_sim, hash_sim, diff) )
        info(self.logger, 'avg diff {}'.format( avg / nruns))
   
        self.helper.clear()



#%%
if __name__ == '__main__':
  

    from simplelogger import simplelogger 
    from time import time
    
    def init_log():
        #logging.config.fileConfig('logging.conf')
        
        # create logger
        logger = simplelogger()
        logger.init('c:/temp/file2.log', file_level=simplelogger.INFO, std_level=simplelogger.INFO, profiling=True)
        
        # 'application' code
        debug(logger, 'debug message')
        info(logger, 'info message')
        #logger.warning('warn message')
        logger.error('error message')
    
        return logger

    
    def test2(logger):
        n = 4
        d = 2**n
        dim = 5
        maxB = 500
        tables = 5
        nruns = 1000000
        
        logger = init_log()
        helper = MathHelper(logger)
        
    
        ll = LSH(dimensionSize=dim, logger=logger, numberTables=tables, hyperPlanesNumber=d, maxBucketSize=maxB)
        ll.evaluate(100)
    
        ID = 100
        before = time()
        for run in range(nruns):
            p = helper.randomPoint(dim)
            
            #logger.error('Generated point D{0}:\n{1}'.format(ID, p))
            
            ll.add('D'+str(ID), p)
            ID += 1
            perc = 100.0*run/nruns
            if perc % 10 == 0:
                t = time()-before
                info(logger, '%.2f %% (time: %.8f)' % (perc, t))
                before = time()
    
        ll.myprint()
     
        
        return ll
    
    
    logger = init_log()
#    helper = MathHelper(logger)
#    
#    p1 = helper.randomPoint(3)
#    p2 = helper.randomPoint(3)
#    
#    print (helper.angular_similarity(1, 2, p1, p2))
#    helper.clear()
    
    #print angular_similarity(p1, p1)
    
    ll = test2(logger)
    
#    rvs = stats.randint(low = -5, high = 5).rvs #.norm(scale=2, loc=0).rvs
#    planes = sparse.random(4, 10, format='csr', density=0.80, data_rvs=rvs)
#    
#    temp = planes
#    print temp.shape
#    temp = np.square(temp)
#    print temp.shape
#    temp = planes.sum(axis=0)
#    print temp.shape
#    temp = np.sqrt(temp )
#    print temp.shape
    #temp = np.sum(np.abs(planes)**2,axis=(0, 1))**(1./2)
    
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
    
    logger.close()