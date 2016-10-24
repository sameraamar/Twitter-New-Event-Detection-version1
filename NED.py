# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""

import LSH_2 as lsh
import json

#%%

from simplelogger import simplelogger
import time, codecs

def init_log():
    logger = simplelogger()
    logger.init(filename='c:/temp/lsh.log', std_level=simplelogger.INFO, file_level=simplelogger.DEBUG)
    
    return logger

#%%

class NED_LSH_model:
    lsh = None
    tables = 0
    hyper_planes = 0
    dimension = 0
    max_bucket_size = 0
    logger = None
    epsilon = 0
    
    #%%
    threads = {}
    tweet2thread = {}
    text_data = []
    id_list = []
    text_metadata = {}

    #'''
    counts = None
    count_vect = None

    def init(self, logger, hyper_planes, tables, max_bucket_size=50, dimension=3, epsilon=0.5):
        self.logger = logger
        self.hyper_planes = hyper_planes
        self.tables = tables
        self.max_bucket_size = max_bucket_size
        self.dimension = dimension
        self.epsilon = epsilon

    def rebuild(self):
        self.lsh = lsh.LSH(logger=self.logger, dimensionSize=self.dimension , numberTables=self.tables, 
                     hyperPlanesNumber=self.hyper_planes, maxBucketSize=self.max_bucket_size)

    def run(self, text_data, id_list, text_metadata):
        self.text_data = text_data
        self.id_list = id_list
        self.text_metadata = text_metadata
        
        self.count_vect = CountVectorizer() #stop_words='english')
        self.counts = self.count_vect.fit_transform(text_data)
            
        # update the dimension
        self.dimension = self.counts.shape[1]
        self.rebuild()
            
        nn = len(text_data)
        p = 0.0
        comparisons_all = 0
        before = time.time()
        for sample in range(nn):
            ID = self.id_list[sample]
            doc = self.counts[sample, :]

            #ID = text_data[sample]['id_str'] #text_metadata[sample]['ID']
            self.logger.debug ('Adding document {0} ({2}) out of {1}'.format(sample, self.counts.shape[0], ID))
            nearest, nearestDist, comparisons = lshmodel.lsh.add(ID, doc)
            
            if nearestDist == None or nearestDist > self.epsilon:
                self.threads[ID] = [ID]
                self.tweet2thread[ID] = ID
            else:
                nearID = nearest['ID']
                nearThreadID = self.tweet2thread[nearID]
                self.threads[nearThreadID].append(ID)
                self.tweet2thread[ID] = nearThreadID
            
            comparisons_all += comparisons
            tmp = int(10.0*sample/nn)
            if (p < tmp):
                p = tmp
                after = time.time()
                self.logger.info('{2}/{3} = {0} %% - {1} seconds [comparisons: {4}]'.format(10*p, after-before, sample, nn, comparisons_all))
                before = after
                comparisons_all = 0
                
        self.logger.info ('corpus size: {0} '.format(doc.shape[1]))
        
    def dumpThreads(self, filename, max_threads):
        file = codecs.open(filename, 'w', encoding='utf-8')
        
        file.write('Printing {} threads...\n'.format( min(max_threads, len(self.threads) ) ) )
        thr = 1
        for x in sorted(self.threads, key=lambda x: len(self.threads[x]), reverse=True):
            threadSize = len(self.threads[x])
            
            self.logger.info('{0}: {1}'.format(x, threadSize))
            text = self.text_metadata[x]['text'] #.replace('\t', ' ')
            #text = text.encode(encoding='utf-8')
            file.write('\n' + '-'*40 + ' THREAD {0} - {1} documents '.format(thr, threadSize) + '-'*40 + '\n')
            file.write('Leader is {0}: "{1}"\n'.format(x, text))
            file.write('thread\tleading doc\titem#\titem ID\titem text\titem text(original)\n')
            c = 1
            for item in self.threads[x]:
                i = self.text_metadata[item]['#'] 
                text1 = self.text_data[i]
                text2 = self.text_metadata[item]['text'] 
            
                file.write('{0}\t{1}\t{2}\t{3}\t"{4}"\t"{5}"\n'.format( thr, x, c, item, text1, text2 ))
                c+=1
            thr += 1
            if thr>max_threads:
                break
            
        file.close()
       
        
class Listener:
    logger = None
    
    def __init__(self, logger):
        self.logger = logger
        
    def act(self, data):
        # do nothing and continue
        return True

class Action:
    listeners = []
    logger = None
    
    def __init__(self, logger):
        self.logger = logger
        
    def register(self, listener):
        self.listeners.append(listener)

    def publish(self, data):
        proceed = True
        for l in self.listeners:
            proceed = l.act(data) and proceed

        return proceed
        
class TextStreamer(Action):
    source = None

    def init(self, filename):
        self.source = open(filename, 'r')
 
    def start(self):
        for line in self.source:
            if line.strip() == '':
                continue
            
            # json
            data = json.loads(line)
            # convert to text 
            itemTime = data['created_at']
            
            # publish to listeners
            if not self.publish(data):
                break

            
            # time controller
       
        self.logger.info('TextStreamer is shutting down')

        
from pymongo import MongoClient, ASCENDING
import pymongo 
        
class MongoDBStreamer(Action):
    dbcoll = None

    def init(self, host, port, dbname, collname):
        client = MongoClient(host, int(port))
        db = client[dbname]
        self.dbcoll = db[collname]
 
    def start(self):
        for item in self.dbcoll.find().sort("_id", pymongo.ASCENDING):
            # json
            data = item.get('json', None)
            if data == None:
                continue
            
            #data = json.loads(data)
            # convert to text 
            itemTime = data['created_at']
            
            # publish to listeners
            if not self.publish(data):
                break

            
            # time controller
            
        self.logger.info('Database Streamer is shutting down')

        
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import TweetTokenizer

        
class TextListener(Listener):
    lshmodel = None
    text_data = []
    id_list = []
    text_metadata = {}

    max_documents = 0
        
    def init(self, lshmodel, max_documents):
        self.lshmodel = lshmodel
        self.max_documents = max_documents

    def act(self, data):
        metadata = {}
        ID = data['id_str']
        
        metadata['retweet'] = (data.get('retweet', None) != None)
        
        itemText = data['text']
        itemText = self.process(itemText)
        metadata['text'] = data['text'].replace('\t', ' ').replace('\n', '. ')
        self.text_data.append( itemText )
        self.id_list.append ( ID )

        index = len(self.text_data)-1
        metadata['#'] = index
        self.text_metadata[ ID ] = metadata

        if index+1 == self.max_documents:
            self.logger.info('running LSH on {} documents'.format(index+1))
            lshmodel.run(self.text_data, self.id_list, self.text_metadata)
                
            return False
            
        if index % 100 == 0:
            self.logger.debug('Loaded {} documents'.format(index))
            
        return True
          
    def process(self, text):
        
#        tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
#        word_tokens = tknzr.tokenize(text)
#        text = ' '.join(word_tokens)
        text = text.replace('\t', ' ').replace('\n', ' ')
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return text.lower().strip()
        
        
#%%        
n = 3
hp = 2**n
maxB = 50
#dim=3
tables = 8
epsilon=0.5
#%%
max_threads = 10
max_docs = 10000

#%%

#mongodb
host = 'localhost'
port = 27017
db = 'events2012'
collection = 'posts'
#db = 'test'
#collection = 'test'

#%%
logger = init_log()

#%%
lshmodel = NED_LSH_model()
lshmodel.init( logger, hp, tables, max_bucket_size=maxB, dimension=3, epsilon=epsilon)


#streamer = TextStreamer(logger)
#streamer.init('C:\data\_Personal\DataScientist\datasets\Italy1.json')

streamer = MongoDBStreamer(logger)
streamer.init(host, port, db, collection)

listener = TextListener(logger)
listener.init(lshmodel, max_docs)

streamer.register(listener)
streamer.start()

nn = len(listener.text_data)
logger.info('Loaded {} text documents.'.format(nn))

#%%
#print (type(listener.text_data))
#
#print (listener.text_data[5])
#print (lshmodel.counts[5, :])
#print (lshmodel.count_vect.get_feature_names()[572])

lshmodel.lsh.myprint()

#%%
    
_thr = lshmodel.dumpThreads('c:/temp/threads.txt', max_threads)


#%%
#print (lshmodel.threads['768236668625252352'])


    
#%%
logger.info('Done.')
#logger.close()

