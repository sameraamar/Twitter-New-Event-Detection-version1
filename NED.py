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
    doc_indices = {}
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
        self.threads = {}

    def rebuild(self):
        self.lsh = lsh.LSH(logger=self.logger, dimensionSize=self.dimension , numberTables=self.tables, 
                     hyperPlanesNumber=self.hyper_planes, maxBucketSize=self.max_bucket_size)

    def run(self, text_data, id_list, text_metadata, doc_indices):
        self.logger.entry('NED_LSH_model.run')
        self.text_data = text_data
        self.id_list = id_list
        self.text_metadata = text_metadata
        self.doc_indices = doc_indices
        self.threads = {}
        self.tweet2thread = {}
        
        self.count_vect = CountVectorizer() #stop_words='english')
        self.counts = self.count_vect.fit_transform(text_data)
            
        # update the dimension
        self.dimension = self.counts.shape[1]
        self.rebuild()
            
        nn = len(text_data)
        p = 0.0
        comparisons_all = 0
        base = before = time.time()
        
#        self.logger.debug ('Adding document {0} ({2}) out of {1}'.format(sample, self.counts.shape[0], ID))
#        nearest, nearestDist, comparisons = lshmodel.lsh.add_all(self.id_list, self.counts)
        lshmodel.lsh.add_all(self.doc_indices, self.counts)
        #lshmodel.lsh.add_all2(self.doc_indices, self.counts, self.id_list)


        block = nn / 20
        for sample in range(nn):
            ID = self.id_list[sample]
            doc = self.counts[sample, :]

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
            
            if (sample % block == 0):
                p = 100.0*sample/nn
                after = time.time()
                tmp = after-base
                tmp = (int(tmp/60) , int(tmp)%60)
                self.logger.info('{2}/{3} = {0} %% - {1:.1f} seconds [comparisons: {4}]. Total time spent: {5}:{6}'.format(p, (after-before), sample, nn, comparisons_all, tmp[0], tmp[1]))
                before = after
                comparisons_all = 0
                
        self.logger.info ('corpus size: {0} '.format(doc.shape[1]))
        self.logger.exit('NED_LSH_model.run')
        
    def dumpThreads(self, filename, max_threads):
        self.logger.entry('dumpThreads')
        file = codecs.open(filename, 'w', encoding='utf-8')
        
        file.write('Printing {} threads...\n'.format( min(max_threads, len(self.threads) ) ) )
        thr = 1
        for x in sorted(self.threads, key=lambda x: len(self.threads[x]), reverse=True):
            threadSize = len(self.threads[x])
            
            if threadSize<3:    
                #not interesting anymore
                break
            
            self.logger.info('Thread {0}: has {1} documents'.format(x, threadSize))
            text = self.text_metadata[x]['text'] #.replace('\t', ' ')
            #text = text.encode(encoding='utf-8')
            file.write('\n' + '-'*40 + ' THREAD {0} - {1} documents '.format(thr, threadSize) + '-'*40 + '\n')
            file.write('Leader is {0}: "{1}"\n'.format(x, text))
            file.write('thread\tleading doc\titem#\titem ID\tuser\titem text\titem text(original)\n')
            c = 1
            for item in self.threads[x]:
                i = self.doc_indices[item]
                text1 = self.text_data[i]
                text2 = self.text_metadata[item]['text'] 
                user = self.text_metadata[item]['user']
            
                file.write('{0}\t{1}\t{2}\t{3}\t{4}\t"{5}"\t"{6}"\n'.format( thr, x, c, item, user, text1, text2 ))
                c+=1
            thr += 1
            if thr>max_threads:
                break
            
        file.close()
        self.logger.exit('dumpThreads')
       
        
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
            created_at = data['created_at']

            itemTimestamp = time.mktime(time.strptime(created_at,"%a %b %d %H:%M:%S +0000 %Y"))
            data['timestamp'] = itemTimestamp
            
            # publish to listeners
            if not self.publish(data):
                break

            
            # time controller
       
        self.logger.info('TextStreamer is shutting down')

        
from pymongo import MongoClient
import pymongo 
     

class MongoDBStreamer(Action):
    dbcoll = None
    offset = 0

    def init(self, host, port, dbname, collname, offset):
        client = MongoClient(host, int(port))
        db = client[dbname]
        self.dbcoll = db[collname]
        self.offset = offset
 
    def start(self):
        self.logger.entry('MongoDBStreamer.start')
        previous = None
        maxDelta = 0
        for item in self.dbcoll.find().sort("_id", pymongo.ASCENDING).skip(self.offset):
            # json
            data = item.get('json', None)
            if data == None:
                continue
            
            #data = json.loads(data)
            # convert to text 
            created_at = data['created_at']

            itemTimestamp = time.mktime(time.strptime(created_at,"%a %b %d %H:%M:%S +0000 %Y"))
            data['timestamp'] = itemTimestamp

            if previous!=None:
                delta = data['timestamp'] - previous['timestamp']
                if delta > maxDelta:
                    maxDelta = delta
                    text = 'Delta between tweets is {4}  ---> {0}: {2}. {1}: {3}'.format(data['id_str'], previous['id_str'], data['timestamp'], previous['timestamp'], maxDelta)
                    self.logger.error(text)
                
            previous = data

            
            # publish to listeners
            if not self.publish(data):
                break

            
            # time controller
            
        self.logger.info('Database Streamer is shutting down')
        self.logger.exit('MongoDBStreamer.start')

        
from sklearn.feature_extraction.text import CountVectorizer
import re
#from nltk.tokenize import TweetTokenizer

        
class TextListener(Listener):
    lshmodel = None
    text_data = []
    id_list = []
    text_metadata = {}
    doc_indices = {}

    max_documents = 0
        
    def init(self, lshmodel, max_documents):
        self.lshmodel = lshmodel
        self.max_documents = max_documents
        
        self.text_data = []
        self.id_list = []
        self.text_metadata = {}
        self.doc_indices = {}        

    def act(self, data):
        metadata = {}
        ID = data['id_str']
        
        metadata['retweet'] = (data.get('retweet', None) != None)
        
        metadata['user'] = data['user']['screen_name']
        metadata['timestamp'] = data['timestamp']
        itemText = data['text']
        itemText = self.process(itemText)
        metadata['text'] = data['text'].replace('\t', ' ').replace('\n', '. ')
        self.text_data.append( itemText )
        self.id_list.append ( ID )

        index = len(self.text_data)-1
        self.text_metadata[ ID ] = metadata
        self.doc_indices[ ID ] = index

        if index == 1:
            self.logger.info("First tweet: {}".format(metadata))

        if index+1 == self.max_documents:
            before = time.time()
            self.logger.info("Last tweet: {}".format(metadata))
            self.logger.info('running LSH on {} documents'.format(index+1))
            
            lshmodel.run(self.text_data, self.id_list, self.text_metadata, self.doc_indices)
            x = time.time()-before
            x = (int(x/60), int(x)%60)

            self.logger.info('Time for running the LSH model was: {0} min and {1} sec'.format(x[0], x[1]))
            return False
            
        if index % 100 == 0:
            self.logger.debug('Loaded {} documents'.format(index))
            
        return True
          
    def process(self, text):
        
#        tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
#        word_tokens = tknzr.tokenize(text)
#        text = ' '.join(word_tokens)
        text = text.replace('\t', ' ').replace('\n', ' ')
        
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return text.lower().strip()
        
        
#%%        

import psutil
import os

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss #[0] / float(2 ** 20)
    return mem
    
n = 4
hp = 2**n
maxB = 50  # should be less than 0.5 of max_docs/(2^hp)
#dim=3
tables = 13
epsilon=0.5
#%%
max_threads = 1000
max_docs = 10000

#%%

import sys
if len(sys.argv)>1:
    max_docs = int(sys.argv[1])
    
preformance_file = 'c:/temp/performance-{}.txt'.format(max_docs)
file = open(preformance_file, 'a')
file.write('max_docs\tseconds\tminutes\tusage\n')

for round in range(0, 30):
#if True:
    print('Running LSH with {0} tweets ..... '.format(max_docs), end='')
    
    #mongodb
    host = 'localhost' #'192.168.1.100'
    port = 27017
    db = 'events2012'#'petrovic'
    collection = 'posts' #'relevance_judgments'
    #db = 'test'
    #collection = 'test'
    
    log_filename = 'c:/temp/{0:07d}_docs_round_{1:02d}.log'.format(max_docs, round)
    threads_filename = 'c:/temp/{0:07d}_docs_round_{1:02d}_threads.txt'.format(max_docs, round)
    
    #%%
    logger = simplelogger()
    logger.init(filename=log_filename, std_level=simplelogger.INFO, file_level=simplelogger.INFO, profiling=False)
    #logger.init(filename=log_filename, std_level=simplelogger.INFO, file_level=simplelogger.DEBUG, profiling=False)
    
    #%%
    start = time.time()
    lshmodel = NED_LSH_model()
    lshmodel.init( logger, hp, tables, max_bucket_size=maxB, dimension=3, epsilon=epsilon)
    
    
    #streamer = TextStreamer(logger)
    #streamer.init('C:\data\_Personal\DataScientist\datasets\Italy1.json')
    
    streamer = MongoDBStreamer(logger)
    streamer.init(host, port, db, collection, offset=int(round*max_docs/2))
    
    listener = TextListener(logger)
    listener.init(lshmodel, max_docs)
    
    streamer.register(listener)
    streamer.start()
    
    nn = len(listener.text_data)
    logger.info('Loaded {} text documents.'.format(nn))
    
    #%%
        
    _thr = lshmodel.dumpThreads(threads_filename, max_threads)

    logger.info('print profiling!')
    
    logger.profiling_dump()
    
    measured_time = time.time() - start
    
    usage_psutil = memory_usage_psutil()

    logger.info('done with {0:.2f} seconds (= {1:.2f} minutes). usage: {2}'.format(measured_time, measured_time/60, usage_psutil))
    file.write('{0}\t{1}\t{2}\t{3}\n'.format(max_docs, measured_time, measured_time/60, usage_psutil))
    #%%
    #print (type(listener.text_data))
    #
    #print (listener.text_data[5])
    #print (lshmodel.counts[5, :])
    #print (lshmodel.count_vect.get_feature_names()[572])
    
    #lshmodel.lsh.myprint()
    
    
    
        
    #%%
    logger.info('Done.')
    logger.close()

    #input('Round {} is done. Press Enter...'.format(round))

file.close()
