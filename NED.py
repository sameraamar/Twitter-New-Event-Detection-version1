# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""

import LSH as lsh
import json

#%%

from simplelogger import simplelogger
import time, codecs
from tweet_threads import TweetThread

#%%

class NED_LSH_model:
    lsh = None
    tables = 0
    hyper_planes = 0
    dimension = 0
    max_bucket_size = 0
    logger = None
    threshold = 0
    resent_documents = 0 
    
    #%%
    threads_queue = {}
    threads = {}
    tweet2thread = {}
    text_data = []
    doc_indices = {}
    id_list = []
    text_metadata = {}
    first_timestamp = None
    last_timestamp = None       


    #'''
    counts = None
    count_vect = None

    def init(self, logger, hyper_planes, tables, max_bucket_size=50, dimension=3, threshold=0.5, resent_documents=10000):
        self.logger = logger
        self.resent_documents = resent_documents
        self.hyper_planes = hyper_planes
        self.tables = tables
        self.max_bucket_size = max_bucket_size
        self.tweet2thread = {}
        self.text_data = []
        self.doc_indices = {}
        self.lsh = None
        self.id_list = []
        self.text_metadata = {}
        self.dimension = dimension
        self.threshold = threshold
        self.threads = {}
        self.threads_queue = {}
        self.first_timestamp = None
        self.last_timestamp = None
        
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
        self.threads_queue = {}
        self.tweet2thread = {}
        self.first_timestamp = None
        self.last_timestamp = None
        self.resent = []
        
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
        self.lsh.add_all(self.doc_indices, self.counts)
        #lshmodel.lsh.add_all2(self.doc_indices, self.counts, self.id_list)


        block = nn / 20
        for sample in range(nn):
            ID = self.id_list[sample]
            doc = self.counts[sample, :]

            self.logger.debug ('Adding document {0} ({2}) out of {1}'.format(sample, self.counts.shape[0], ID))
            nearest, nearestDist, comparisons = self.lsh.add(ID, doc)
            
            data = text_metadata[ID]
            if self.first_timestamp == None:
                self.first_timestamp = data['timestamp']
                
            if self.last_timestamp == None or self.last_timestamp < data['timestamp']:
                self.last_timestamp = data['timestamp']

            if nearestDist == None or nearestDist > self.threshold:
                #compare d to a fixed number of most recent documents
                flag = False
                for other in self.resent:
                    tmp = self.lsh.helper.angular_distance(ID, other, doc, self.counts[self.doc_indices[other], :])
                    if nearestDist == None or nearestDist > tmp:
                        nearestDist = tmp
                        nearest = {'ID' : other}
                        flag = True
                if flag: #found a new neighbor
                    self.logger.debug('*** CANDIDATE OF NEW THREAD ***: {0} ("{1}") was found as close to {2} ("{3}") distance {4}.'.format(ID, nearest['ID'], self.text_metadata[ID]['text'], self.text_metadata.get(other, ''), tmp))
                        
            nearestID = None
            if nearest != None:
                nearestID = nearest['ID']

            if nearestDist == None or nearestDist > self.threshold:
                self.threads[ID] = [ID]
                self.tweet2thread[ID] = ID
                self.threads_queue[ID] = TweetThread(ID, doc, data['user'], data['timestamp'])
                self.logger.debug('*** NEW THREAD ***: leader is {0} ("{3}"). Nearest is {1} ("{4}") with distance {2}.'.format(ID, nearestID, nearestDist, self.text_metadata[ID]['text'], self.text_metadata.get(nearestID, '')))

                

            else:
                nearThreadID = self.tweet2thread[nearestID]
                self.threads[nearThreadID].append(ID)
                self.threads_queue[nearThreadID].append(ID, doc, data['user'], data['timestamp'], nearestID, nearestDist)
                self.tweet2thread[ID] = nearThreadID
                self.logger.debug('*** EXISTING THREAD ***: Add document {0} ("{1}") to existing thread {2} ("{3}"). Nearest document is {4} ("{5}") with distance {6}.'.format(ID, self.text_metadata[ID]['text'], nearThreadID, self.text_metadata[nearThreadID]['text'], nearestID, self.text_metadata.get(nearestID, ''), nearestDist))
            
            self.logger.entry('NED_LSH_model.run.resent-docs')
            self.resent.append(ID)
            if len(self.resent) > self.resent_documents:
                self.resent = self.resent[1:]
            self.logger.exit('NED_LSH_model.run.resent-docs')

            comparisons_all += comparisons
            
            if (sample % block == 0):
                p = 100.0*sample/nn
                after = time.time()
                tmp = after-base
                tmp = (int(tmp/60) , int(tmp)%60)
                self.logger.info('{2}/{3} = {0} % - {1:.1f} seconds. Total time spent: {4}:{5}'.format(p, (after-before), sample, nn, tmp[0], tmp[1]))
                before = after
                comparisons_all = 0
                
        self.logger.info ('corpus size: {0} '.format(doc.shape[1]))
        self.logger.exit('NED_LSH_model.run')
        
    def dumpThreads(self, filename, max_threads):
        self.logger.entry('dumpThreads')
        file = codecs.open(filename, 'w', encoding='utf-8')
        
        ttt = self.last_timestamp - self.first_timestamp
        file.write('Printing {1} threads... total period: {0}\n'.format( ttt, min(max_threads, len(self.threads_queue) ))) 
        thr = 1
        for x in sorted(self.threads, key=lambda x: len(self.threads[x]), reverse=True):
            threadSize = len(self.threads[x])
            
            #if threadSize<3:    
            #    #not interesting anymore
            #    break
            self.logger.debug('Thread: {0}, size: {1} documents'.format(x, threadSize))
            text = self.text_metadata[x]['text'] #.replace('\t', ' ')
            #text = text.encode(encoding='utf-8')
            file.write('\n' + '-'*40 + ' THREAD {0} - {1} documents score: {2} and {3} users'.format(thr, threadSize, 0, 0) + '-'*40 + '\n')
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
       
    def dumpThreads2(self, filename, max_threads):
        self.logger.entry('dumpThreads')
        file = codecs.open(filename, 'w', encoding='utf-8')
        
        ttt = self.last_timestamp - self.first_timestamp
        file.write('Printing {1} threads... total period: {0}\n'.format( ttt, min(max_threads, len(self.threads_queue) ))) 
        thr = 1
        for x in sorted(self.threads_queue, key=lambda x: self.threads_queue[x].size(), reverse=True):
            threadSize = self.threads_queue[x].size()
            
            #if threadSize<3:    
            #    #not interesting anymore
            #    break
            
            self.logger.debug('Thread: {0}, size: {1} documents'.format(x, threadSize))
            text = self.text_metadata[x]['text'] #.replace('\t', ' ')
            #text = text.encode(encoding='utf-8')
            file.write('\n' + '-'*40 + ' THREAD {0} - {1} documents score: {2} and {3} users. period of {4} seconds'.format(thr, threadSize, self.threads_queue[x].entropy(), self.threads_queue[x].users_count(), self.threads_queue[x].thread_time()) + '-'*40 + '\n')
            file.write('Leader is {0}: "{1}"\n'.format(x, text))
            file.write('thread\tleading doc\titem#\titem ID\tuser\tnearest ID\tdistance\titem text\titem text(original)\n')
            c = 1
            for item in self.threads_queue[x].idlist:
                i = self.doc_indices[item]
                text1 = self.text_data[i]
                text2 = self.text_metadata[item]['text'] 
                user = self.text_metadata[item]['user']
                nearID = self.threads_queue[x].document_contents[item][1]
                nearestDist = self.threads_queue[x].document_contents[item][2]
                file.write('{0}\t{1}\t{2}\t{3}\t{7}\t{8}\t{4}\t"{5}"\t"{6}"\n'.format( thr, x, c, item, user, text1, text2, nearID, nearestDist ))
                c+=1
            thr += 1
            if thr>max_threads:
                break
            
        file.close()
        self.logger.exit('dumpThreads')
      
    def helper_lambda(self, x):
        return '-'.join( [str(self.threads_queue[x].entropy()) , str(self.threads_queue[x].users_count()) ] )
        #return self.threads_queue[x].entropy()
    
    def jsonify(self, max_threads):
        threads = self.jsonify_threads(max_threads)
        tables = {}
        i = 1
        tables['dimension'] = self.lsh.dimSize
        tables['tables'] = []
        for table in self.lsh.hList:
            data = {}
            data['hyperplanes'] = []
            #for hp in table.hyperPlanes:
            #    data['hyperplanes'].append ( hp)
            
            data['buckets'] = []
            data['count'] = len(table.buckets)
            sorted_keys = table.buckets.keys()
            for b in sorted(sorted_keys, reverse=False):
                bucket_data = {}
                bucket_data['hashcode'] = b
                
                temp = table.buckets[b] 
                bucket_data['documents'] = []
                for neighbor in temp:
                    tmp = {}
                    tmp['ID'] = neighbor['ID'] 
                    tmp['text'] = self.text_metadata[tmp['ID']]['text']
                    bucket_data['documents'].append(tmp)

                data['buckets'].append ( bucket_data )
                
            tables['tables'].append(data)
            i += 1

        return threads, tables
        
    def jsonify_threads(self, max_threads):
        data = {}

        data['thread_timeslot'] = self.last_timestamp - self.first_timestamp
        data['threads_count'] = min(max_threads, len(self.threads_queue) )
        data['_list_'] = []
        thr = 1
        for x in sorted(self.threads_queue, key=lambda x: self.helper_lambda(x), reverse=True):
            thread = {}
            threadSize = self.threads_queue[x].size()
            
            #if threadSize<3:    
            #    #not interesting anymore
            #    break
            
            text = self.text_metadata[x]['text'] 
            thread['leader_id'] = x
            thread['leader_text'] = text
            thread['size'] = threadSize
            thread['entropy'] = self.threads_queue[x].entropy()
            thread['users'] = self.threads_queue[x].users_count()
            thread['speed(sec)'] = self.threads_queue[x].thread_time()
            
            array = []
            c = 1
            for item in self.threads_queue[x].idlist:
                i = self.doc_indices[item]
                text1 = self.text_data[i]
                text2 = self.text_metadata[item]['text'] 
                user = self.text_metadata[item]['user']
                nearID = self.threads_queue[x].document_contents[item][1]
                nearestDist = self.threads_queue[x].document_contents[item][2]
                doc = {}
                doc['index'] = c
                doc['id'] = item
                doc['user'] = user
                doc['text_original'] = text2
                doc['text_clean'] = text1
                doc['user'] = user
                doc['nearest'] = nearID
                doc['distance'] = nearestDist
                
                array.append( doc )
                c+=1
            thread['list'] = array
            
            thr += 1
            data['_list_'].append(thread)
            if thr>max_threads:
                break
        
        return data
        
    def dumpThreads3(self, filename, max_threads):
        self.logger.entry('dumpThreads')
        file = codecs.open(filename, 'w', encoding='utf-8')
        
        ttt = self.last_timestamp - self.first_timestamp
        file.write('Printing {1} threads... total period: {0}\n'.format( ttt, min(max_threads, len(self.threads_queue) ))) 
        thr = 1
        for x in sorted(self.threads_queue, key=lambda x: self.helper_lambda(x), reverse=True):
            threadSize = self.threads_queue[x].size()
            
            #if threadSize<3:    
            #    #not interesting anymore
            #    break
            
            self.logger.debug('Thread: {0}, size: {1} documents'.format(x, threadSize))
            text = self.text_metadata[x]['text'] #.replace('\t', ' ')
            #text = text.encode(encoding='utf-8')
            file.write('\n' + '-'*40 + ' THREAD {0} - {1} documents score: {2} and {3} users. period of {4} seconds'.format(thr, threadSize, self.threads_queue[x].entropy(), self.threads_queue[x].users_count(), self.threads_queue[x].thread_time()) + '-'*40 + '\n')
            file.write('Leader is {0}: "{1}"\n'.format(x, text))
            file.write('thread\tleading doc\titem#\titem ID\tuser\tnearest ID\tdistance\titem text\titem text(original)\n')
            c = 1
            for item in self.threads_queue[x].idlist:
                i = self.doc_indices[item]
                text1 = self.text_data[i]
                text2 = self.text_metadata[item]['text'] 
                user = self.text_metadata[item]['user']
                nearID = self.threads_queue[x].document_contents[item][1]
                nearestDist = self.threads_queue[x].document_contents[item][2]
                file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{7}\t{8}\t"{5}"\t"{6}"\n'.format( thr, x, c, item, user, text1, text2, nearID, nearestDist ))
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

    def __init__(self):
        self.listeners = []
        self.logger = None

    def __init__(self, logger):
        self.listeners = []
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
        super(self.__class__, self).__init__()
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
    client = None
    dbcoll = None
    offset = 0

    def __init__(self, logger):
        super(self.__class__, self).__init__(logger)
        self.dbcoll = None
        self.client = None
        self.offset = 0

    def init(self, host, port, dbname, collname, offset):
        self.client = MongoClient(host, int(port))
        db = self.client[dbname]
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
from simple_twitter_parser import preprocess

        
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
        itemText = data['text']
        itemText = self.process(itemText)
        
        if len(itemText) == 0:
            return True

        metadata = {}
        ID = data['id_str']
        
        metadata['retweet'] = (data.get('retweet', None) != None)
        
        metadata['user'] = data['user']['screen_name']
        metadata['timestamp'] = data['timestamp']

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
            
            self.lshmodel.run(self.text_data, self.id_list, self.text_metadata, self.doc_indices)
            x = time.time()-before
            x = (int(x/60), int(x)%60)

            self.logger.info('Time for running the LSH model was: {0} min and {1} sec'.format(x[0], x[1]))
            return False
            
        if index % 100 == 0:
            self.logger.debug('Loaded {} documents'.format(index))
            
        return True
          
    def process(self, text):
        return preprocess(text, return_text=True, numbers=True, mentions=False, stop_words=False, hashtag=True)
        
        
#%%        

import psutil
import os

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss #[0] / float(2 ** 20)
    return mem


def init_mongodb(k, maxB, tables, threshold, max_docs, page, resent_documents):
    print('Running LSH with {0} tweets ..... '.format(max_docs), end='')

    
    log_filename = 'c:/temp/{0:07d}_docs_round_{1:03d}.log'.format(max_docs, page)
    
    #%%
    logger = simplelogger()
    logger.init(filename=log_filename, std_level=simplelogger.INFO, file_level=simplelogger.DEBUG, profiling=True)
    #logger.init(filename=log_filename, std_level=simplelogger.INFO, file_level=simplelogger.DEBUG, profiling=False)
    
    #%%
    lshmodel = NED_LSH_model()
    lshmodel.init( logger, k, tables, max_bucket_size=maxB, dimension=3, threshold=threshold, resent_documents=resent_documents)
    

    
    return lshmodel
    
def execute(lshmodel, page, max_docs, host, port, db, collection, max_threads):
    preformance_file = 'c:/temp/performance-{}.txt'.format(max_docs)
    file = open(preformance_file, 'a')
    file.write('max_docs\tseconds\tminutes\tusage\n')

    starttime = time.time()
    
    #streamer = TextStreamer(logger)
    #streamer.init('C:\data\_Personal\DataScientist\datasets\Italy1.json')
    
    streamer = MongoDBStreamer(lshmodel.logger)
    streamer.init(host, port, db, collection, offset=int(page*max_docs))
    
    listener = TextListener(lshmodel.logger)
    listener.init(lshmodel, max_docs)
    
    streamer.register(listener)
    streamer.start()
    
    nn = len(listener.text_data)
    lshmodel.logger.info('Loaded {} text documents.'.format(nn))
    
    #%%
    threads_filename = 'c:/temp/{0:07d}_docs_round_{1:02d}_threads.txt'.format(max_docs, page)
        
    lshmodel.dumpThreads(threads_filename.replace('.txt', '1.txt'), max_threads)
    lshmodel.dumpThreads2(threads_filename.replace('.txt', '2.txt'), max_threads)
    lshmodel.dumpThreads3(threads_filename.replace('.txt', '3.txt'), max_threads)
    #print ( lshmodel.jsonify(max_threads) )

    lshmodel.logger.info('print profiling!')
    
    lshmodel.logger.profiling_dump()
    
    measured_time = time.time() - starttime
    
    usage_psutil = memory_usage_psutil()

    lshmodel.logger.info('done with {0:.2f} seconds (= {1:.2f} minutes). usage: {2}'.format(measured_time, measured_time/60, usage_psutil))
    file.write('{0}\t{1}\t{2}\t{3}\n'.format(max_docs, measured_time, measured_time/60, usage_psutil))
        
    #%%
    lshmodel.logger.info('Done.')
    lshmodel.logger.close()

    #input('Round {} is done. Press Enter...'.format(r))
    
    file.close()
    return lshmodel


if __name__ == '__main__':
    
    k = 13
    maxB = 500  # should be less than 0.5 of max_docs/(2^k)
    tables = 64
    threshold = 0.5
    #%%
    max_threads = 2000
    max_docs = 10
    
    #%%
    #mongodb
    host = 'localhost' #'192.168.1.100' #'localhost' 
    port = 27017
    db = 'test' # 'events2012'#'petrovic'
    collection = 'test' #'posts' #'relevance_judgments'
    #db = 'test'
    #collection = 'test'
    
    min_rounds = 0
    max_rounds = 10
    page = 0


    import sys
    if len(sys.argv)>1:
        max_docs = int(sys.argv[1])
    
    if len(sys.argv)>2:
        min_rounds = int(sys.argv[2])
        max_rounds = int(sys.argv[3])

    
    lshmodel = init_mongodb(k, maxB, tables, threshold, max_docs, page, 10)
    execute(lshmodel, page, max_docs, host, port, db, collection, max_threads)

