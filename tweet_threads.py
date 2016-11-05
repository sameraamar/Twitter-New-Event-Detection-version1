from simplelogger import simplelogger
from math import log
import numpy as np

class TweetThread:
    thread_id = None
    thread_doc = None
    score = 0
    document_contents = None
    idlist = list()
    counter = {}
    users = set()
    count_all = 0
    
    min_timestamp = None
    max_timestamp = None
    

    def __init__(self, thread_id, thread_doc, user, timestamp):
        self.thread_id = thread_id
        self.thread_doc = thread_doc
        self.document_contents = {}
        self.idlist = list()
        self.counter = {}
        self.users = set()
        self.count_all = 0
        self.min_timestamp = timestamp
        self.max_timestamp = timestamp
        
        self.append(thread_id, thread_doc, user, timestamp, None, None)


    def append(self, ID, document, user, timestamp, nearID, nearestDist):
        self.document_contents[ID] = (document, nearID, nearestDist)
        self.idlist.append(ID)
        nonzeros = np.nonzero(document)
        for i in nonzeros[1]:
            c = document[0,i]
            self.counter[i] = self.counter.get(i, 0) + c
            self.count_all += c
        self.users.add(user) 
        
        if self.min_timestamp > timestamp:
            self.min_timestamp = timestamp
            
        if self.max_timestamp < timestamp:
            self.max_timestamp = timestamp    

    def users_count(self):
        return len(self.users)

    def size(self):
        return len(self.idlist)

    def entropy(self):
        segma = 0
        for i in self.counter:
            d = self.counter[i]/self.count_all
            segma += d * log(d) 
        return (-1) * segma
        
    def thread_time(self):
        return self.max_timestamp-self.min_timestamp
        