# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:51:47 2016

@author: SAMERA
"""

def wikidocs():
    
    #text1 = ["This is a document", "I like this document", "Documents are something good"]
    
    from sklearn.datasets import fetch_20newsgroups
    
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    text = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    #%%
    print text.data [ 572  ] , '---------------'       
    print text.data [ 596  ] , '---------------'       
    print text.data [ 703  ] , '---------------'       
    print text.data [ 772  ] , '---------------'       
    print text.data [ 832  ] , '---------------'       
    print text.data [ 861  ] , '---------------'       
    print text.data [ 868  ] , '---------------'       
    print text.data [ 887  ] , '---------------'       
    print text.data [ 1093 ] , '---------------'       
    print text.data [ 1161 ] , '---------------'       
    print text.data [ 1201 ] , '---------------'       
    print text.data [ 1614 ] , '---------------'       
    print text.data [ 1647 ] , '---------------'       
    print text.data [ 1651 ] , '---------------'       
    print text.data [ 1682 ] , '---------------'       
    print text.data [ 1691 ] , '---------------'       
    print text.data [ 1739 ] , '---------------'       
    print text.data [1819  ] , '---------------'       
    print text.data [ 2002 ] , '---------------'       
    print text.data [ 2243 ] , '---------------'       
    
    
    
    #%%
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer() #stop_words='english')
    X_counts = count_vect.fit_transform(text.data)
    
    #%%
    #generate counts per document / word. sparse matix of [n_documents, n_features]
    print text.data[2256]
    print X_counts[2256:]
    print X_counts[2256,6507]
    print count_vect.get_feature_names()[32142]
    
    doc_counts = count_vect.transform(['Samer is my name', 'i live in Israel', 'der alasad is my village'])
    
    print count_vect.vocabulary_['berkeley'], count_vect.vocabulary_['my'], count_vect.vocabulary_['name']
    
    print count_vect.get_feature_names()[32142]
    
    
    
    ##%%
    #text = text.split('\n')
    #tweets = {}
    #dim = 0
    #for t in text:
    #    (tweetID, words) = t.split(', ')
    #    words = words.strip().split(' ')
    #    sparse = {}
    #    for w in words:
    #        w = int(w)
    #        sparse[w] = sparse.get(w,0) + 1
    #        
    #    tmp = max(sparse.keys())
    #    if tmp>dim:
    #        dim=tmp
    #    tweets[tweetID] = sparse
    
    return X_counts, text

#%%
import LSH_2 as lsh

X_counts, text = wikidocs()

dim = X_counts.shape[1]

n = 3
hp = 2**n
maxB = 50
tables = 6
ll = lsh.LSH(dimensionSize=dim, numberTables=tables, hyperPlanesNumber=hp, maxBucketSize=maxB)

#%%
for sample in range(X_counts.shape[0]):
    doc = X_counts[sample, :]
    print 'Adding document ', sample, ' out of ', X_counts.shape[0], ' shape = ', doc.shape
    nearest, dist = ll.add(sample, doc)
    
    
#%%

ll.myprint()



#%%

print [(x, len(ll.hList[0].buckets[x])) for x in ll.hList[0].buckets]
       
       
print ll.hList[0].buckets['00000000000000000000000000001001']