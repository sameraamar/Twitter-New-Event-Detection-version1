# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:54:43 2016

@author: SAMERA
"""


import threading
 
class FuncThread(threading.Thread):
    results = None
    
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args 
        
        
    def run(self):
        assert( self._target!= None )
        self.results = self._target(*self._args)
        

def run_all(func_tupels):
    threads = []

    for (func, *args) in func_tupels:
        t = FuncThread(func, *args)
        t.start()
        threads.append(t)
        
    # Wait for all threads to complete
    #print ('********** waiting for threads to complete')
    results = []
    for t in threads:
        t.join()
    for t in threads:
        results.append(t.results)
        #print ('Finished: ' + str(t.results))
    #print ('********** threads are done')
    return results
        
## Example usage
#def someOtherFunc(data, key):
#    print ("data=%s; key=%s" % (str(data), str(key)))
#    c = 1
#    while(True):
#        c += 1
#        if c % 10000000 == 0:
#            break
#    print ('Finished')
#    
#threads = []
#    
#t1 = FuncThread(someOtherFunc, [1,2], 6)
#t1.start()
#threads.append(t1)
#
#t2 = FuncThread(someOtherFunc, [1,2], 6)
#t2.start()
#threads.append(t2)
#
## Wait for all threads to complete
#for t in threads:
#    t.join()

if __name__ == '__main__':
    def func1(data, key):
        print ("data=%s; key=%s" % (str(data), str(key)))
        c = 0
        while(True):
            c += 5
            if c % key == 0:
                break
        return c
        
    def func2(data, key):
        print ("data=%s; key=%s" % (str(data), str(key)))
        c = 1
        m = 1
        while(True):
            m *= c
            c += 1
            print(c, key)
            if c % key == 0:
                break
        return c,m
        
    print('\nregular call:\n')
    print(func1(444, 1000))
    print(func2(555, 500))
    print('\nnow let\'s try with threads:\n')
    
    results = run_all([ [func1, 444, 100] , (func2, 555, 100000) ])
    print (results)
    