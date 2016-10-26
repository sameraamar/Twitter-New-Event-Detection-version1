# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:01:57 2016

@author: SAMERA
"""
from datetime import datetime
import sys
import numpy as np
import threading

class simplelogger:
    DEBUG = 3
    INFO = 2
    WARN = 1
    ERROR = 0
    handlers = None
    loglevels= None
    profiling = False
    profiling_res = {}
    helper = {}

    def init(self, filename=None, std_level=INFO, file_level=DEBUG, profiling=False):
        self.loglevels = [std_level, file_level]

        self.profiling_res = {}
        self.helper = {}
        self.turn_profiling( profiling )
        
        self.handlers = [sys.stdout]
        if filename != None:
            file = open(filename, 'w')
            self.handlers.append (file)
    
    def checklevel(self, level, handler):
        return self.loglevels[handler] >= level

    def info(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.INFO, i):
                self.write(i, 'INFO', message)

    def debug(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.DEBUG, i):
                self.write(i, 'DEBUG', message)
                
    def error(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.ERROR, i):
                self.write(i, 'ERROR', message)
    
    def turn_profiling(self, on=True):
        if on and not self.profiling:
            self.profiling_res = {}
            self.helper = {}
        self.profiling = on
                
    def entry(self, func):
        if not self.profiling:
            return 
            
#        dt = datetime.utcnow()
#        #datetime.datetime(2012, 12, 4, 19, 51, 25, 362455)
#        dt64 = np.datetime64(dt)
        ts = datetime.utcnow()
        
        th = threading.current_thread()
        self.helper[str(th) + func] = ts.timestamp()

        #s = datetime.fromtimestamp(ts)
        self.debug('Entry: {0} at {1}'.format(func, ts))
        
    def exit(self, func):
        if not self.profiling:
            return 
            
        th = threading.current_thread()
        key = str(th) + func
        entry_time = self.helper.get(key, None)
        
        assert(entry_time != None)
        self.helper[key] = None

        ts = datetime.utcnow()

        temp, count = self.profiling_res.get(func, (0, 0))
        self.profiling_res[func] = (temp + ts.timestamp() - entry_time, count+1)

        self.debug('Exit: {0} at {1}'.format(func, ts))

    def profiling_dump(self):
        for func in self.profiling_res:
            seconds, count = self.profiling_res[func]
            mins = int(seconds / 60)
            secs = seconds % 60
            self.info('invoked {1:10} times, total {2}\' {3:.2f}\'\' - {0}'.format(func, count, mins, secs))
        assert(len(self.profiling_res)>0)
        
    def write(self, handler, levelname, message):
        dt = datetime.now()
        text = '{:{dfmt} {tfmt}}'.format(dt, dfmt='%Y-%m-%d', tfmt='%H:%M')
        text += ' ' + levelname + ': ' + message + '\n'
        
        self.handlers[handler].write(text)
        
#    def debug(self, message):
#        if self.loglevel >= self.DEBUG:
#            dt = datetime.datetime.now()
#            text = 'DEBUG:'
#            text += '{:{dfmt} {tfmt}}'.format(dt, dfmt='%Y-%m-%d', tfmt='%H:%M:%S')
#            text += ' - ' + message + '\n'
#            
#            self.handler.write ( text)
#        
#    def error(self, message):
#        if self.loglevel >= self.ERROR:
#            dt = datetime.datetime.now()
#            text = 'ERROR:'
#            text += '{:{dfmt} {tfmt}}'.format(dt, dfmt='%Y-%m-%d', tfmt='%H:%M')
#            text += ' ' + message + '\n'
#            self.handler.write (text)
        
    def close(self):
        if self.handlers[1] != None:
            self.handlers[1].close()
            

