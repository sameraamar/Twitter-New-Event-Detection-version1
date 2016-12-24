# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:01:57 2016

@author: SAMERA
"""
from datetime import datetime
import sys
import threading

class simplelogger:
    DEBUG = 4
    INFO = 3
    WARN = 2
    ERROR = 1
    CRITICAL = 0
	
    handlers = None
    loglevels= None
    profiling = False
    profiling_res = {}
    helper = {}

    def init(self, filename=None, std_level=INFO, file_level=DEBUG, profiling=False, bufsize = 10):
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
                
    def warning(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.WARN, i):
                self.write(i, 'WARNING', message)
                    
    def error(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.ERROR, i):
                self.write(i, 'ERROR', message)
                
    def critical(self, message):
        for i in range(len(self.loglevels)):
            if self.checklevel(self.CRITICAL, i):
                self.write(i, 'CRITICAL', message)
        
    def turn_profiling(self, on=True):
        if on and not self.profiling:
            self.profiling_res = {}
            self.helper = {}
        self.profiling = on
                
    def entry(self, func):
        if not self.profiling:
            return 

        th = threading.current_thread()
        key = str(th) + func
            
#        dt = datetime.utcnow()
#        #datetime.datetime(2012, 12, 4, 19, 51, 25, 362455)
#        dt64 = np.datetime64(dt)
        ts = datetime.utcnow()
        
        th = threading.current_thread()
        self.helper[key] = ts.timestamp()

        #s = datetime.fromtimestamp(ts)
        self.debug('Entry: {0} at {1} [Thread {2}]'.format(func, ts, str(th)))
        
    def exit(self, func):
        if not self.profiling:
            return 
            
        th = threading.current_thread()
        key = str(th) + func
        entry_time = self.helper.get(key, None)
        
        assert(entry_time != None)
        self.helper[key] = None

        ts = datetime.utcnow()

        temp, count = self.profiling_res.get(key, (0, 0))
        self.profiling_res[key] = (temp + ts.timestamp() - entry_time, count+1)

        self.debug('Exit: {0} at {1} [Thread {2}]'.format(func, ts, str(th)))

    def profiling_dump(self):
        for func in self.profiling_res:
            seconds, count = self.profiling_res[func]
            if seconds < 0.01:
                continue
            mins = int(seconds / 60)
            secs = seconds % 60
            self.info('invoked {1:10} times, total {2}\' {3:.2f}\'\' - {0}'.format(func, count, mins, secs))
        
    def write(self, handler, levelname, message):
        dt = datetime.now()
        text = '{:{dfmt} {tfmt}}'.format(dt, dfmt='%Y-%m-%d', tfmt='%H:%M')
        text += ' ' + levelname + ': ' + message + '\n'
        

        #if 'c:/temp/0000010_docs_round_00.log' == self.handlers[1].name:
        #    debug = 1

        try:
            self.handlers[handler].write(text)
        except UnicodeEncodeError as e:
            tmp = text.encode('ascii', 'ignore')
            tmp = tmp.decode('utf-8')
            self.handlers[handler].write(tmp)
        except Exception as unexpected:
            #print('Received error ({0}) while writing to log file ({2})'.format(unexpected, text, self.handlers[handler]))
            raise
        
    def flush(self):
        if self.handlers[1] != None:
            self.handlers[1].flush() #.close()
            
    def close(self):
        if self.handlers[1] != None:
            self.handlers[1].close()
            #print('Log file ({0}) is closed now'.format(self.handlers[1]))
            

