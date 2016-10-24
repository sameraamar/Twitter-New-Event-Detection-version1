# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:01:57 2016

@author: SAMERA
"""
import datetime
import sys

class simplelogger:
    DEBUG = 3
    INFO = 2
    WARN = 1
    ERROR = 0
    handlers = None
    loglevels= None

    def init(self, filename=None, std_level=INFO, file_level=DEBUG):
        self.loglevels = [std_level, file_level]
        
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
                
    def write(self, handler, levelname, message):
        dt = datetime.datetime.now()
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
            

