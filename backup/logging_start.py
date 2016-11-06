# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:49:12 2016

@author: SAMERA
"""

import logging
import logging.config
import time

before = time.time()

logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('LSH')

# 'application' code
logger.debug('debug message %d something', 3)
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

time.sleep(2)

logger.critical('test: %.8f', (time.time()-before))


perc = 0.7
t = 1.99928
print ('%.2f %% (time: %.8f)' % (perc, t))