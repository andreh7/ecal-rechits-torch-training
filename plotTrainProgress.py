#!/usr/bin/env python


import sys
import re

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 1

trainAUCs = []
testAUCs = []

epochTimes = []

for line in open(ARGV[0]).readlines():

    ## if len(line) < 500:
    ##     print line

    mo = re.search("test AUC=\s+(\S+)\s*$", line)

    if mo:
        testAUCs.append(float(mo.group(1)))
        continue

    mo = re.search("train AUC=\s+(\S+)\s*$", line)

    if mo:
        trainAUCs.append(float(mo.group(1)))
        continue

    mo = re.search("time for entire batch:\s+(\S+)\s+min\s*$",line)
    if mo:
        epochTimes.append(float(mo.group(1)))

import pylab
import numpy as np

pylab.plot(range(1, len(trainAUCs) + 1), trainAUCs, '-o', label = 'train', color = 'blue', linewidth = 2);
pylab.plot(range(1, len(testAUCs) + 1),  testAUCs, '-o', label = 'test', color = 'red', linewidth = 2);
pylab.grid()
pylab.xlabel('training epoch')
pylab.ylabel('AUC')

x0 = 0.05
y0 = 0.95
dy = -0.05

pylab.text(x0, y0, 
           'training time per epoch: %.1f min' % np.mean(epochTimes),
           horizontalalignment='left',
           verticalalignment='top',
           transform = pylab.gca().transAxes)

pylab.legend(loc = 'lower right')
pylab.show()