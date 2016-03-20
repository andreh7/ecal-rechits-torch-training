#!/usr/bin/env python

# makes an animation with the ROC curves vs. epoch

import matplotlib
matplotlib.use('TkAgg')


import plotROCs
import sys
#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
ARGV = sys.argv[1:]
assert len(ARGV) == 1
inputDir = ARGV.pop(0)

import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

# read data 
mvaROC, epochNumbers, rocFnames = plotROCs.readROCfiles(inputDir)

maxEpochNumber = plotROCs.findLastCompleteEpoch(epochNumbers)

# for the moment, just draw all ROC curves and adjust the maximum y scale afterards
# (beware the high number of windows opened...)

# maps from epoch number to figure
figures = {}
highestTPRs = []

xmax = 0.05

import pylab

#----------

def makeFigure(epochNumber, ymax = None):
    # we make the figures on demand in order
    # not to exhaust memory

    print "epoch=",epochNumber

    # if epochNumber > 10:
    #     break

    fig = plt.figure(facecolor="white")

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):
        
        # take the last epoch
        fpr, tpr = plotROCs.drawSingleROCcurve(rocFnames[sample][epochNumber - 1], sample, color, '-', 2)
        plotROCs.updateHighestTPR(highestTPRs, fpr, tpr, xmax)

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr = plotROCs.drawSingleROCcurve(fname, "MVA " + sample, color, '--', 1)
            plotROCs.updateHighestTPR(highestTPRs, fpr, tpr, xmax)            

    # end of loop over test/train

    pylab.xlabel('fraction of fake photons')
    pylab.ylabel('fraction of true photons')

    pylab.title('epoch %d' % epochNumber)

    pylab.grid()
    pylab.legend(loc = 'lower right')

    if xmax != None:
        pylab.xlim(xmax = xmax)

    if ymax != None:
        # adjust y scale        
        pylab.ylim(ymax = ymax)

    return fig

#----------

# find the maximum y range if x is restricted
ymax = None
if xmax != None:
    
    # assume that the highest epoch has the highest y limit
    figures[maxEpochNumber] = makeFigure(maxEpochNumber)
    ymax = 1.1 * max(highestTPRs)

    plt.figure(figures[maxEpochNumber].number)
    pylab.ylim(ymax = ymax)

#----------

lastEpochNumber = None

epochsPerSecond = 2.0

def animate(t):
    # t seems to be in seconds

    global lastEpochNumber, last_frame

    epochNumber = int(t * epochsPerSecond) + 1
    epochNumber = min(epochNumber, maxEpochNumber)

    if epochNumber == lastEpochNumber:
        return last_frame

    if not figures.has_key(epochNumber):
        # produce this frame
        fig = makeFigure(epochNumber, ymax)
        figures[epochNumber] = fig

        # delete the old one
        if lastEpochNumber != None:
            figures[lastEpochNumber].clear()
            pylab.close(figures[lastEpochNumber])
            
            del figures[lastEpochNumber]

    fig = figures[epochNumber]
    lastEpochNumber = epochNumber

    last_frame = mplfig_to_npimage(fig)
    return last_frame

#----------

# duration in seconds (?)
duration = maxEpochNumber / float(epochsPerSecond) + 2
fps = 15
animation = mpy.VideoClip(animate, duration=duration)
animation.write_videofile("/tmp/test.mp4", fps=fps)
