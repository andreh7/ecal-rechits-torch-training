#!/usr/bin/env python

# makes an animation with the ROC curves vs. epoch

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

figures = []
highestTPRs = []

xmax = 0.05

for epochNumber in range(1, maxEpochNumber+1):

    fig = plt.figure(facecolor="white")
    figures.append(fig)

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):
        
        # take the last epoch
        fpr, tpr = plotROCs.drawSingleROCcurve(rocFnames[sample][epochNumber - 1], sample, color, '-', 2)
        updateHighestTPR(highestTPRs, fpr, tpr, xmax)

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr = plotROCs.drawSingleROCcurve(fname, "MVA " + sample, color, '--', 1)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)            

    # end of loop over test/train

# end of loop over epochs

# update the x and y ranges
if xmax != None:
    for fig in figures:
        # switch to the figure
        plt.figure(fig.number)

        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

#----------

def animate(t):
    global last_i, last_frame

    i = int(t)
    if i == last_i:
        return last_frame

    fig = figures[i]

    last_frame = mplfig_to_npimage(fig)
    return last_frame

#----------
duration = len(figures)
fps = 15
animation = mpy.VideoClip(animate, duration=duration)
animation.write_videofile("/tmp/test.mp4", fps=fps)
