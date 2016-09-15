#!/usr/bin/env python

# given a results-* directory, draws train AUC
# vs. test AUC

import sys, os
import plotROCs
import scipy.stats
import pylab

#----------------------------------------------------------------------

def doPlot(inputdir, maxEpoch = None):

    description = plotROCs.readDescription(inputDir)

    #----------
    # plot evolution of area under ROC curve vs. epoch
    #----------

    mvaROC, rocValues = plotROCs.readROCfiles(inputDir, plotROCs.readROC, includeCached = True, maxEpoch = maxEpoch)

    pylab.figure(facecolor='white')

    epochs = sorted(rocValues['train'].keys())

    trainAUCs = [ rocValues['train'][epoch] for epoch in epochs ]
    testAUCs  = [ rocValues['test'][epoch] for epoch in epochs ]

    # calculate Pearson's linear correlation coefficient
    rho, pvalue = scipy.stats.pearsonr(trainAUCs, testAUCs)

    print "plotting"

    pylab.plot(trainAUCs, testAUCs, 'o', label = 'rho = %.3f' % rho)

    pylab.grid()
    pylab.xlabel('train AUC')
    pylab.ylabel('test AUC')

    pylab.legend(loc = 'upper left')

    if description != None:
        pylab.title(description)

    plotROCs.addTimestamp(inputDir)
    plotROCs.addDirname(inputDir)

    #----------


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser("""

      usage: %prog [options] result-directory

    """
    )

    parser.add_option("--max-epoch",
                      dest = 'maxEpoch',
                      type = int,
                      default = None,
                      help="last epoch to plot (useful e.g. if the training diverges at some point)",
                      )

    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotAUCcorr.py result-directory"

    inputDir = ARGV.pop(0)

    #----------

    doPlot(inputDir, maxEpoch = options.maxEpoch)

    pylab.show()


