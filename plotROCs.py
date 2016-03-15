#!/usr/bin/env python

# given a results-* directory, finds the 
# Torch tensor files with the network outputs
# for different epochs, calculates the ROC area
# and plots the progress

# given a table of serialized Torch labels,
# weights and target values, calculates and prints
# the area under the ROC curve (using sklearn)

import sys, os
sys.path.append(os.path.expanduser("~aholz/torchio"))
import torchio

import glob, re

#----------------------------------------------------------------------

def addTimestamp(x = 0.0, y = 1.07, ha = 'left', va = 'bottom'):

    import pylab, time

    # static variable
    if not hasattr(addTimestamp, 'text'):
        # make all timestamps the same during one invocation of this script
        addTimestamp.text = time.strftime("%a %d %b %Y %H:%M")

    pylab.gca().text(x, y, addTimestamp.text,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )
#----------------------------------------------------------------------

    
def addDirname(inputDir, x = 1.0, y = 1.07, ha = 'right', va = 'bottom'):

    import pylab
    pylab.gca().text(x, y, inputDir,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )


#----------------------------------------------------------------------

def readROC(fname):
    # reads a torch file and calculates the area under the ROC
    # curve for it

    print "reading",fname
    
    fin = torchio.InputFile(fname, "binary")

    data = fin.readObject()

    weights = data['weight'].asndarray()
    labels  = data['label'].asndarray()
    outputs = data['output'].asndarray()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)

    aucValue = auc(fpr, tpr, reorder = True)

    return aucValue

#----------------------------------------------------------------------

def readDescription(inputDir):
    descriptionFile = os.path.join(inputDir, "samples.txt")

    if os.path.exists(descriptionFile):

        description = []

        # assume that these are file names (of the training set)
        fnames = open(descriptionFile).read().splitlines()

        for fname in fnames:
            if not fname:
                continue

            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]

            if fname.endswith("-train"):
                fname = fname[:-6]
            elif fname.endswith("-test"):
                fname = fname[:-5]
            description.append(fname)

        return ", ".join(description)

    else:
        return None

#----------------------------------------------------------------------

def readROCfiles(inputDir, transformation = None):
    # returns mvaROC, epochNumbers, rocValues
    # which are dicts of 'test'/'train' to the single value
    # (for MVAid) or a list of values (epochNumbers and rocValues)
    #
    # transformation is a function taking the file name 
    # which is run on each file
    # found and stored in the return values. If None,
    # just the name is stored.

    if transformation == None:
        transformation = lambda fname: fname

    #----------
    inputFiles = glob.glob(os.path.join(inputDir, "roc-data-*.t7"))

    # ROCs values and epoch numbers for training and test
    rocValues    = dict(train = [], test = [])
    epochNumbers = dict(train = [], test = [])

    # MVA id ROC areas
    mvaROC = dict(train = None, test = None)

    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  roc-data-test-mva.t7
        #  roc-data-train-0002.t7

        mo = re.match("roc-data-(\S+)-mva\.t7$", basename)

        if mo:
            sampleType = mo.group(1)

            assert mvaROC.has_key(sampleType)
            assert mvaROC[sampleType] == None

            mvaROC[sampleType] = transformation(inputFname)

            continue

        mo = re.match("roc-data-(\S+)-(\d+)\.t7$", basename)

        if mo:
            sampleType = mo.group(1)
            epoch = int(mo.group(2), 10)

            assert epochNumbers.has_key(sampleType)
            assert not epoch in epochNumbers[sampleType]

            rocValues[sampleType].append(transformation(inputFname))
            epochNumbers[sampleType].append(epoch)
            continue


        print >> sys.stderr,"WARNING: unmatched filename",inputFname

    # sort by increasing epochs
    for sample in epochNumbers.keys():
        epochNumbers[sample], rocValues[sample] = zip(*sorted(zip(epochNumbers[sample], rocValues[sample])))        
    
    return mvaROC, epochNumbers, rocValues

#----------------------------------------------------------------------

def drawSingleROCcurve(inputFname, label, color, lineStyle, linewidth):

    print "reading",inputFname
    
    fin = torchio.InputFile(inputFname, "binary")

    data = fin.readObject()

    weights = data['weight'].asndarray()
    labels  = data['label'].asndarray()
    outputs = data['output'].asndarray()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, label = label)

    return fpr, tpr

#----------------------------------------------------------------------

def drawLast(inputDir, description, xmax = None):
    # plot ROC curve for last epoch only
    pylab.figure()
    
    # read only the file names
    mvaROC, epochNumbers, rocFnames = readROCfiles(inputDir)

    #----------

    def findLastCompleteEpoch():
        if not epochNumbers['train']:
            print >> sys.stderr,"WARNING: no training files found"
            return None

        if not epochNumbers['test']:
            print >> sys.stderr,"WARNING: no test files found"
            return None

        for sample in ('train','test'):
            assert epochNumbers[sample][0] == 1
            assert len(epochNumbers[sample]) == epochNumbers[sample][-1]

        return min(epochNumbers['train'][-1], epochNumbers['test'][-1])

    #----------

    # find the highest epoch for which both
    # train and test samples are available

    epochNumber = findLastCompleteEpoch()

    highestTPRs = []

    #----------
    def updateHighestTPR(fpr, tpr, maxfpr):
        if maxfpr == None:
            return

        # find highest TPR for which the FPR is <= maxfpr
        highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
        highestTPRs.append(highestTPR)
    #----------

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):
        
        # take the last epoch
        if epochNumber != None:
            fpr, tpr = drawSingleROCcurve(rocFnames[sample][epochNumber - 1], sample, color, '-', 2)
            updateHighestTPR(fpr, tpr, xmax)
            

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr = drawSingleROCcurve(fname, "MVA " + sample, color, '--', 1)
            updateHighestTPR(fpr, tpr, xmax)            



    pylab.xlabel('fraction of fake photons')
    pylab.ylabel('fraction of true photons')

    if xmax != None:
        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

    pylab.grid()
    pylab.legend(loc = 'lower right')

    addTimestamp()
    addDirname(inputDir)

    if description != None:

        if epochNumber != None:
            description += " (epoch %d)" % epochNumber

        pylab.title(description)


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser("""

  usage: %prog [options] result-directory

"""
)

parser.add_option("--last",
                  default = False,
                  action="store_true",
                  help="plot ROC curve for last epoch only",
                  )

(options, ARGV) = parser.parse_args()

assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

inputDir = ARGV.pop(0)

#----------

description = readDescription(inputDir)

import pylab

if options.last:

    drawLast(inputDir, description)

    # zoomed version
    # autoscaling in y with x axis range manually
    # set seems not to work, so we implement
    # something ourselves..
    drawLast(inputDir, description, xmax = 0.05)


else:
    # plot evolution of area under ROC curve vs. epoch

    mvaROC, epochNumbers, rocValues = readROCfiles(inputDir, readROC)

    pylab.figure()

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        assert len(epochNumbers[sample]) == len(rocValues[sample])

        # these are already sorted by ascending epoch
        epochs, aucs = epochNumbers[sample], rocValues[sample]

        pylab.plot(epochs, aucs, '-o', label = sample, color = color, linewidth = 2)

        # draw a line for the MVA id ROC if available
        auc = mvaROC[sample]
        if auc != None:
            pylab.plot( pylab.gca().get_xlim(), [ auc, auc ], '--', color = color, 
                        label = "MVA " + sample)

    pylab.grid()
    pylab.xlabel('training epoch')
    pylab.ylabel('AUC')

    pylab.legend(loc = 'lower right')

    if description != None:
        pylab.title(description)

    addTimestamp()
    addDirname(inputDir)

#----------


pylab.show()


