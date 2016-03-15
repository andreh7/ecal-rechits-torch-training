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
    # plot ROC curve for last epoch only
    pylab.figure()
    
    # read only the file names
    mvaROC, epochNumbers, rocFnames = readROCfiles(inputDir)

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):
        
        # take the last epoch
        if not rocFnames[sample]:
            print >> sys.stderr,"WARNING: no files found for", sample
            continue
        
        drawSingleROCcurve(rocFnames[sample][-1], sample, color, '-', 2)

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            drawSingleROCcurve(fname, "MVA " + sample, color, '--', 1)


    pylab.xlabel('fraction of fake photons')
    pylab.ylabel('fraction of true photons')

    pylab.grid()
    pylab.legend(loc = 'lower right')


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

#----------

if description != None:
    pylab.title(description)

pylab.show()


