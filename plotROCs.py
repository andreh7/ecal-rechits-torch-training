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
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

inputDir = ARGV.pop(0)

#----------

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

    description = ", ".join(description)
    
else:
    description = None


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

        mvaROC[sampleType] = readROC(inputFname)

        continue

    mo = re.match("roc-data-(\S+)-(\d+)\.t7$", basename)

    if mo:
        sampleType = mo.group(1)
        epoch = int(mo.group(2), 10)

        assert epochNumbers.has_key(sampleType)
        assert not epoch in epochNumbers[sampleType]

        rocValues[sampleType].append(readROC(inputFname))
        epochNumbers[sampleType].append(epoch)
        continue
        

    print >> sys.stderr,"WARNING: unmatched filename",inputFname

import pylab

pylab.figure()

for sample, color in (
    ('train', 'blue'),
    ('test', 'red'),
    ):

    assert len(epochNumbers[sample]) == len(rocValues[sample])

    # sort by increasing epoch
    epochs, aucs = zip(*sorted(zip(epochNumbers[sample], rocValues[sample])))

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

pylab.show()


