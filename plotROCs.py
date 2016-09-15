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
import numpy as np

#----------------------------------------------------------------------

def addTimestamp(inputDir, x = 0.0, y = 1.07, ha = 'left', va = 'bottom'):

    import pylab, time


    # static variable
    if not hasattr(addTimestamp, 'text'):
        # make all timestamps the same during one invocation of this script

        now = time.time()

        addTimestamp.text = time.strftime("%a %d %b %Y %H:%M", time.localtime(now))

        # use the timestamp of the samples.txt file
        # as the starting point of the training
        # to determine the wall clock time elapsed
        # for the training

        fname = os.path.join(inputDir, "samples.txt")
        if os.path.exists(fname):
            startTime = os.path.getmtime(fname)
            deltaT = now - startTime

            addTimestamp.text += " (%.1f days)" % (deltaT / 86400.)


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

    if inputDir.endswith('/'):
        inputDir = inputDir[:-1]

    pylab.gca().text(x, y, inputDir,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )

#----------------------------------------------------------------------

def addNumEvents(numEventsTrain, numEventsTest):

    for numEvents, label, x0, halign in (
        (numEventsTrain, 'train', 0.00, 'left'),
        (numEventsTest, 'test',   1.00, 'right'),
        ):

        if numEvents != None:
            pylab.gca().text(x0, -0.08, '# ' + label + ' ev.: ' + str(numEvents),
                             horizontalalignment = halign,
                             verticalalignment = 'center',
                             transform = pylab.gca().transAxes,
                             fontsize = 10,
                             )

#----------------------------------------------------------------------

def readROC(fname):
    # reads a torch file and calculates the area under the ROC
    # curve for it
    # 
    # also looks for a cached file

    
    if fname.endswith(".cached-auc.py"):
        # read the cached file
        print "reading",fname
        auc = float(open(fname).read())
        return auc

    print "reading",fname
    
    if fname.endswith(".npz"):
        data = np.load(fname)

        weights = data['weight']
        labels  = data['label']
        outputs = data['output']

    else:
        # assume this is a torch file
        fin = torchio.InputFile(fname, "binary")
        data = fin.readObject()

        weights = data['weight'].asndarray()
        labels  = data['label'].asndarray()
        outputs = data['output'].asndarray()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)

    aucValue = auc(fpr, tpr, reorder = True)

    # write to cache
    cachedFname = fname + ".cached-auc.py"
    fout = open(cachedFname,"w")
    print >> fout,aucValue
    fout.close()

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

            fname = fname.replace("_rechits","")

            description.append(fname)

        return ", ".join(description)

    else:
        return None

#----------------------------------------------------------------------

def readROCfiles(inputDir, transformation = None, includeCached = False, maxEpoch = None):
    # returns mvaROC, rocValues
    # which are dicts of 'test'/'train' to the single value
    # (for MVAid) or a dict epoch -> values (rocValues)
    #
    # transformation is a function taking the file name 
    # which is run on each file
    # found and stored in the return values. If None,
    # just the name is stored.

    if transformation == None:
        transformation = lambda fname: fname

    #----------
    inputFiles = []

    if includeCached:
        # read cached version first
        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.t7.cached-auc.py")) 
        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.t7.bz2.cached-auc.py")) 
        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz.cached-auc.py")) 

    inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.t7")) 
    inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.t7.bz2")) 
    inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz")) 

    if not inputFiles:
        print >> sys.stderr,"no files roc-data-* found, exiting"
        sys.exit(1)

    # ROCs values and epoch numbers for training and test
    # first index is 'train or 'test'
    # second index is epoch number
    rocValues    = dict(train = {}, test = {})

    # MVA id ROC areas
    mvaROC = dict(train = None, test = None)

    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  roc-data-test-mva.t7
        #  roc-data-train-0002.t7

        mo = re.match("roc-data-(\S+)-mva\.t7(\.gz|\.bz2)?$", basename)
        if not mo:
            mo = re.match("roc-data-(\S+)-mva\.npz$", basename)

        if mo:
            sampleType = mo.group(1)

            assert mvaROC.has_key(sampleType)
            assert mvaROC[sampleType] == None

            mvaROC[sampleType] = transformation(inputFname)

            continue

        mo = re.match("roc-data-(\S+)-(\d+)\.t7(\.gz|\.bz2)?$", basename)
        if not mo:
            mo = re.match("roc-data-(\S+)-(\d+)\.npz$", basename)

        if not mo and includeCached:
            mo = re.match("roc-data-(\S+)-(\d+)\.t7(\.gz|\.bz2)?\.cached-auc\.py$", basename)

        if not mo and includeCached:
            mo = re.match("roc-data-(\S+)-(\d+)\.npz\.cached-auc\.py$", basename)

        if mo:
            sampleType = mo.group(1)
            epoch = int(mo.group(2), 10)

            if maxEpoch == None or epoch <= maxEpoch:
                if rocValues[sampleType].has_key(epoch):
                    # skip reading this file: we already have a value
                    # (priority is given to the cached files)
                    continue

                rocValues[sampleType][epoch] = transformation(inputFname)
            continue

        print >> sys.stderr,"WARNING: unmatched filename",inputFname

    return mvaROC, rocValues

#----------------------------------------------------------------------

def drawSingleROCcurve(inputFname, label, color, lineStyle, linewidth):

    print "reading",inputFname

    if inputFname.endswith(".npz"):
        # numpy file
        data = np.load(inputFname)

        weights = data['weight']
        labels  = data['label']
        outputs = data['output']
    else:
        # assume this is a torch file
        fin = torchio.InputFile(inputFname, "binary")

        data = fin.readObject()

        weights = data['weight'].asndarray()
        labels  = data['label'].asndarray()
        outputs = data['output'].asndarray()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)
    auc = auc(fpr, tpr, reorder = True)

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, label = label.format(auc = auc))

    return fpr, tpr, len(weights)

#----------------------------------------------------------------------

def findLastCompleteEpoch(rocFnames, ignoreTrain):
    
    trainEpochNumbers = sorted(rocFnames['train'].keys())
    testEpochNumbers  = sorted(rocFnames['test'].keys())

    if not ignoreTrain and not trainEpochNumbers:
        print >> sys.stderr,"WARNING: no training files found"
        return None

    if not testEpochNumbers:
        print >> sys.stderr,"WARNING: no test files found"
        return None

    retval = testEpochNumbers[-1]
    
    if not ignoreTrain:
        retval = min(trainEpochNumbers[-1], retval)
    return retval

#----------------------------------------------------------------------
def updateHighestTPR(highestTPRs, fpr, tpr, maxfpr):
    if maxfpr == None:
        return

    # find highest TPR for which the FPR is <= maxfpr
    highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
    highestTPRs.append(highestTPR)

#----------------------------------------------------------------------
def drawLast(inputDir, description, xmax = None, ignoreTrain = False, maxEpoch = None, savePlots = False):
    # plot ROC curve for last epoch only
    pylab.figure(facecolor='white')
    
    # read only the file names
    mvaROC, rocFnames = readROCfiles(inputDir, maxEpoch = maxEpoch)

    #----------

    # find the highest epoch for which both
    # train and test samples are available

    epochNumber = findLastCompleteEpoch(rocFnames, ignoreTrain)

    highestTPRs = []
    #----------

    # maps from sample type to number of events
    numEvents = {}

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        if ignoreTrain and sample == 'train':
            continue
        
        # take the last epoch
        if epochNumber != None:
            fpr, tpr, numEvents[sample] = drawSingleROCcurve(rocFnames[sample][epochNumber], sample + " (auc {auc:.3f})", color, '-', 2)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)
            

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr, dummy = drawSingleROCcurve(fname, "MVA " + sample + " (auc {auc:.3f})", color, '--', 1)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)            

    pylab.xlabel('fraction of false positives')
    pylab.ylabel('fraction of true positives')

    if xmax != None:
        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

    pylab.grid()
    pylab.legend(loc = 'lower right')

    addTimestamp(inputDir)
    addDirname(inputDir)
    addNumEvents(numEvents.get('train', None), numEvents.get('test', None))

    if description != None:

        if epochNumber != None:
            description += " (epoch %d)" % epochNumber

        pylab.title(description)

    if savePlots:
        for suffix in (".png", ".pdf", ".svg"):
            outputFname = os.path.join(inputDir, "last-auc")

            if xmax != None:
                outputFname += "-%.2f" % xmax

            outputFname += suffix

            pylab.savefig(outputFname)
            print "saved figure to",outputFname

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

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

    parser.add_option("--both",
                      default = False,
                      action="store_true",
                      help="plot AUC evolution and last AUC curve",
                      )

    parser.add_option("--ignore-train",
                      dest = 'ignoreTrain',
                      default = False,
                      action="store_true",
                      help="do not look at train values",
                      )

    parser.add_option("--max-epoch",
                      dest = 'maxEpoch',
                      type = int,
                      default = None,
                      help="last epoch to plot (useful e.g. if the training diverges at some point)",
                      )

    parser.add_option("--save-plots",
                      dest = 'savePlots',
                      default = False,
                      action="store_true",
                      help="save plots in input directory",
                      )

    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

    inputDir = ARGV.pop(0)

    #----------

    description = readDescription(inputDir)

    import pylab

    if options.last or options.both:

        drawLast(inputDir, description, ignoreTrain = options.ignoreTrain, maxEpoch = options.maxEpoch, savePlots = options.savePlots)

        # zoomed version
        # autoscaling in y with x axis range manually
        # set seems not to work, so we implement
        # something ourselves..
        drawLast(inputDir, description, xmax = 0.05, ignoreTrain = options.ignoreTrain, maxEpoch = options.maxEpoch, savePlots = options.savePlots)


    if not options.last or options.both:
        #----------
        # plot evolution of area under ROC curve vs. epoch
        #----------

        mvaROC, rocValues = readROCfiles(inputDir, readROC, includeCached = True, maxEpoch = options.maxEpoch)

        print "plotting AUC evolution"

        pylab.figure(facecolor='white')

        for sample, color in (
            ('train', 'blue'),
            ('test', 'red'),
            ):

            if options.ignoreTrain and sample == 'train':
                continue

            # sorted by ascending epoch
            epochs = sorted(rocValues[sample].keys())
            aucs = [ rocValues[sample][epoch] for epoch in epochs ]

            pylab.plot(epochs, aucs, '-o', label = sample + " (last auc=%.3f)" % aucs[-1], color = color, linewidth = 2)

            # draw a line for the MVA id ROC if available
            auc = mvaROC[sample]
            if auc != None:
                pylab.plot( pylab.gca().get_xlim(), [ auc, auc ], '--', color = color, 
                            label = "MVA (auc=%.3f) %s" % (auc, sample))

        pylab.grid()
        pylab.xlabel('training epoch')
        pylab.ylabel('AUC')

        pylab.legend(loc = 'lower right')

        if description != None:
            pylab.title(description)

        addTimestamp(inputDir)
        addDirname(inputDir)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "auc-evolution" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname

    #----------

    if not options.last or options.both:

        #----------
        # plot correlation of train and test AUC
        #----------

        import plotAUCcorr

        plotAUCcorr.doPlot(inputDir, maxEpoch = options.maxEpoch)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "auc-corr" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname

    #----------

    pylab.show()


