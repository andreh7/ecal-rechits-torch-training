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

def readROC(fname):
    # reads a torch file and calculates the area under the ROC
    # curve for it
    # 
    # also looks for a cached file


    cachedFname = fname + ".cached-auc.py"

    if os.path.exists(cachedFname):
        print "reading",cachedFname
        auc = float(open(cachedFname).read())
        return auc

    print "reading",fname
    
    fin = torchio.InputFile(fname, "binary")

    data = fin.readObject()

    weights = data['weight'].asndarray()
    labels  = data['label'].asndarray()
    outputs = data['output'].asndarray()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)

    aucValue = auc(fpr, tpr, reorder = True)

    # write to cache
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
    inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.t7.bz2")) 

    if not inputFiles:
        print >> sys.stderr,"no files roc-data-* found, exiting"
        sys.exit(1)

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

        mo = re.match("roc-data-(\S+)-mva\.t7(\.gz|\.bz2)?$", basename)

        if mo:
            sampleType = mo.group(1)

            assert mvaROC.has_key(sampleType)
            assert mvaROC[sampleType] == None

            mvaROC[sampleType] = transformation(inputFname)

            continue

        mo = re.match("roc-data-(\S+)-(\d+)\.t7(\.gz|\.bz2)?$", basename)

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

        # in some cases we have e.g. no files for one of the samples (train or test)
        # avoid zip raising an exception
        if len(epochNumbers[sample]) > 0 and len(rocValues[sample]) > 0:
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
    auc = auc(fpr, tpr, reorder = True)

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, label = label.format(auc = auc))

    return fpr, tpr

#----------------------------------------------------------------------

def findLastCompleteEpoch(epochNumbers, ignoreTrain):
    
    if not ignoreTrain and not epochNumbers['train']:
        print >> sys.stderr,"WARNING: no training files found"
        return None

    if not epochNumbers['test']:
        print >> sys.stderr,"WARNING: no test files found"
        return None

    for sample in ('train','test'):
        if ignoreTrain and sample == 'train':
            continue

        assert epochNumbers[sample][0] == 1
        assert len(epochNumbers[sample]) == epochNumbers[sample][-1]

    retval = epochNumbers['test'][-1]
    
    if not ignoreTrain:
        retval = min(epochNumbers['train'][-1], retval)
    return retval

#----------------------------------------------------------------------
def updateHighestTPR(highestTPRs, fpr, tpr, maxfpr):
    if maxfpr == None:
        return

    # find highest TPR for which the FPR is <= maxfpr
    highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
    highestTPRs.append(highestTPR)

#----------------------------------------------------------------------
def drawLast(inputDir, description, xmax = None, ignoreTrain = False):
    # plot ROC curve for last epoch only
    pylab.figure(facecolor='white')
    
    # read only the file names
    mvaROC, epochNumbers, rocFnames = readROCfiles(inputDir)

    #----------

    # find the highest epoch for which both
    # train and test samples are available

    epochNumber = findLastCompleteEpoch(epochNumbers, ignoreTrain)

    highestTPRs = []
    #----------

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        if ignoreTrain and sample == 'train':
            continue
        
        # take the last epoch
        if epochNumber != None:
            fpr, tpr = drawSingleROCcurve(rocFnames[sample][epochNumber - 1], sample + " (auc {auc:.2f})", color, '-', 2)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)
            

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr = drawSingleROCcurve(fname, "MVA " + sample + " (auc {auc:.2f})", color, '--', 1)
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

    if description != None:

        if epochNumber != None:
            description += " (epoch %d)" % epochNumber

        pylab.title(description)


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

    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

    inputDir = ARGV.pop(0)

    #----------

    description = readDescription(inputDir)

    import pylab

    if options.last or options.both:

        drawLast(inputDir, description, ignoreTrain = options.ignoreTrain)

        # zoomed version
        # autoscaling in y with x axis range manually
        # set seems not to work, so we implement
        # something ourselves..
        drawLast(inputDir, description, xmax = 0.05, ignoreTrain = options.ignoreTrain)


    if not options.last or options.both:
        # plot evolution of area under ROC curve vs. epoch

        mvaROC, epochNumbers, rocValues = readROCfiles(inputDir, readROC)

        pylab.figure(facecolor='white')

        for sample, color in (
            ('train', 'blue'),
            ('test', 'red'),
            ):

            if options.ignoreTrain and sample == 'train':
                continue

            assert len(epochNumbers[sample]) == len(rocValues[sample])

            # these are already sorted by ascending epoch
            epochs, aucs = epochNumbers[sample], rocValues[sample]

            pylab.plot(epochs, aucs, '-o', label = sample + " (last auc=%.2f)" % aucs[-1], color = color, linewidth = 2)

            # draw a line for the MVA id ROC if available
            auc = mvaROC[sample]
            if auc != None:
                pylab.plot( pylab.gca().get_xlim(), [ auc, auc ], '--', color = color, 
                            label = "MVA (auc=%.2f) %s" % (auc, sample))

        pylab.grid()
        pylab.xlabel('training epoch')
        pylab.ylabel('AUC')

        pylab.legend(loc = 'lower right')

        if description != None:
            pylab.title(description)

        addTimestamp(inputDir)
        addDirname(inputDir)

    #----------


    pylab.show()


