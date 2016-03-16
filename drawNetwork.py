#!/usr/bin/env python

# given an output directory, takes the latest stored network
# and draws it

import sys, os
sys.path.append(os.path.expanduser("~/torchio"))
import torchio

import glob, re
import pyx

#----------------------------------------------------------------------

# TODO: for the moment this is hardwired but in the long term
#       this should be stored along with the network structure
# (number_of_layers, width, height)
inputSize = (1, 7, 23)

#----------------------------------------------------------------------
import numpy as np

class DataShapeLayer:
    # represents a data layer between the operations

    def __init__(self, numPlanes, widthInPixels, heightInPixels):

        self.numPlanes = numPlanes
        self.width     = widthInPixels
        self.height    = heightInPixels

#----------------------------------------------------------------------    
    
class NetworkDrawer:
    # this mostly draws the data shapes between
    # the operations, not the operations themselves
    
    #----------------------------------------
    
    def __init__(self, canvas):
        self.canvas = canvas

        self.betweenLayersSpaceHorizontal = 1.

        # space we use for each data layer
        self.layerGridWidth = 3.
        self.layerGridHeight = 3.

        # limit on the number of parallel planes
        # drawn
        #
        # this is mostly neede for the fully connected
        # layers where we can have many hidden
        # nodes in a layer
        self.maxPlanesToDraw = 15
        
        self.layers = []

        # displacement for drawing 'with perspective'
        # corresponding to some displacement in z
        # TODO: must determine this such that
        #       the overall figure fits

        self.dz = np.array([-0.1, 0.1])
        
    #----------------------------------------

    def __advancePos(self):
        self.pos[0] += (self.layerGridWidth + self.betweenLayersSpaceHorizontal)

    #----------------------------------------
        
    ### def newline(self):
    ###     self.pos[1] -= (self.gridWidth * self.cubeSize + self.betweenCubesSpace)
    ###     self.pos[0] = 0.

    #----------------------------------------

    def addDataLayer(self, numPlanes, width, height):
        # width and height are in pixels
        self.layers.append(DataShapeLayer(numPlanes, width, height))

    #----------------------------------------

    def __drawDataShapeLayer(self, layer):
        numPlanes = min(self.maxPlanesToDraw, layer.numPlanes)

        # center the different data planes
        if numPlanes % 2 == 0:
            # even number of layers
            zmid = (numPlanes - 1) / 2.0
        else:
            # even number of layers
            zmid = numPlanes / 2

        zCenters = np.arange(numPlanes) - zmid

        # draw those at the highest z depth first
        # so that they are covered by the ones
        # at the lowest z depth

        halfWidth  = layer.width  / 2.0 * self.pixelSize
        halfHeight = layer.height / 2.0 * self.pixelSize

        # TODO: alternating colors
        colors = [
            pyx.color.grey(0.7),
            pyx.color.grey(0.5),
            ]
                
        for index, z in enumerate(zCenters[::-1]):

            # center of the rectangle
            centerPos = self.pos + z * self.dz

            left = centerPos[0] - halfWidth
            right = centerPos[0] + halfWidth
            top = centerPos[1] - halfHeight
            bottom = centerPos[1] + halfHeight

            path = pyx.path.path(
                pyx.path.moveto(left, top),
                pyx.path.lineto(right, top),
                pyx.path.lineto(right, bottom),
                pyx.path.lineto(left, bottom),
                pyx.path.closepath())

            thisColor = colors[index % len(colors)]

            # draw the rectangle
            if thisColor != None:
                self.canvas.stroke(path,
                                   [ pyx.deco.filled([ thisColor])])
            else:
                self.canvas.stroke(path)


                  
    #----------------------------------------

    def draw(self):
        # calculate sizes of planes and draw all components

        # initial position
        self.pos = np.array([0., 0.])

        # find the maximum number of pixels to be represented
        # in each direction
        numPixelsX = [ layer.width for layer in self.layers if isinstance(layer, DataShapeLayer) ]
        numPixelsY = [ layer.height for layer in self.layers if isinstance(layer, DataShapeLayer) ]

        # size we use for a pixel
        pixelSizeX = self.layerGridWidth / float(max(numPixelsX))
        pixelSizeY = self.layerGridHeight / float(max(numPixelsY))
        
        # make the pixels square
        self.pixelSize = min(pixelSizeX, pixelSizeY)

        for layer in self.layers:
            if isinstance(layer, DataShapeLayer):
                self.__drawDataShapeLayer(layer)
                self.__advancePos()

            
        
        
#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]
assert len(ARGV) == 1, "usage: drawNetwork.py results-dir"

resultsDir = ARGV.pop(0)

fnames = glob.glob(os.path.join(resultsDir, "model*.net"))

maxEpoch = None
maxEpochFname = None

# find the latest one
for fname in fnames:
    
    mo = re.match("model(\d+).net$", os.path.basename(fname))
    if not mo:
        print >> sys.stderr,"warning: unexpected filename", fname
        continue

    epoch = int(mo.group(1), 10)

    if maxEpoch == None or epoch > maxEpoch:
        maxEpoch = epoch
        maxEpochFname = fname


if not maxEpochFname:
    print >> sys.stderr,"no model files found"
    sys.exit(1)


#----------
    
network = torchio.read(maxEpochFname)
    
assert isinstance(network, torchio.nn.Sequential)

layers = [ item[1] for item in sorted(network.modules.items()) ]

canvas = pyx.canvas.canvas()

drawer = NetworkDrawer(canvas)

drawer.addDataLayer(inputSize[0], inputSize[1], inputSize[2])

for layerIndex, layer in enumerate(layers):
    print "layer",layerIndex,layer.__class__

    # TODO: from the network itself we can't infer the 
    #       shape of the input data

    if any([ isinstance(layer, theType) for theType in [
        torchio.nn.SpatialConvolutionMM,
        torchio.nn.ReLU,
        torchio.nn.SpatialMaxPooling,
        torchio.nn.Dropout,
        torchio.nn.Linear,
        torchio.nn.Sigmoid,
        torchio.nn.View,
        ]]):

        outputSize = list(layer.output.size)

        # some layers have only one dimension,
        # assume width 1 in the other dimensions
        while len(outputSize) < 3:
            outputSize += [ 1 ]

        # add the data layer at the output of this operations
        # instead of infering it ourselves, we rely on the 'output'
        # field to be present (which is probably only there
        # after some training ?!)

        drawer.addDataLayer(outputSize[0],
                            outputSize[1],
                            outputSize[2],
                            )
        continue
        

    else:
        print >> sys.stderr,"WARNING: unsupported layer type:",layer.__class__

#----------

drawer.draw()
        
drawer.canvas.writeGSfile("/tmp/test.png")
