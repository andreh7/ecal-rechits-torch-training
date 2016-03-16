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

    def __init__(self, numPlanes, widthInPixels, heightInPixels, description):

        self.numPlanes   = numPlanes
        self.width       = widthInPixels
        self.height      = heightInPixels
        self.description = description

class OperationLayer:
    # corresponds to the operation
    # between two data layers

    def __init__(self, description, kernelSize):
        # description must be a list of strings
        # kernelSize must be None or (width, height)

        self.description = description
        self.kernelSize = kernelSize

#----------------------------------------------------------------------    
    
class NetworkDrawer:
    # this mostly draws the data shapes between
    # the operations, not the operations themselves
    
    #----------------------------------------
    
    def __init__(self, canvas):
        self.canvas = canvas

        self.betweenLayersSpaceHorizontal = 0.5

        # space we use for each data layer
        self.layerGridWidth = 3.
        self.layerGridHeight = 3.

        # space between nominal top of
        # symbol drawing and first line
        # of description
        self.topDescriptionVerticalOffset = 3.0

        # same for below the drawings description
        self.bottomDescriptionVerticalOffset = 3.0

        self.descriptionTextVerticalSpacing = 0.4

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

        #----------

        # by how much the position is changed from
        # one data layer to the next
        self.advanceHorizontal = np.array([
            self.layerGridWidth + self.betweenLayersSpaceHorizontal,
            0])
        
    #----------------------------------------

    def __advancePos(self):

        self.pos += self.advanceHorizontal

    #----------------------------------------
        
    ### def newline(self):
    ###     self.pos[1] -= (self.gridWidth * self.cubeSize + self.betweenCubesSpace)
    ###     self.pos[0] = 0.

    #----------------------------------------

    def addDataLayer(self, numPlanes, width, height, description = None):
        # width and height are in pixels
        # 
        # description should be a string or a list of strings if
        # multiple lines should be shown


        # make sure this is either the input
        # or the previous layer was a operation layer
        assert len(self.layers) == 0 or isinstance(self.layers[-1], OperationLayer)

        if description == None:
            description = []
        elif isinstance(description, str):
            description = [ description ]

        # add size information
        if width == 1 and height == 1:
            description.append("%d" % numPlanes)
        else:
            description.append("%d @ %d x %d" % (numPlanes, width, height))

        self.layers.append(DataShapeLayer(numPlanes, width, height, description))

    #----------------------------------------

    def addOperationLayer(self, description, kernelSize = None):
        # kernelSize must be (height, width) or None
        # 
        # description should either be a string or a list of strings
       
        # make sure the previous layer is a data layer
        assert len(self.layers) > 0 and isinstance(self.layers[-1], DataShapeLayer)

        
        if description == None:
            description = []
        elif isinstance(description, str):
            description = [ description ]

        if kernelSize != None:
            description.append("%dx%d kernel" % (kernelSize[0], kernelSize[1]))

        self.layers.append(OperationLayer(description, kernelSize))

    #----------------------------------------

    def __getPlanecenters(self, numPlanes):
        # returns an array of numpy 2D arrays (coordinates)
        # relative to (0,0)
        # 
        # the position with the highest z is first
        # (so that this is in drawing order)

        # center the different data planes
        if numPlanes % 2 == 0:
            # even number of layers
            zmid = (numPlanes - 1) / 2.0
        else:
            # even number of layers
            zmid = numPlanes / 2

        zCenters = np.arange(numPlanes) - zmid

        return [
            z * self.dz
            for z in zCenters[::-1]
            ]
        
    #----------------------------------------

    def __drawDataShapeLayer(self, layer):
        numPlanes = min(self.maxPlanesToDraw, layer.numPlanes)

        zCenters = self.__getPlanecenters(numPlanes)

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
                
        for index, centerPos in enumerate(zCenters):

            # center of the rectangle
            centerPos += self.pos

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

            #----------
            # draw the description text above the layer
            #----------
            if layer.description:

                # starting position
                x = self.pos[0]
                y = self.pos[1] + self.layerGridHeight / 2.0 + self.topDescriptionVerticalOffset / 2.0

                for line in layer.description[::-1]:

                    self.canvas.text(x, y, line, [pyx.text.halign.boxcenter])


                    y += self.descriptionTextVerticalSpacing


    #----------------------------------------

    def __drawOperationLayerDescription(self, layer):

        #----------
        # put the description between the data layers
        # below the drawings
        #----------

        if layer.description:

            # starting position
            x = self.pos[0] + (self.layerGridWidth + self.betweenLayersSpaceHorizontal) / 2.0
            y = self.pos[1] - self.layerGridHeight / 2.0 - self.bottomDescriptionVerticalOffset / 2.0

            for line in layer.description:

                self.canvas.text(x, y, line, [pyx.text.halign.boxcenter])

                y -= self.descriptionTextVerticalSpacing


    #----------------------------------------

    def __drawOperationLayerKernel(self, layerIndex):

        operationLayer = self.layers[layerIndex]
        kernelSize = operationLayer.kernelSize
        if not kernelSize:
            return

        leftDataLayer = self.layers[layerIndex - 1]
        rightDataLayer = self.layers[layerIndex + 1]

        numPlanesLeft  = min(self.maxPlanesToDraw, leftDataLayer.numPlanes)
        numPlanesRight = min(self.maxPlanesToDraw, rightDataLayer.numPlanes)

        # find the center of the front most
        # layer of this and the next layer
        leftCenter  = self.__getPlanecenters(numPlanesLeft)[-1] + self.pos
        rightCenter = self.__getPlanecenters(numPlanesRight)[-1] + self.pos + self.advanceHorizontal

        # source: for the moment, put the kernel
        # window into the bottom left corner
        # shifted away by one pixel
        # from the corner

        left   = leftCenter[0] + (- leftDataLayer.width / 2.0 + 1) * self.pixelSize
        right  = leftCenter[0] + (- leftDataLayer.width / 2.0 + 1 + kernelSize[0]) * self.pixelSize

        top    = leftCenter[1] + (- leftDataLayer.height / 2.0 + 1) * self.pixelSize
        bottom = leftCenter[1] + (- leftDataLayer.height / 2.0 + 1 + kernelSize[1]) * self.pixelSize

        path = pyx.path.path(
            pyx.path.moveto(left, top),
            pyx.path.lineto(right, top),
            pyx.path.lineto(right, bottom),
            pyx.path.lineto(left, bottom),
            pyx.path.closepath())

        self.canvas.stroke(path)

        if True:
            # draw lines from each of the four corners of the source
            # kernel window to the center of the output layer
            for source in (
               #  [ left, top ],
               [ right, top ],

                [ left, bottom ],
                # [ right, bottom ],
                ):
                path = pyx.path.path(
                    pyx.path.moveto(source[0], source[1]),
                    pyx.path.lineto(rightCenter[0], rightCenter[1]),
                    pyx.path.closepath())

                self.canvas.stroke(path)

        if False:
            # draw line from the center of the source window
            # to the center of the destination plane

            path = pyx.path.path(
                    pyx.path.moveto(0.5 * (left + right), 0.5 * (top + bottom)),
                    pyx.path.lineto(rightCenter[0], rightCenter[1]),
                    pyx.path.closepath())
            
            self.canvas.stroke(path)
        
        
    
    #----------------------------------------

    def __resetDrawingPosition(self):
        # initial position
        self.pos = np.array([0., 0.])

    #----------------------------------------

    def draw(self):
        # calculate sizes of planes and draw all components

        self.__resetDrawingPosition()

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

            elif isinstance(layer, OperationLayer):
                self.__drawOperationLayerDescription(layer)
                
                self.__advancePos()

            else:
                raise Exception("internal error")
        # end of loop over layers

        #----------
        # draw kernels after drawing all layers
        #----------
        
        self.__resetDrawingPosition()

        for layerIndex, layer in enumerate(self.layers):
            if isinstance(layer, DataShapeLayer):
                continue

            elif isinstance(layer, OperationLayer):
                self.__drawOperationLayerKernel(layerIndex)
                self.__advancePos()
            else:
                raise Exception("internal error")
        # end of loop over layers

        
        
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

drawer.addDataLayer(inputSize[0], inputSize[1], inputSize[2], 'input')

for layerIndex, layer in enumerate(layers):
    print "layer",layerIndex,layer.__class__

    #----------
    # add an operation layer
    #----------

    description = []
    kernelSize = None
    knownType = True
    
    if isinstance(layer, torchio.nn.SpatialConvolutionMM):
        kernelSize = (layer.kW, layer.kH)
        description = ["convolution"]

    elif isinstance(layer, torchio.nn.SpatialMaxPooling):

        kernelSize = (layer.kW, layer.kH)
        description = ["max pooling"]

    elif isinstance(layer, torchio.nn.Sigmoid):
        description = ["sigmoid"]

    elif isinstance(layer, torchio.nn.ReLU):
        description = ["ReLU"]

    elif isinstance(layer, torchio.nn.View):
        description = ["view"]

    elif isinstance(layer, torchio.nn.Linear):
        description = ["linear"]

    elif isinstance(layer, torchio.nn.Dropout):
        description = ["dropout"]
        description.append("p=%.2f" % layer.p)

    else:
        knownType = False
        print >> sys.stderr,"WARNING: unsupported layer type (operation):",layer.__class__

    if knownType:
        if hasattr(layer, 'weight'):
            description.append(' x '.join([str(x) for x in layer.weight.size]) + " weights")

        if hasattr(layer, 'bias'):
            description.append(' x '.join([str(x) for x in layer.bias.size]) + " biases")

        drawer.addOperationLayer(description,
                                 kernelSize = kernelSize,
                                 )

    else:
        drawer.addOperationLayer(None)
    

    #----------
    # data shape at output of this operation
    #----------

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

        description = []
        if layerIndex == len(layers) - 1:
            description.append("output")

        elif outputSize[1] == 1 and outputSize[2] == 1:
            description.append("hidden layer")
        else:
            description.append("feature maps")

        drawer.addDataLayer(outputSize[0],
                            outputSize[1],
                            outputSize[2],
                            description,
                            )


        

    else:
        print >> sys.stderr,"WARNING: unsupported layer type (output data):",layer.__class__

#----------

drawer.draw()
        
drawer.canvas.writeGSfile("/tmp/test.png", resolution = 300)
