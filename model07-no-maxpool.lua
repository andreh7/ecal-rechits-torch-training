#!/usr/bin/env th

-- like model06 but with dropout layer only applied
-- to the rechits variables, not the other (track iso)
-- variables

require 'PrintingLayer'

----------------------------------------------------------------------
-- model
----------------------------------------------------------------------

-- 2-class problem
noutputs = 1
ninputs = nfeats*width*height

-- dummy input to get the size of the output after a given layer
dummyInput = torch.zeros(nfeats, width, height)

-- hidden units, filter sizes for convolutional network
nstates = {64,64,128}
filtsize = 5
poolsize = 2

----------------------------------------------------------------------
-- a typical modern convolution network (conv+relu+pool)
recHitsModel = nn.Sequential()

-- see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialModules
-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
recHitsModel:add(nn.SpatialConvolutionMM(nfeats,             -- nInputPlane
                                  nstates[1],         -- nOutputPlane
                                  filtsize,           -- kernel width
                                  filtsize,           -- kernel height
                                  1,                  -- horizontal step size
                                  1,                  -- vertical step size
                                  (filtsize - 1) / 2, -- padW
                                  (filtsize - 1) / 2 -- padH
                            ))
recHitsModel:add(nn.ReLU())


-- no maxpooling layer here


-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
recHitsModel:add(nn.SpatialConvolutionMM(nstates[1],         -- nInputPlane
                                  nstates[2],         -- nOutputPlane
                                  3,                  -- kernel width
                                  3,                  -- kernel height
                                  1,                  -- horizontal step size
                                  1,                  -- vertical step size
                                  (3 - 1) / 2, -- padW
                                  (3 - 1) / 2 -- padH
                            ))
recHitsModel:add(nn.ReLU())

-- no maxpooling here

-- stage 3 : standard 2-layer neural network

-- see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
-- recHitsModel:add(nn.View(nstates[2]*filtsize*filtsize))
recHitsModel:add(nn.View(nstates[2]*width*height))

recHitsModel:add(nn.Dropout(0.5))

----------
-- track isolation variables 
----------
-- see e.g. http://stackoverflow.com/questions/32630635/torch-nn-handling-text-and-numeric-input


trackIsoModelChosen = nn.Identity()
trackIsoModelWorst  = nn.Identity() 

----------
-- put rechits and track iso networks in parallel
----------
parallelModel = nn.ParallelTable()
parallelModel:add(recHitsModel)
parallelModel:add(trackIsoModelChosen)
parallelModel:add(trackIsoModelWorst)

----------
-- common output part
----------
outputModel = nn.Sequential()

outputModel:add(nn.Linear(nstates[2]*width*height + 2, -- +2 for the track isolation variables
                                          nstates[3]))
outputModel:add(nn.ReLU())
outputModel:add(nn.Linear(nstates[3], noutputs))

----------
-- put the pieces together
----------
model = nn.Sequential()
model:add(parallelModel)
model:add(nn.JoinTable(1))
model:add(outputModel)


