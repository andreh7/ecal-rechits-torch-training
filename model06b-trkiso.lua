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

-- see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialMaxPooling
recHitsModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

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
recHitsModel:add(nn.SpatialMaxPooling(poolsize, -- kernel width
                               poolsize, -- kernel height
                               poolsize, -- dW step size in the width (horizontal) dimension 
                               poolsize,  -- dH step size in the height (vertical) dimension
                               (poolsize - 1) / 2, -- pad size
                               (poolsize - 1) / 2 -- pad size
                         ))

-- stage 3 : standard 2-layer neural network

-- see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
-- recHitsModel:add(nn.View(nstates[2]*filtsize*filtsize))
recHitsModel:add(nn.View(nstates[2]*1*5))

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

outputModel:add(nn.Linear(nstates[2]*1*5 + 2, -- +2 for the track isolation variables
                                          nstates[3]))
outputModel:add(nn.ReLU())
outputModel:add(nn.Linear(nstates[3], noutputs))

----------
-- put the pieces together
----------
model = nn.Sequential()
model:add(parallelModel)
model:add(nn.JoinTable(2,2))
model:add(outputModel)


----------------------------------------------------------------------
-- function to prepare input data samples
----------------------------------------------------------------------
function makeInput(dataset, rowIndices, inputDataIsSparse)

  local batchSize = rowIndices:size()[1]

  local input = {}
  local recHits

  if inputDataIsSparse then
    -- TODO: move this into a function so that we can also
    --       use it in test(..)

    -- ----------
    -- unpack the sparse data
    -- ----------

    -- TODO: can we move the creation of the tensor out of the loop ?
    --       seems to be 2x slower ?!
    --       also one has to pay attention to actually clear the vector here
    recHits = torch.zeros(batchSize, nfeats, width, height)

    for i=1,batchSize do

      local rowIndex = rowIndices[i]

      local indexOffset = dataset.data.firstIndex[rowIndex] - 1
  
      for recHitIndex = 1,dataset.data.numRecHits[rowIndex] do
  
        xx = dataset.data.x[indexOffset + recHitIndex] + recHitsXoffset
        yy = dataset.data.y[indexOffset + recHitIndex] + recHitsYoffset
  
        if xx >= 1 and xx <= width and yy >= 1 and yy <= height then
          recHits[{i, 1, xx, yy}] = dataset.data.energy[indexOffset + recHitIndex]
        end
  
      end -- loop over rechits of this photon
    end -- loop over minibatch indices


    -- ----------
  else
    -- rechits are not sparse
    assert(false, 'we should not come here currently')
    input = dataset.data[rowIndex]
  end

  table.insert(input, recHits)

  -- note that we use two dimensions here to be able
  -- to use JoinTable with minibatch
  table.insert(input, torch.zeros(batchSize,1))
  table.insert(input, torch.zeros(batchSize,1))

  for i=1,batchSize do
    local rowIndex = rowIndices[i]
    input[2][{i,1}] = dataset.chgIsoWrtChosenVtx[rowIndex]
    input[3][{i,1}] = dataset.chgIsoWrtWorstVtx[rowIndex]
  end

  return input

end

----------------------------------------------------------------------
