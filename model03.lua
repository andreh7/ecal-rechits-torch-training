#!/usr/bin/env th

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
model = nn.Sequential()

-- see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialModules
-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats,             -- nInputPlane
                                  nstates[1],         -- nOutputPlane
                                  filtsize,           -- kernel width
                                  filtsize,           -- kernel height
                                  1,                  -- horizontal step size
                                  1,                  -- vertical step size
                                  (filtsize - 1) / 2, -- padW
                                  (filtsize - 1) / 2 -- padH
                            ))
model:add(nn.ReLU())

-- see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialMaxPooling
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1],         -- nInputPlane
                                  nstates[2],         -- nOutputPlane
                                  3,                  -- kernel width
                                  3,                  -- kernel height
                                  1,                  -- horizontal step size
                                  1,                  -- vertical step size
                                  (3 - 1) / 2, -- padW
                                  (3 - 1) / 2 -- padH
                            ))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, -- kernel width
                               poolsize, -- kernel height
                               poolsize, -- dW step size in the width (horizontal) dimension 
                               poolsize,  -- dH step size in the height (vertical) dimension
                               (poolsize - 1) / 2, -- pad size
                               (poolsize - 1) / 2 -- pad size
                         ))

-- stage 3 : standard 2-layer neural network

-- see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
-- model:add(nn.View(nstates[2]*filtsize*filtsize))
model:add(nn.View(nstates[2]*1*5))

model:add(nn.Dropout(0.5))
-- model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Linear(nstates[2]*1*5, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))

----------------------------------------------------------------------
-- function to prepare input data samples
----------------------------------------------------------------------
function makeInput(dataset, rowIndices, inputDataIsSparse)

  local batchSize = rowIndices:size()[1]

  local recHits

  if inputDataIsSparse then
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

  return recHits

end

----------------------------------------------------------------------
