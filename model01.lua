#!/usr/bin/env th

----------------------------------------------------------------------
-- model
----------------------------------------------------------------------

-- 2-class problem
noutputs = 1
ninputs = nfeats*width*height


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
                                  filtsize,           -- kernel width
                                  filtsize,           -- kernel height
                                  1,                  -- horizontal step size
                                  1,                  -- vertical step size
                                  (filtsize - 1) / 2, -- padW
                                  (filtsize - 1) / 2 -- padH
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

model:add(nn.Dropout(0.1))
-- model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Linear(nstates[2]*1*5, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))
