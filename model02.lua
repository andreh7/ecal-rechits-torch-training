#!/usr/bin/env th

----------------------------------------------------------------------
-- model
--
-- non convolutional network
----------------------------------------------------------------------

-- 2-class problem
noutputs = 1
ninputs = nfeats * width * height

----------------------------------------------------------------------
model = nn.Sequential()

-- see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
model:add(nn.View(ninputs))

-- model:add(nn.Dropout(0.3))
model:add(nn.ReLU())
model:add(nn.Linear(ninputs, ninputs*2))
model:add(nn.ReLU())
model:add(nn.Linear(ninputs*2, ninputs*2))
model:add(nn.ReLU())
model:add(nn.Linear(ninputs*2, noutputs))
