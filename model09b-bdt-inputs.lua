#!/usr/bin/env th

----------------------------------------------------------------------
-- model
--
-- non convolutional network
----------------------------------------------------------------------


-- 2-class problem
noutputs = 1

-- 13 input variables
--   phoIdInput :
--     {
--       s4 : FloatTensor - size: 1299819
--       scRawE : FloatTensor - size: 1299819
--       scEta : FloatTensor - size: 1299819
--       covIEtaIEta : FloatTensor - size: 1299819
--       rho : FloatTensor - size: 1299819
--       pfPhoIso03 : FloatTensor - size: 1299819
--       phiWidth : FloatTensor - size: 1299819
--       covIEtaIPhi : FloatTensor - size: 1299819
--       etaWidth : FloatTensor - size: 1299819
--       esEffSigmaRR : FloatTensor - size: 1299819
--       r9 : FloatTensor - size: 1299819
--       pfChgIso03 : FloatTensor - size: 1299819
--       pfChgIso03worst : FloatTensor - size: 1299819
--     }

ninputs = 13

nodesPerHiddenLayer = 100

numHiddenLayers = 10

-- size of minibatch
batchSize = 32

-- how many minibatches to unpack at a time
-- and to store in the GPU (to have fewer
-- data transfers to the GPU)
batchesPerSuperBatch = math.floor(6636386 / batchSize)


----------------------------------------------------------------------
model = nn.Sequential()

-- see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
model:add(nn.View(ninputs))


for i = 1, numHiddenLayers do

  model:add(nn.ReLU())

  if i == 1 then
    model:add(nn.Linear(ninputs, nodesPerHiddenLayer))
  elseif i == numHiddenLayers then

    -- add a dropout layer at the end
    -- model:add(nn.Dropout(0.3))
    model:add(nn.Linear(nodesPerHiddenLayer, noutputs))

  else

    model:add(nn.Linear(nodesPerHiddenLayer, nodesPerHiddenLayer))

  end -- if

end -- loop over hidden layers

----------------------------------------------------------------------
-- function to prepare input data samples
----------------------------------------------------------------------
function makeInput(dataset, rowIndices, inputDataIsSparse)

  assert(not inputDataIsSparse, "input data is not expected to be sparse")

  local batchSize = rowIndices:size()[1]

  local retval = torch.zeros(batchSize, ninputs)

  -- ----------

  for i=1,batchSize do

    local rowIndex = rowIndices[i]
    retval[i] = dataset.data[rowIndex]

  end -- loop over minibatch indices

  return retval

end

----------------------------------------------------------------------
function makeInputView(inputValues, first, last)

  assert(first >= 1)
  assert(last <= inputValues:size()[1])

  return inputValues:sub(first,last)

end

----------------------------------------------------------------------
