#!/usr/bin/env th

require 'torch' 
require 'nn'    
require 'optim'   
require 'os'

-- needed for AUC
-- metrics = require 'metrics';
require 'metrics'

----------------------------------------------------------------------
-- parameters
----------------------------------------------------------------------

-- TODO should use HT range 40-100 first
train_file = '../torch-utils/gjet-ht-400-600-test.t7'
test_file  = '../torch-utils/gjet-ht-400-600-train.t7'

-- input dimensions
nfeats = 1
width = 7
height = 23

-- hidden units, filter sizes for convolutional network
nstates = {64,64,128}
filtsize = 5
poolsize = 2


-- in the above example, it's abut 75% training and 25% testing...
-- TODO: FIX THESE NUMBERS
trsize = 2709506
tesize =  904455

threads = 1

-- subdirectory to results in
outputDir = 'results-' + os.date("%Y-%m-%-d-%H%M%S")

----------------------------------------------------------------------

----------
-- parse command line arguments
----------

----------------------------------------------------------------------
print '==> loading dataset'

-- Note: the data, in X, is 3-d: the 1st dim indexes the samples
-- and the last two dims index the width and height of the samples.

loaded = torch.load(train_file,'binary')
trainData = {
   data   = loaded.X,

   -- labels are 0/1 because we use cross-entropy loss
   labels = loaded.y,

   weights = loaded.weight,

   size = function() return trsize end
}


-- Finally we load the test data.

loaded = torch.load(test_file,'binary')
testData = {

   data = loaded.X,
   labels = loaded.y,

   weights = loaded.weight,
   size = function() return tesize end
}

----------------------------------------------------------------------
-- print '==> preprocessing data'

----------------------------------------------------------------------
-- model
----------------------------------------------------------------------

-- 2-class problem
noutputs = 1
ninputs = nfeats*width*height


----------------------------------------------------------------------
print '==> construct model'

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

model:add(nn.Dropout(0.5))
-- model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Linear(nstates[2]*1*5, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))

----------------------------------------------------------------------
-- loss function
----------------------------------------------------------------------
print 'defining the loss function'

-- binary cross entropy loss

-- we keep the target output at 0..1

model:add(nn.Sigmoid())
criterion = nn.BCECriterion(train:weights)

print '----------'
print 'loss function:'
print '----------'
print(criterion)
print '----------'


----------------------------------------

-- print model after the output layer has potentially been modified

print '----------'
print 'model after adding the loss function:'
print
print(model)
print '----------'

----------------------------------------------------------------------
-- training
----------------------------------------------------------------------

print("setting number of threads to", threads)
torch.setnumthreads(threads)

----------------------------------------------------------------------

-- Log results to files
trainLogger = optim.Logger(paths.concat(outputDir, 'train.log'))
testLogger  = optim.Logger(paths.concat(outputDir, 'test.log'))

-- get all trainable parameters as a flat vector
parameters, gradParameters = model:getParameters()

print("the model has", #parameters, "parameters") 

----------------------------------------

-- stochastic gradient descent
optimState = {
      -- learning rate at beginning
      learningRate = 1e-3,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 1e-7
   }
optimMethod = optim.sgd

----------
-- function for training steps
----------
-- we do not use a progress bar here as this fills
-- the disk when we redirect the output to a file...

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- to measure the time needed for one training batch
   local startTime = sys.clock()
   print("starting epoch",epoch,"at",os.date("%Y-%m-%d %H:%M:%S",startTime))

   -- set model to training mode (for modules that differ in training and testing, like dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- calculate the inverse mapping
   invshuffle = torch.IntTensor(trsize)
   for i = 1, trsize do
     invshuffle[shuffle[i]] = i
   end

   -- for calculating the AUC
   --
   -- not sure why we have to define them locally,
   -- defining them outside seems to suddenly
   -- reduce the size of e.g. shuffledTargets to half the entries...
   local shuffledTargets = torch.FloatTensor(trsize)
   local shuffledWeights = torch.FloatTensor(trsize)
   local trainOutput     = torch.FloatTensor(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- call garbage collector
      if (t % 300) == 0 then
        collectgarbage()

      end

      -- create a mini batch
      local inputs = {}
      local targets = {}
      local weights = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         local weight = trainDAta.weights[shuffle[i]]

         -- TODO: may take some unnecessary CPU time
         target = torch.FloatTensor({target})

         -- for ROC curve evaluation on training sample
         shuffledTargets[i] = target[1]

         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- note that #inputs is the minibatch size !

                          -- estimate f
                          local output = model:forward(inputs[i])

                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- note that i is the index inside the minibatch
                          -- note that t and i are 1 based, so when
                          -- adding them, one must subtract 1
                          trainOutput[t + i - 1] = output[1]
                       end -- end of loop over minibatch members

                       -- normalize function value and gradient
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)

   end

   -- time taken
   local time = sys.clock() - startTime

   print("\ntime to learn 1 sample = " .. (time / trainData:size() * 1000) .. 'ms')
   print("\ntime for entire batch:",time / 60.0,"min")

   -- we have values 0 and 1 as class labels
   roc_points, roc_thresholds = metrics.roc.points(trainOutput, shuffledTargets, 0, 1)

   local trainAUC = metrics.roc.area(roc_points)
   print("train AUC=", trainAUC)
   trainLogger:add{['AUC (train set)'] = trainAUC}

   -- save/log current net
   local filename = paths.concat(outputDir, 'model' .. string.format("%04d", epoch) .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('saving model to '..filename)
   torch.save(filename, model)

   collectgarbage()

   -- next epoch
   epoch = epoch + 1

end -- function train()

----------------------------------------
-- test function
----------------------------------------

function test()
   -- local vars
   local startTime = sys.clock()
   local testOutput = torch.FloatTensor(tesize)
   local shuffledTargets = torch.FloatTensor(tesize)

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      if (t % 10 == 0) then
         collectgarbage()
      end

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)

      if opt.loss == 'bce' then
          -- we need outputs in the range -1..+1 for AUC
          -- but we use a sigmoid at the output layer
          -- (to be able to use the binary cross entropy loss)
          pred = pred * 2 - 1
          target = target * 2 - 1
      end


      -- confusion:add(pred, target)
      testOutput[t] = pred[1]

      -- TODO: may take some unnecessary CPU time
      target = torch.FloatTensor({target})

      shuffledTargets[t] = target[1]
   end

   -- timing
   local time = sys.clock() - startTime
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   print("\ntime for entire test batch:",time / 60.0,"min")

   roc_points, roc_thresholds = metrics.roc.points(testOutput, shuffledTargets, 0, 1)
   local testAUC = metrics.roc.area(roc_points)
   print("test AUC=",testAUC)
   trainLogger:add{['AUC (test set)'] = testAUC}

end

----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
require 'torch'
require 'os'

----------------------------------------------------------------------

torch.manualSeed(1)

print 'starting training'

while true do
   train()
   test()
end

