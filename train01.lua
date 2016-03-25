#!/usr/bin/env th

require 'torch' 
require 'nn'    
require 'optim'   
require 'os'
require 'math'
require 'io'
require 'xlua' -- for progress bars

-- needed for AUC
metrics = require 'metrics';

----------------------------------------------------------------------
-- parameters
----------------------------------------------------------------------

-- TODO should use HT range 40-100 first
-- train_files = { '../torch-utils/gjet-ht-400-600-train.t7' }; test_files  = { '../torch-utils/gjet-ht-400-600-test.t7' }

-- two highest weighted processes for m(gamma,gamma) = 80 .. 180 GeV
train_files = { '../torch-utils/gjet-ht-40-100-train.t7', '../torch-utils/gjet-ht-100-200-train.t7' }
test_files  = { '../torch-utils/gjet-ht-40-100-test.t7',  '../torch-utils/gjet-ht-100-200-test.t7' }


-- input dimensions
nfeats = 1
width = 7
height = 23

-- if one specifies nothing (or nil), the full sizes
-- from the input samples are taken
-- 
-- if one specifies values < 1 these are interpreted
-- as fractions of the sample
-- trsize, tesize = 10000, 1000
-- trsize, tesize = 0.1, 0.1



threads = 1

-- subdirectory to results in
outputDir = 'results-' .. os.date("%Y-%m-%-d-%H%M%S")

print('output directory is ' .. outputDir)

batchSize = 1

progressBarSteps = 500

-- if we don't do this, the weights will be double and
-- the data will be float and we get an error
torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------

-- round to batch size
progressBarSteps = math.floor(progressBarSteps / batchSize) * batchSize


----------------------------------------------------------------------
-- command line arguments

modelFile = arg[1]

----------------------------------------------------------------------

function loadDataset(fnames, size)

  local data = nil

  local totsize = 0

  -- load all input files
  for i = 1, #fnames do

    local loaded = torch.load(fnames[i],'binary')

    local thisSize

    -- determine the size
    if size ~= nil and size < 1 then
      thisSize = math.floor(size * loaded.y:size()[1] + 0.5)
    else
      thisSize = size or loaded.y:size()[1]
      thisSize = math.min(thisSize, loaded.y:size()[1])
    end

    totsize = totsize + thisSize

    if data == nil then
      -- create the first entry

      data = {
         data   = loaded.X:sub(1, thisSize),
      
         -- labels are 0/1 because we use cross-entropy loss
         labels = loaded.y:sub(1, thisSize),
      
         weights = loaded.weight:sub(1, thisSize),
    
         mvaid = loaded.mvaid:sub(1, thisSize),
      }

    else
      -- append
      data.data    = data.data:cat(loaded.X:sub(1, thisSize), 1)
      
      data.labels  = data.labels:cat(loaded.y:sub(1, thisSize), 1)

      data.weights = data.weights:cat(loaded.weight:sub(1, thisSize), 1)

      data.mvaid   = data.mvaid:cat(loaded.mvaid:sub(1, thisSize), 1)
      
    end


  end -- loop over files


  data.size = function() return totsize end
  
  assert (totsize == data.data:size()[1])

  -- normalize weights to have an average
  -- of one per sample
  -- (weights should in principle directly
  -- affect the effective learning rate of SGD)
  data.weights:mul(data.weights:size()[1] / data.weights:sum())

  -- DEBUG: fix weights to one
  --  data.weights = torch.Tensor(data.weights:size()[1]):fill(1):float()
 
  return data, totsize

end -- function loadDataset

----------------------------------------------------------------------

-- writes out a table containing information to calculate
-- a ROC curve
function writeROCdata(relFname, targetValues, outputValues, weights)

   local dataForRoc = {
     label = targetValues,
     output = outputValues,
     weight = weights
   }

   torch.save(paths.concat(outputDir, relFname), dataForRoc)

end -- function

----------------------------------------------------------------------

----------
-- parse command line arguments
----------

----------------------------------------------------------------------
print 'loading dataset'

-- Note: the data, in X, is 3-d: the 1st dim indexes the samples
-- and the last two dims index the width and height of the samples.

trainData, trsize = loadDataset(train_files, trsize)
testData,  tesize = loadDataset(test_files, tesize)

----------
-- open log file
----------
os.execute('mkdir -p ' .. outputDir)
log,err = io.open(paths.concat(outputDir, 'train.log'), "w")

if log == nil then
  print("could not open log file",err)
end

----------
-- write ROC data for mvaid
----------
writeROCdata('roc-data-train-mva.t7',
             trainData.labels,
             trainData.mvaid,
             trainData.weights)

writeROCdata('roc-data-test-mva.t7',
             testData.labels,
             testData.mvaid,
             testData.weights)

----------
-- write training samples 
----------

fout = io.open(paths.concat(outputDir, 'samples.txt'), "w")
for i = 1, #train_files do
  fout:write(train_files[i] .. "\n")
end
fout:close()

dofile(modelFile)

----------------------------------------------------------------------
-- loss function
----------------------------------------------------------------------
-- binary cross entropy loss

-- we keep the target output at 0..1

model:add(nn.Sigmoid())

batchWeights = torch.Tensor(batchSize)
criterion = nn.BCECriterion(batchWeights)

print('loss function:', criterion)
log:write('loss function: ' .. tostring(criterion) .. "\n")

----------------------------------------

-- print model after the output layer has potentially been modified

print('----------')
print('model after adding the loss function:')
print()
print(model)
print('----------')

log:write('----------\n')
log:write('model after adding the loss function:\n')
log:write("\n")
log:write(tostring(model) .. "\n")
log:write('----------\n')


----------------------------------------------------------------------
-- training
----------------------------------------------------------------------

print("setting number of threads to" .. threads)
log:write("setting number of threads to " .. threads .. "\n")
torch.setnumthreads(threads)

----------------------------------------------------------------------


-- get all trainable parameters as a flat vector
parameters, gradParameters = model:getParameters()

print("the model has " ..  parameters:size()[1] .. " parameters") 
log:write("the model has " .. parameters:size()[1] .. " parameters\n") 

print("using " .. trsize .. " train samples and " .. tesize .. " test samples\n")

log:write("\n")
log:write("using " .. trsize .. " train samples and " .. tesize .. " test samples\n\n")

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
   epoch = epoch or 0

   -- increase the epoch here (rather than at the end of training)
   -- such that we have the same epoch number for training and testing
   epoch = epoch + 1

   -- to measure the time needed for one training batch
   local startTime = sys.clock()
   print("----------------------------------------")
   print("starting epoch " .. epoch .. " at ",os.date("%Y-%m-%d %H:%M:%S",startTime))
   print("----------------------------------------")

   log:write("----------------------------------------\n")
   log:write("starting epoch " .. epoch .. " at " .. os.date("%Y-%m-%d %H:%M:%S",startTime) .. "\n")
   log:write("----------------------------------------\n")
   log:flush()

   -- set model to training mode (for modules that differ in training and testing, like dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- for calculating the AUC
   --
   -- not sure why we have to define them locally,
   -- defining them outside seems to suddenly
   -- reduce the size of e.g. shuffledTargets to half the entries...
   local shuffledTargets = torch.FloatTensor(trsize)
   local shuffledWeights = torch.FloatTensor(trsize)
   local trainOutput     = torch.FloatTensor(trsize)

   -- do one epoch
   print("training epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   log:write("training epoch # " .. tostring(epoch) .. ' [batchSize = ' .. tostring(batchSize) .. ']\n')
   log:flush()

   for t = 1,trainData:size(), batchSize do
      -- call garbage collector
      if (t % 300) == 0 then
        collectgarbage()
      end

      -- display progress
      if (t % progressBarSteps) == 1 then
        xlua.progress(t, trainData:size())
      end

      -- create a mini batch
      local inputs = {}
      local targets = {}
      local weights = {}
      for i = t,math.min(t + batchSize - 1, trainData:size()) do
         -- load new sample
         local input  = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         local weight = trainData.weights[shuffle[i]]

         -- TODO: may take some unnecessary CPU time
         target = torch.FloatTensor({target})

         -- for ROC curve evaluation on training sample
         shuffledTargets[i] = target
         shuffledWeights[i] = weight

         table.insert(inputs, input)
         table.insert(targets, target)

         -- copy weights into the weights variable
         -- for the current batch
         batchWeights[i - t + 1] = trainData.weights[i]

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

   print("\n")
   print("time to learn 1 sample: " .. (time / trainData:size() * 1000) .. ' ms')
   print("time for entire batch:",time / 60.0,"min")

   log:write("\n")
   log:write("time to learn 1 sample: " .. tostring(time / trainData:size() * 1000) .. ' ms\n')
   log:write("time for entire batch: " .. tostring(time / 60.0) .. " min\n")

   -- write out network outputs, labels and weights
   -- to a file so that we can calculate the ROC value with some other tool

   writeROCdata('roc-data-train-' .. string.format("%04d", epoch) .. '.t7',
                shuffledTargets,
                trainOutput,
                shuffledWeights)

   -- we have values 0 and 1 as class labels
   roc_points, roc_thresholds = metrics.roc.points(trainOutput, shuffledTargets, 0, 1, shuffledWeights)

   local trainAUC = metrics.roc.area(roc_points)
   print("train AUC:", trainAUC)
   log:write("train AUC: " .. tostring(trainAUC) .. "\n")

   -- save/log current net
   local filename = paths.concat(outputDir, 'model' .. string.format("%04d", epoch) .. '.net')
   print('saving model to '.. filename)
   log:write('saving model to '.. filename .. "\n")
   torch.save(filename, model)

   collectgarbage()

end -- function train()

----------------------------------------
-- test function
----------------------------------------

function test()
   -- local vars
   local startTime = sys.clock()
   local testOutput = torch.FloatTensor(tesize)

   -- TODO: change naming: these are not shuffled but
   --       a potential subtensor of the full set
   local shuffledTargets = torch.FloatTensor(tesize)
   local shuffledWeights = torch.FloatTensor(tesize)

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('testing on test set')
   log:write('testing on test set\n')
   for t = 1,testData:size() do

      if (t % 10 == 0) then
         collectgarbage()
      end

      -- show progress bar
      if (t % progressBarSteps) == 1 then
        xlua.progress(t, testData:size())
      end

      -- get new sample
      local input = testData.data[t]
      local target = testData.labels[t]
      local weight = trainData.weights[t]

      -- test sample
      local pred = model:forward(input)

      -- confusion:add(pred, target)
      testOutput[t] = pred[1]

      -- TODO: may take some unnecessary CPU time
      target = torch.FloatTensor({target})

      shuffledTargets[t] = target[1]
      shuffledWeights[t] = weight
   end

   -- timing
   local time = sys.clock() - startTime
   time = time / testData:size()
   print("\n")
   print("time to test 1 sample: " .. (time*1000) .. ' ms')
   print("time for entire test batch:",time / 60.0,"min")

   log:write('\n')
   log:write("time to test 1 sample: " .. tostring(time*1000) .. ' ms\n')
   log:write("time for entire test batch: " .. tostring(time / 60.0) .. " min\n")

   -- write out network outputs, labels and weights
   -- to a file so that we can calculate the ROC value with some other tool
   writeROCdata('roc-data-test-' .. string.format("%04d", epoch) .. '.t7',
                shuffledTargets,                                      
                testOutput,
                shuffledWeights)

   roc_points, roc_thresholds = metrics.roc.points(testOutput, shuffledTargets, 0, 1, shuffledWeights)
   local testAUC = metrics.roc.area(roc_points)
   print("test AUC:", testAUC)
   print()

   log:write("test AUC: " .. tostring(testAUC) .. "\n")
   log:write("\n")
   log:flush()

end

----------------------------------------------------------------------
-- main
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

