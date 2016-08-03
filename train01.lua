#!/usr/bin/env th

require 'torch' 
require 'nn'    
require 'optim'   
require 'os'
require 'math'
require 'io'
require 'xlua' -- for progress bars

require 'cutorch';
require 'nn.THNN';
require 'cunn';

-- common code
require 'myutils'

-- needed for AUC
metrics = require 'metrics';

----------------------------------------------------------------------
-- command line arguments
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Script for training')
cmd:text()
cmd:text('Options')
cmd:option('-model',"",'network model file')
cmd:option('-data', "",'dataset file')
cmd:option('-cuda', false, 'use CUDA tensors')
cmd:option('-opt', 'adam', 'optimizer to use')
cmd:text()

params = cmd:parse(arg)

modelFile = params.model
assert(modelFile ~= "", "must specify a model file with -model")

datasetFile = params.data
assert(datasetFile ~= "", "must specify a dataset file with -data")


----------------------------------------------------------------------
-- parameters
----------------------------------------------------------------------

-- if we don't do this, the weights will be double and
-- the data will be float and we get an error
if params.cuda then
  torch.setdefaulttensortype('torch.CudaTensor')
  print("setting default tensor type to torch.CudaTensor")
else
  torch.setdefaulttensortype('torch.FloatTensor')
  print("setting default tensor type to torch.FloatTensor")
end

----------

-- read data set information
dofile(datasetFile)

threads = 1

-- subdirectory to results in
outputDir = 'results/' .. os.date("%Y-%m-%d-%H%M%S")

print('output directory is ' .. outputDir)

progressBarSteps = 500

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

-- Note: the data, in X, is 3-d: the 1st dim indexes the samples
-- and the last two dims index the width and height of the samples.

if inputDataIsSparse then
  print 'loading sparse dataset'

  trainData, trsize = myutils.loadSparseDataset(train_files, trsize)
  testData,  tesize = myutils.loadSparseDataset(test_files, tesize)

  -- TODO: should print the selected value to the log file in case of typos...
  recHitsXoffset = recHitsXoffset or 0
  recHitsYoffset = recHitsYoffset or 0

else
  -- non-sparse rechit format
  print 'loading dataset'

  trainData, trsize = myutils.loadDataset(train_files, trsize)
  testData,  tesize = myutils.loadDataset(test_files, tesize)

  assert(recHitsXoffset == nil and recHitsYoffset == nil, "rechit center shifting is only supported for sparse data")
end

----------
-- open log file
----------
os.execute('mkdir -p ' .. outputDir)
log,err = io.open(paths.concat(outputDir, 'train.log'), "w")

if log == nil then
  print("could not open log file",err)
end


----------
-- post process loaded data if specified
----------
if postLoadDataset ~= nil then
  postLoadDataset('train', trainData)
  postLoadDataset('test',  testData)
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

assert(makeInput ~= nil, "must define a function 'makeInput' to prepare input to the model")

assert(makeInputView ~= nil, "must define a function 'makeInputView' to make views of input variables for minibatches")

if batchSize == nil then

  batchSize = 32

  print("WARNING: batchSize not specified in model file, setting it to" .. tostring(batchSize))
  fout:write("WARNING: batchSize not specified in model file, setting it to" .. tostring(batchSize) .. "\n")
end

if batchesPerSuperBatch == nil then

  batchesPerSuperBatch = 1

  print("WARNING: batchesPerSuperBatch not specified in model file, setting it to" .. tostring(batchesPerSuperBatch))
  fout:write("WARNING: batchesPerSuperBatch not specified in model file, setting it to" .. tostring(batchesPerSuperBatch) .. "\n")
end

--------------------
-- round progress bar steps to batch size
--------------------

progressBarSteps = math.floor(progressBarSteps / batchSize) * batchSize

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

if params.cuda then
  -- convert the model to cuda
  model:cuda()
  criterion:cuda()
end

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

if params.opt == 'adam' then
  -- ADAM, taking the default values in the optimizer 
  -- (which are those from the paper arxiv:1412.6980)

  print("using ADAM")
  log:write("using ADAM")

  optimState = { }
  optimMethod = optim.adam

elseif params.opt == 'sgd' then
  -- stochastic gradient descent

  print("using stochastic gradient descent")
  log:write("using stochastic gradient descent")

  optimState = {
        -- learning rate at beginning
        learningRate = 1e-3,
        weightDecay = 0,
        momentum = 0,
        learningRateDecay = 1e-7
     }
  optimMethod = optim.sgd
else
  assert(false, "unsupported optimizer '" .. params.opt .. "'")
end

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
   shuffle = torch.IntTensor():randperm(trsize)

   -- in order to avoid problems e.g. with BCECriterion which does not work
   -- when the input and target size differs from the size of the
   -- weight vector (which is given to the BCECriterion constructor)
   -- we only run over minibatches of the same size
   --
   -- we also need to make sure that we do not leave any uninitialized
   -- values, so we shorten the training set to an integer multiple
   -- of the minibatch size

   local effectiveTrainingSize = math.floor(trsize / batchSize) * batchSize

   -- for calculating the AUC
   --
   -- not sure why we have to define them locally,
   -- defining them outside seems to suddenly
   -- reduce the size of e.g. shuffledTargets to half the entries...
   local shuffledTargets = torch.Tensor(effectiveTrainingSize)
   local shuffledWeights = torch.Tensor(effectiveTrainingSize)
   local trainOutput     = torch.Tensor(effectiveTrainingSize)

   -- do one epoch
   print("training epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   log:write("training epoch # " .. tostring(epoch) .. ' [batchSize = ' .. tostring(batchSize) .. ']\n')
   log:flush()

   -- create a mini batch
   -- see also https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
   local targets = torch.zeros(batchSize)
   local weights = torch.zeros(batchSize)

   -- for unpacking more than one minibatch worth's on inputs
   -- and store them on the GPU
   --
   -- note that inputs returned by makeInput can actually be quite complex
   -- so we rely on another custom function which then can just create e.g. a struct/table
   -- with the views of the (larger) unpacked inputs
   local unpackedInputs = nil

   local superBatchStart, superBatchEnd

   for t = 1,effectiveTrainingSize, batchSize do
      -- call garbage collector
      if (t % 300) == 0 then
        collectgarbage()
      end

      -- display progress
      if (t % progressBarSteps) == 1 then
        xlua.progress(t, trainData:size())
      end

      -- calculate effective size of this batch
      local thisEnd = math.min(t + batchSize - 1, trainData:size())
      local thisBatchSize = thisEnd - t + 1

      ----------
      -- create/unpack the inputs on demand
      ----------

      if unpackedInputs == nil or t > superBatchEnd then
        -- unpack another superbatch of inputs
        superBatchStart = t
        superBatchEnd = math.min(t + batchesPerSuperBatch * batchSize - 1, effectiveTrainingSize)

        -- make a list of indices in this superbatch
        local rowIndices = shuffle:sub(superBatchStart, superBatchEnd)

        unpackedInputs = makeInput(trainData, rowIndices, inputDataIsSparse)

      end

      local inputs = makeInputView(unpackedInputs, t - superBatchStart + 1, thisEnd - superBatchStart + 1)

      ----------

      for i = t,thisEnd do

         local iLocal = i - t + 1

         targets[iLocal] = trainData.labels[shuffle[i]]
         weights[iLocal] = trainData.weights[shuffle[i]]

         -- for ROC curve evaluation on training sample
         shuffledTargets[i] = targets[iLocal]
         shuffledWeights[i] = weights[iLocal]

         -- copy weights into the weights variable
         -- for the current batch
         batchWeights[iLocal] = trainData.weights[shuffle[i]]

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
                          -- note that #inputs is the minibatch size !

                          -- estimate f
                          local output = model:forward(inputs)

                          local err = criterion:forward(output, targets)
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets)
                          model:backward(inputs, df_do)

                          -- note that i is the index inside the minibatch
                          -- note that t and i are 1 based, so when
                          -- adding them, one must subtract 1

                       for i = 1,thisBatchSize do
                          trainOutput[t + i - 1] = output[i]
                       end -- end of loop over minibatch members

                       -- normalize function value and gradient
                       gradParameters:div(thisBatchSize)
                       f = f / thisBatchSize

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   local time = sys.clock() - startTime

   print("\n")
   print(string.format("time to learn 1 sample: %.2f ms", time / trainData:size() * 1000))
   print(string.format("time for entire batch: %.2f min", time / 60.0))

   log:write("\n")
   log:write(string.format("time to learn 1 sample: %.2f ms\n", time / trainData:size() * 1000))
   log:write(string.format("time for entire batch: %.2f min\n", time / 60.0))

   -- write out network outputs, labels and weights
   -- to a file so that we can calculate the ROC value with some other tool

   writeROCdata('roc-data-train-' .. string.format("%04d", epoch) .. '.t7',
                shuffledTargets,
                trainOutput,
                shuffledWeights)

   -- we have values 0 and 1 as class labels
   roc_points, roc_thresholds = metrics.roc.points(trainOutput, shuffledTargets, 0, 1, shuffledWeights)

   local trainAUC = metrics.roc.area(roc_points)
   print(string.format("train AUC: %.3f", trainAUC))
   log:write(string.format("train AUC: %.3f\n", trainAUC))

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
   local testOutput = torch.Tensor(tesize)

   -- TODO: change naming: these are not shuffled but
   --       a potential subtensor of the full set
   local shuffledTargets = torch.Tensor(tesize)
   local shuffledWeights = torch.Tensor(tesize)

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

   -- TODO: also use batches for testing
   --       (as much as fits into memory)

   local target = torch.Tensor(batchSize)
   local weight = torch.Tensor(batchSize)

   local rowIndices = torch.IntTensor(batchSize)

   for t = 1,testData:size(), batchSize do

      local thisEnd = math.min(t + batchSize - 1, testData:size())
      local thisBatchSize = thisEnd - t + 1

      if (t % 10 == 0) then
         collectgarbage()
      end

      -- show progress bar
      if (t % progressBarSteps) == 1 then
        xlua.progress(t, testData:size())
      end

      -- get new samples
      rowIndices = rowIndices:range(t, thisEnd) -- can't use torch.range(..) with CudaTensor as default
      target     = testData.labels:sub(t, thisEnd)
      weight     = testData.weights:sub(t, thisEnd)

      local input = makeInput(testData, rowIndices, inputDataIsSparse)

      -- test sample
      local pred = model:forward(input)

      testOutput[{{t, thisEnd}}] = pred
      shuffledTargets[{{t, thisEnd}}] = target
      shuffledWeights[{{t, thisEnd}}] = weight

   end

   -- timing
   local time = sys.clock() - startTime
   print("\n")
   print(string.format("time to test 1 sample: %.2f ms", time / testData:size() * 1000))
   print(string.format("time for entire test batch: %.2f min",time / 60.0))

   log:write('\n')
   log:write(string.format("time to test 1 sample: %.2f ms\n", time / testData:size() * 1000))
   log:write(string.format("time for entire test batch: %.2f min\n", time / 60.0))

   -- write out network outputs, labels and weights
   -- to a file so that we can calculate the ROC value with some other tool
   writeROCdata('roc-data-test-' .. string.format("%04d", epoch) .. '.t7',
                shuffledTargets,                                      
                testOutput,
                shuffledWeights)

   roc_points, roc_thresholds = metrics.roc.points(testOutput, shuffledTargets, 0, 1, shuffledWeights)
   local testAUC = metrics.roc.area(roc_points)
   print(string.format("test AUC: %.3f", testAUC))
   print()

   log:write(string.format("test AUC: %.3f\n", testAUC))
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

