----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

metrics = require 'metrics';

-- classes
-- classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
-- confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- for calculating the AUC
   --
   -- not sure why we have to define them locally,
   -- defining them outside seems to suddenly
   -- reduce the size of e.g. shuffledTargets to half the entries...
   local shuffledTargets = torch.FloatTensor(trsize)
   local trainOutput = torch.FloatTensor(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- call garbage collector
      if (t % 300) == 0 then
        collectgarbage()

      end

      -- disp progress
      if (t % 100) == 0 then
        xlua.progress(t, trainData:size())
      end

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]

         -- this is for the calculation of the AUC
         -- note that the metrics package requires labels
         -- to be -1 or +1
         -- print("i=",i, "target=",target[1],'shuffledTargets[i]=',shuffledTargets[i])

         -- conversion from 0 / 1 to -1 / +1 already happened elsewhere

         if opt.loss == 'bce' then
           -- TODO: may take some unnecessary CPU time
           target = torch.FloatTensor({target})
         end

         shuffledTargets[i] = target[1]

         if opt.type == 'double' then 
           input = input:double()
           target = target:double()
         elseif opt.type == 'cuda' then 
           input = input:cuda() 
           target = target:cuda()
         end

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
                          -- estimate f
                          local output = model:forward(inputs[i])
                          -- print("ZZ targets=",targets)
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          -- confusion:add(output, targets[i])
                          trainOutput[i] = output[1]
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --   print(confusion)

   -- print AUC
   -- print("trainOutput=",trainOutput[1], trainOutput[2], trainOutput[3], trainOutput[4])
   -- 
   -- for k = 1,50 do
   --   if (shuffledTargets[k] ~= 1 and shuffledTargets[k] ~= -1) then
   --     print("shuffledTargets[",k,"]=", shuffledTargets[k])
   --   end
   -- end

   roc_points, roc_thresholds = metrics.roc.points(trainOutput, shuffledTargets)

   if opt.loss == 'bce' then
     -- need to convert from 0..1 to -1..+1 for ROC curve calculation
     -- note that the order in which the tensor operations are done is important
     -- see https://github.com/torch/torch7/issues/28#issuecomment-39877253
     roc_points, roc_thresholds = metrics.roc.points(trainOutput *2  - 1, shuffledTargets * 2 - 1)
   else
     roc_points, roc_thresholds = metrics.roc.points(trainOutput, shuffledTargets)
   end

   print("train AUC=",metrics.roc.area(roc_points))

   -- update logger/plot
   -- trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   -- if opt.plot then
   --    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   --    trainLogger:plot()
   -- end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   collectgarbage()

   -- next epoch
   -- confusion:zero()
   epoch = epoch + 1
end
