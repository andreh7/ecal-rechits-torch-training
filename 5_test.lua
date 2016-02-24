----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

metrics = require 'metrics';

----------------------------------------------------------------------
print '==> defining test procedure'


-- test function
function test()
   -- local vars
   local time = sys.clock()
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

      if opt.loss == 'bce' then
        -- TODO: may take some unnecessary CPU time
        target = torch.FloatTensor({target})
      end

      -- the metrics package needs labels to be +1 and -1
      -- conversion from 0 / 1 to -1 / +1 already happened elsewhere
      shuffledTargets[t] = target[1]
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   -- print(confusion)

   -- print AUC

   -- collectgarbage()
   -- print("HERE1")
   roc_points, roc_thresholds = metrics.roc.points(testOutput, shuffledTargets)
   -- print("HERE2")
   print("test AUC=",metrics.roc.area(roc_points))
   -- print("HERE3")   


   -- update log/plot
   -- testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   -- if opt.plot then
   --    testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --    testLogger:plot()
   -- end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   -- confusion:zero()
end
