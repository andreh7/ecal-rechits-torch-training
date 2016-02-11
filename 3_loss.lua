----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
--   + mean-square error
--   + margin loss (SVM-like)
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin | bce')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

-- 2-class problem
noutputs = 1

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

   -- This loss takes a vector of classes, and the index of
   -- the grountruth class as arguments. It is an SVM-like loss
   -- with a default margin of 1.

   criterion = nn.MultiMarginCriterion()

elseif opt.loss == 'nll' then

   -- This loss requires the outputs of the trainable model to
   -- be properly normalized log-probabilities, which can be
   -- achieved using a softmax function

   model:add(nn.LogSoftMax())

   -- The loss works like the MultiMarginCriterion: it takes
   -- a vector of classes, and the index of the grountruth class
   -- as arguments.

   criterion = nn.ClassNLLCriterion()

   trlabels = trainData.labels[i]
   telabels = testData.labels[i]

elseif opt.loss == 'bce' then
   -- we keep the target output at 0..1

   model:add(nn.Sigmoid())
   criterion = nn.BCECriterion()

elseif opt.loss == 'mse' then

   -- for MSE, we add a tanh, to restrict the model's output
   model:add(nn.Tanh())

   -- The mean-square error is not recommended for classification
   -- tasks, as it typically tries to do too much, by exactly modeling
   -- the 1-of-N distribution. For the sake of showing more examples,
   -- we still provide it here:

   criterion = nn.MSECriterion()
   criterion.sizeAverage = false

   -- Compared to the other losses, the MSE criterion needs a distribution
   -- as a target, instead of an index. Indeed, it is a regression loss!
   -- So we need to transform the entire label vectors:

   if trainData then
      -- convert training labels:
      local trsize = (#trainData.labels)[1]
      local trlabels = torch.Tensor( trsize, noutputs )
      print("HERE1")

      trlabels:fill(-1)

      print("HERE2")
      -- for i = 1,trsize do
      --    trlabels[{ i,trainData.labels[i] }] = 1
      -- end
      
      for i = 1,trsize do
         trlabels[{ i,1 }] = 2 * trainData.labels[i] - 1

         if (i % 100 == 0) then
            collectgarbage()
         end
      end

      print("HERE3")

      trainData.labels = trlabels

      -- convert test labels
      local tesize = (#testData.labels)[1]
      local telabels = torch.Tensor( tesize, noutputs )
      telabels:fill(-1)
      -- for i = 1,tesize do
      --    telabels[{ i,testData.labels[i] }] = 1
      -- end
      for i = 1,tesize do
         telabels[{ i,1 }] = 2 * testData.labels[i] - 1

         if (i % 100 == 0) then
            collectgarbage()
         end

      end

      testData.labels = telabels
   end

else

   error('unknown -loss')

end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)

----------------------------------------------------------------------

-- print model after the output layer has potentially been modified
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
