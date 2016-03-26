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

progressBarSteps = 500

-- if we don't do this, the weights will be double and
-- the data will be float and we get an error
torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------


----------------------------------------------------------------------
-- command line arguments

-- the trained network file
networkFile = arg[1]



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

----------
-- parse command line arguments
----------

----------------------------------------------------------------------
print 'loading dataset'

-- Note: the data, in X, is 3-d: the 1st dim indexes the samples
-- and the last two dims index the width and height of the samples.

trainData, trsize = loadDataset(train_files, trsize)
testData,  tesize = loadDataset(test_files, tesize)

-- load the trained network to be analyzed
network = torch.load(networkFile)

-- assume it's a nn.Sequential for the moment
-- variable names:
-- mX_A_B_C_...
-- where A B C are the tensor indices 

varnames = {}

for moduleIndex = 1,#network.modules do
  module = network.modules[moduleIndex]

  local idx = {}
  local ndim = module.output:dim()
  local dims = module.output:size()

  idx[ndim] = 0
  for i=1, (ndim - 1) do
    idx[i] = 1
  end

  -- loop over all indices
  local lastIndexFound = false 

  while (not lastIndexFound) do

    -- increase index by one
    for i=ndim, 1, -1 do
      last_i = i

      idx[i] = idx[i] + 1
      if idx[i] <= dims[i] then
        break
      end
      idx[i] = 1
  
      if (i == 1) then
        lastIndexFound = true
        break
      end
    end

    if lastIndexFound then
      break
    end

    -- build a name for this output value
    local varname = "m" .. moduleIndex

    for i = 1,ndim do
      varname = varname .. "_i" .. idx[i]
    end

    print(varname)

  end -- loop over all indices

end


