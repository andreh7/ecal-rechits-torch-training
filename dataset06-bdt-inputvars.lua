-- datasets including chose and worst track isolation (2016-05-28)

datasetDir = '../data/2016-07-06-bdt-inputs'

train_files = { datasetDir .. '/GJet20to40_rechits-barrel-train.t7',
                datasetDir .. '/GJet20toInf_rechits-barrel-train.t7',
                datasetDir .. '/GJet40toInf_rechits-barrel-train.t7'
                }

test_files  = { datasetDir .. '/GJet20to40_rechits-barrel-test.t7',
                datasetDir .. '/GJet20toInf_rechits-barrel-test.t7',
                datasetDir .. '/GJet40toInf_rechits-barrel-test.t7'
                }

inputDataIsSparse = false


-- if one specifies nothing (or nil), the full sizes
-- from the input samples are taken
-- 
-- if one specifies values < 1 these are interpreted
-- as fractions of the sample
-- trsize, tesize = 10000, 1000
-- trsize, tesize = 0.1, 0.1
-- trsize, tesize = 0.01, 0.01

-- limiting the size for the moment because
-- with the full set we ran out of memory after training
-- on the first epoch
-- trsize, tesize = 0.5, 0.5

trsize, tesize = nil, nil


-- DEBUG
-- trsize, tesize = 0.01, 0.01
-- trsize, tesize = 100, 100

----------------------------------------


-- this is called after loading and combining the given
-- input files
function postLoadDataset(label, dataset)

end

----------------------------------------
-- input variables
----------------------------------------

--   phoIdInput :
--     {
--       s4 : FloatTensor - size: 431989
--       scRawE : FloatTensor - size: 431989
--       scEta : FloatTensor - size: 431989
--       covIEtaIEta : FloatTensor - size: 431989
--       rho : FloatTensor - size: 431989
--       pfPhoIso03 : FloatTensor - size: 431989
--       phiWidth : FloatTensor - size: 431989
--       covIEtaIPhi : FloatTensor - size: 431989
--       etaWidth : FloatTensor - size: 431989
--       esEffSigmaRR : FloatTensor - size: 431989
--       r9 : FloatTensor - size: 431989
--       pfChgIso03 : FloatTensor - size: 431989
--       pfChgIso03worst : FloatTensor - size: 431989
--     }

----------------------------------------------------------------------

function datasetLoadFunction(fnames, size, cuda)
  local data = nil

  local totsize = 0

  -- sort the names of the input variables
  -- so that we get reproducible results
  local sortedVarnames = {}
  local numVars

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
         data   = {},
      
         -- labels are 0/1 because we use cross-entropy loss
         labels = loaded.y:sub(1, thisSize),
      
         weights = loaded.weight:sub(1, thisSize),
    
         mvaid = loaded.mvaid:sub(1, thisSize),
      }

      -- fill the individual variable names
      local varname
      for varname in pairs(loaded.phoIdInput) do
        table.insert(sortedVarnames, varname)
      end
      table.sort(sortedVarnames)
      numvars = #sortedVarnames

      -- allocate a 2D Tensor
      data.data = torch.Tensor(thisSize, numvars)

      -- copy over the individual variables: use a 2D tensor
      -- with each column representing a variables
      
      local varindex
      for varindex, varname in pairs(sortedVarnames) do
        data.data[{{}, {varindex}}] = loaded.phoIdInput[varname]:sub(1,thisSize)
      end -- loop over input variables
      
    else
      -- append
      
      data.labels  = data.labels:cat(loaded.y:sub(1, thisSize), 1)

      data.weights = data.weights:cat(loaded.weight:sub(1, thisSize), 1)

      data.mvaid   = data.mvaid:cat(loaded.mvaid:sub(1, thisSize), 1)

      -- special treatment for input variables

      -- note that we can not use resize(..) here as the contents
      -- of the resized tensor are undefined according to 
      -- https://github.com/torch/torch7/blob/master/doc/tensor.md#resizing
      --
      -- so we build first a tensor with the new values
      -- and then concatenate this to the previously loaded data
      local newData = torch.Tensor(thisSize, numvars)

      local varindex
      local varname
      for varindex, varname in pairs(sortedVarnames) do
        newData[{{},{varindex}}] = loaded.phoIdInput[varname]:sub(1,thisSize)
      end -- loop over input variables

      -- and append
      data.data    = data.data:cat(newData, 1)
      
    end -- appending

  end -- loop over files

  ----------
  -- convert to CUDA tensors if required
  ----------
  if cuda then
    data.labels  = data.labels:cuda()
    data.weights = data.weights:cuda()
    data.mvaid   = data.mvaid:cuda()
  end

  ----------


  data.size = function() return totsize end
  
  assert (totsize == data.data:size()[1])

  -- normalize weights to have an average
  -- of one per sample
  -- (weights should in principle directly
  -- affect the effective learning rate of SGD)
  data.weights:mul(data.weights:size()[1] / data.weights:sum())

  return data, totsize

end