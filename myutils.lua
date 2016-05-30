#!/usr/bin/env th

-- Lua 5.1 way of declaring modules
module("myutils", package.seeall)

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

function loadSparseDataset(fnames, size)

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

    -- determine last rechit index
    if thisSize < loaded.X.firstIndex:size()[1] then
      lastRecHitIndex = loaded.X.firstIndex[thisSize + 1] - 1
    else
      -- we take the entire dataset
      lastRecHitIndex = loaded.X.energy:size()[1]
    end

    if data == nil then
      -- create the first entry

      data = {
         data = {}
      }

      -- copy other items (and map some names)

      for name, values in pairs(loaded) do
         if name ~= 'X' then

           outputName = name
  
           if name == 'y' then
             outputName = 'labels'
           elseif name == 'weight' then
             outputName = 'weights'
           end
           
           -- labels are 0/1 because we use cross-entropy loss
           data[outputName] = loaded[name]:sub(1, thisSize)
        end

      end -- loop over input items

      -- copy rechits

      -- copy the indices and lengths
      data.data.firstIndex = loaded.X.firstIndex:sub(1, thisSize)
      data.data.numRecHits = loaded.X.numRecHits:sub(1, thisSize)
      
      -- copy the rechits
      data.data.x = loaded.X.x:sub(1,lastRecHitIndex)
      data.data.y = loaded.X.y:sub(1,lastRecHitIndex)
      data.data.energy = loaded.X.energy:sub(1,lastRecHitIndex)

    else
      -- append
      for name, values in pairs(loaded) do

        if name ~= 'X' then
           outputName = name
  
           if name == 'y' then
             outputName = 'labels'
           elseif name == 'weight' then
             outputName = 'weights'
           end
           
           -- labels are 0/1 because we use cross-entropy loss
           data[outputName] = data[outputName]:cat(loaded[name]:sub(1, thisSize), 1)
        end

      end -- loop over input items

      numPhotonsBefore = data.data.firstIndex:size()[1]
      numRecHitsBefore = data.data.energy:size()[1]

      -- append sparse rechits

      -- copy the rechits
      data.data.x = data.data.x:cat(loaded.X.x:sub(1,lastRecHitIndex), 1)
      data.data.y = data.data.y:cat(loaded.X.y:sub(1,lastRecHitIndex), 1)
      data.data.energy = data.data.energy:cat(loaded.X.energy:sub(1,lastRecHitIndex), 1)
      
      -- copy the indices and lengths
      data.data.firstIndex = data.data.firstIndex:cat(loaded.X.firstIndex:sub(1, thisSize), 1)
      data.data.numRecHits = data.data.numRecHits:cat(loaded.X.numRecHits:sub(1, thisSize), 1)

      -- for the indices we have to shift them
      for i=1,thisSize do
        data.data.firstIndex[numPhotonsBefore + i] = data.data.firstIndex[numPhotonsBefore + i] + numRecHitsBefore
      end      
      
    end


  end -- loop over files


  data.size = function() return totsize end
  
  assert (totsize == data.data.firstIndex:size()[1])

  -- normalize weights to have an average
  -- of one per sample
  -- (weights should in principle directly
  -- affect the effective learning rate of SGD)
  data.weights:mul(data.weights:size()[1] / data.weights:sum())

  -- DEBUG: fix weights to one
  --  data.weights = torch.Tensor(data.weights:size()[1]):fill(1):float()
 
  return data, totsize

end -- function loadSparseDataset

----------------------------------------------------------------------

function normalizeVector(vec)
  -- normalizes (shift to zero and scale to unit standard deviation)
  -- 1D tensors in place

  -- subtract mean
  vec:add( - vec:mean())

  -- normalize to unit standard deviation
  vec:div(vec:std())

end

----------------------------------------------------------------------