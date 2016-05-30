-- datasets including chose and worst track isolation (2016-05-28)

datasetDir = '../data/2016-05-28-sparse-with-track-iso-vars'


train_files = { datasetDir .. '/GJet20to40_rechits-endcap-train.t7',
                datasetDir .. '/GJet20toInf_rechits-endcap-train.t7',
                datasetDir .. '/GJet40toInf_rechits-endcap-train.t7'
                }

test_files  = { datasetDir .. '/GJet20to40_rechits-endcap-test.t7',
                datasetDir .. '/GJet20toInf_rechits-endcap-test.t7',
                datasetDir .. '/GJet40toInf_rechits-endcap-test.t7'
                }

inputDataIsSparse = true

-- input dimensions
nfeats = 1
width = 7
height = 23

-- for shifting 18,18 to 4,12
recHitsXoffset = -18 + 4
recHitsYoffset = -18 + 12


-- if one specifies nothing (or nil), the full sizes
-- from the input samples are taken
-- 
-- if one specifies values < 1 these are interpreted
-- as fractions of the sample
-- trsize, tesize = 10000, 1000
-- trsize, tesize = 0.1, 0.1

-- training time ~ 1 hour
-- trsize, tesize = 0.125, 0.125

----------------------------------------------------------------------

-- this is called after loading and combining the given
-- input files
function postLoadDataset(label, dataset)
  -- normalize in place

  myutils.normalizeVector(dataset.chgIsoWrtChosenVtx)
  myutils.normalizeVector(dataset.chgIsoWrtWorstVtx)

  -- DEBUG: just set these values to zero -> we should have the same performance as for the previous training
  -- dataset.chgIsoWrtChosenVtx:zero()
  -- dataset.chgIsoWrtWorstVtx:zero()

  -- checking mean and stddev after normalization

  print(label, "chgIsoWrtChosenVtx:", dataset.chgIsoWrtChosenVtx:mean(), dataset.chgIsoWrtChosenVtx:std())
  print(label, "chgIsoWrtWorstVtx:",  dataset.chgIsoWrtWorstVtx:mean(),  dataset.chgIsoWrtWorstVtx:std()) 

end