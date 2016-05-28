-- datasets for trainings starting from 2016-05-09

-- train_files = { '../data/2016-05-sparse-test/GJet20to40/rechits-barrel-train.t7',
--                 '../data/2016-05-sparse-test/GJet20toInf/rechits-barrel-train.t7',
--                 '../data/2016-05-sparse-test/GJet40toInf/rechits-barrel-train.t7'
--                 }
-- 
-- test_files  = { '../data/2016-05-sparse-test/GJet20to40/rechits-barrel-test.t7',
--                 '../data/2016-05-sparse-test/GJet20toInf/rechits-barrel-test.t7',
--                 '../data/2016-05-sparse-test/GJet40toInf/rechits-barrel-test.t7'
--                 }
-- 
-- -- input dimensions
-- nfeats = 1
-- width = 7
-- height = 23

train_files = { '../data/2016-05-larger-window/GJet20to40/rechits-barrel-train.t7',
                '../data/2016-05-larger-window/GJet20toInf/rechits-barrel-train.t7',
                '../data/2016-05-larger-window/GJet40toInf/rechits-barrel-train.t7'
                }

test_files  = { '../data/2016-05-larger-window/GJet20to40/rechits-barrel-test.t7',
                '../data/2016-05-larger-window/GJet20toInf/rechits-barrel-test.t7',
                '../data/2016-05-larger-window/GJet40toInf/rechits-barrel-test.t7'
                }

-- input dimensions
nfeats = 1
width = 7
height = 23

-- for shifting 18,18 to 4,12
recHitsXoffset = -18 + 4
recHitsYoffset = -18 + 12

inputDataIsSparse = true


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
trsize, tesize = 0.5, 0.5

-- trsize, tesize = 100, 100

-- DEBUG
-- trsize, tesize = 0.01, 0.01