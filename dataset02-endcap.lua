-- datasets for trainings starting from 2016-05-09

train_files = { '../data/2016-05/GJet20to40/rechits-endcap-train.t7',
                '../data/2016-05/GJet20toInf/rechits-endcap-train.t7',
                '../data/2016-05/GJet40toInf/rechits-endcap-train.t7'
                }

test_files  = { '../data/2016-05/GJet20to40/rechits-endcap-test.t7',
                '../data/2016-05/GJet20toInf/rechits-endcap-test.t7',
                '../data/2016-05/GJet40toInf/rechits-endcap-test.t7'
                }


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
