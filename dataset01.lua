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
