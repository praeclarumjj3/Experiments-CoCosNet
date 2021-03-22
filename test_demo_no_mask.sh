#!/bin/sh

python test.py --name ade20k \
--dataset_mode ade20k \
--dataroot ./imgs/ade20k \
--gpu_ids 0 \
--nThreads 0 \
--batchSize 6 \
--use_attention \
--maskmix \
--noise_for_mask \
--warp_mask_losstype direct \
--PONO --PONO_C \
--which_epoch 90