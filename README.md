# Experiments with CoCosNet

This repo contains the code that I used for conducting my experiments with CoCosNet on the ADE20K Dataset.

## Installation
Clone the Synchronized-BatchNorm-PyTorch repository.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Inference Using Pretrained Model
 
Download the pretrained model from [here](https://drive.google.com/drive/folders/1BEBBENbEr9tutZsyGGc3REUuuOYqf6M3?usp=sharing) and save them in `checkpoints/ade20k`. Then run the command 
````bash
sh test_demo.sh
````
The results are saved in `output/test/ade20k`.

If you don't want to use mask of exemplar image when testing, you can download model from [here](https://drive.google.com/drive/folders/1m4LXbOc00cu8hXCgf-_N55AIAE9R__m6?usp=sharing), save them in `checkpoints/ade20k`, and run
```` bash
sh test_demo_no_mask.sh
````

## Training

**Pretrained VGG model** Download from [here](https://drive.google.com/file/d/1fp7DAiXdf0Ay-jANb8f0RHYLTRyjNv4m/view?usp=sharing), move it to `models/`. This model is used to calculate training loss.

#### ADE20k (mask-to-image)  
- **Dataset** Download [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/), move `ADEChallengeData2016/annotations/ADE_train_*.png` to `ADEChallengeData2016/images/training/`, `ADEChallengeData2016/annotations/ADE_val_*.png` to `ADEChallengeData2016/images/validation/

- **Retrieval_pairs** We use image retrieval to find exemplars for exemplar-based training. Download `ade20k_ref.txt` and `ade20k_ref_test.txt` from [here](https://drive.google.com/drive/folders/1BKrEtEE2u5eZgAkviBo0TJJNDM4F4wga?usp=sharing), save or replace them in `data/`

- Run the command, note `dataset_path` is your ade20k root, e.g., `/data/Dataset/ADEChallengeData2016/images`. We use 8 32GB Tesla V100 GPUs for training. You can set `batchSize` to 16, 8 or 4 with fewer GPUs and change `gpu_ids`.
    ````bash
    python train.py --name ade20k --dataset_mode ade20k --dataroot dataset_path --niter 100 --niter_decay 100 --use_attention --maskmix --warp_mask_losstype direct --weight_mask 100.0 --PONO --PONO_C --batchSize 32 --vgg_normal_correct --gpu_ids 0,1,2,3,4,5,6,7
    ````

## Acknowledgments
This code borrows heavily from the [official CoCosNet repo](https://github.com/microsoft/CoCosNet).
