CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node 8 /data/caojunhao/ELN/train.py \
--exp-name=city_2_res50 \
--train-split=2 \
--dataset=city \
--pre_epoch=1 \
--eln_epoch=1 \
--backbone_name=50 \
--batch-size-labeled=4 \
--batch-size-unlabeled=4