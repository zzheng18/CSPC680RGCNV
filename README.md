# CSPC680RGCNV

Run our RGCNV model: `python src/train.py --dataset pubmed` 
(It first outputs the result of RGCN, after the model runs for the second time, it will output our results.) \
Run our GPC + GCN: `python src/train_gpc.py --dataset pubmed` \
Run GCN: `python src/train_gcn.py --dataset pubmed` \
To switch to other datasets, replace 'pubmed' with 'citeseer' or 'cora' \
