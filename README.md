This is the code released for the paper
# Perssonalized Session-Based Recommendation Using Graph Attention Networks
accepted by _The annual International Joint Conference on Neural Networks_ IJCNN2021.
The proposed PSR-GAT model is an attentional Graph Nerual Network on the GAT basis. Their architecuture are similar, with several improvements being introduced. Interested readers may find the description of differences in Section IV-E of the paper.

Our paper get some inspirations from GAT and FGNN, many thanks to the authors.

## Datasets
- _SYNC Screen_ is not released to public because we have confidential protocal with the data contributors.
- _AppUsage_ is avaiable at [AppUsage](http://www.recg.org/downloads.html)
- _Cosmetic Shop_ is available at [Cosmetic Shop](https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop)

## Requirements
- Python3
- PyTorch>=1.4 (earlier versions are not tested)
- PyTorch-Geometric>=1.4.2 (earlier versions are not tested)

## Benchmarks
- readers can go to [FGNN](https://github.com/RuihongQiu/FGNN) for the implementation of FGNN.
- Readers can go to [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN) for the implementation of SR-GNN.
- Readers can go to [Caser](https://github.com/graytowne/caser_pytorch) for the implementation of Caser.
- Readers can go to the [Pytorch-Gemetric document](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) for other implementing information

## About the codes
- data_processing.py: build graphs from sessions
- custermized_gcn_layers.py: the custermized layer with our designed messange propagation rules, used by the PSR-GAT model as the attentional layers. The two Graph Convolutional classes provided are exactly the same, compitable with different Pytorch-Geometric versions.
- model_build.py: build the the PSR-GAT model architecture.
- util.py:  helpful variables and functions
- GNN_run: scripts for training and evaluating, the top file for running the experiments

If you find it useful, please cite our paper "Personalized Session-Based Recommendation Using Graph Attention Networks".
