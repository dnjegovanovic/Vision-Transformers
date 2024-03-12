This is a PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]https://arxiv.org/abs/2010.11929). The goal of this project is to provide a simple implementation of the paper.

## Theory Explanation
The architecture of this model is inspired by the BERT language architecture. 
The basic idea behind this approach is that the input image can be viewed as a series of patches that then represent "tokens" that can be further processed through the Attention architecture, which was created for the needs of NLP problems.

![alt ViT Architecture](theory_imgs/arch.jpg)

In the following, through several steps, we will explain to the general pipline what we want to do on a theoretical level.

### Create Embeddings

The first step is that we want to divide the input image into a network of non-overlapping patches, and then project each patch linearly in order to obtain an embedded vector of a certain size.

![alt ViT Architecture](theory_imgs/embedding.PNG)

## Usage

## Results