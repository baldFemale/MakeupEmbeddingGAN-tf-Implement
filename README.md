# MakeupEmbeddingGAN-tf-Implement
* A tensorflow Implement for MakeupTransfer & MakeupEmbedding based on [cycleGAN](https://github.com/junyanz/CycleGAN/) and [ZM-Net](https://arxiv.org/pdf/1703.07255.pdf)
## Model Design
* Using AdaIN in parallel Pnet(StylePredictingNet) and Tnet(StyleTransferNet) Architecture  

![image](https://github.com/baldFemale/MakeupEmbeddingGAN-tf-Implement/raw/master/results/Architecture_1.png)  
![image](https://github.com/baldFemale/MakeupEmbeddingGAN-tf-Implement/raw/master/results/generator_arch.png)