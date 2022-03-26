# Trans-Driver
Trans-Driver, a deep supervised learning method, a multilayer perceptron converter network with an unbalanced loss function, which integrates multi-omics data to learn the differences and associations between different omics data for cancer drivers’ discovery. Compared with other state-of-the-art driver gene identification methods, Trans-Driver has achieved excellent performance on TCGA and CGC data sets. Among ~20,000 protein-coding genes, Trans-Driver reported 189 candidate driver genes, of which 105 genes (about 55%) were included in the gold standard CGC data set. Finally, we analyzed the contribution of each feature to the identification of driver genes. We found that the integration of multi-omics data can improve the performance of our method compared with using only somatic mutation data. Through detailed analysis, we found that the candidate drivers are clinically meaningful, proving the practicability of Trans-Driver.

Trans-Driver's dataset includes the model's training set, test set, and TCGA and CGC data, which you can find under ./data. Trans-Driver was evaluated using Fisher's test, and results for 33 cancers and comparison methods are in ./results. The Trans-Driver program includes model training and testing in ./program.

## We can run the Trans-Driver model with the following command: 
```Python
python ./program/Trans-Driver.py
```
## Trans-Driver's performance comparison module is used as follows: 
```Python
python ./program/performance.py
```
## The result analysis module of Trans-Driver is used as follows:
```R
analysis.R  evaluation.R  fea_imports.R
```
Trans-Driver's model implementation is based on Pytorch. its dependency packages are: Python (3.7.10), PyTorch (1.8.1), NumPy (1.19.5), Pandas (1.2.4), Keras (2.4.3), Scipy(1.6.2). The operating system is windows10. The CPU is Intel Core i5-10400 (2.90 GHz), and the GPU is NVIDIA GeForce GTX 1650 Ti.
