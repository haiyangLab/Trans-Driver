# Trans-Driver
Trans-Driver is a deep supervised learning framework utilizing a multilayer perceptron converter network with an unbalanced loss function. It integrates multi-omics data to learn both the differences and associations among diverse omics types for the identification of cancer driver genes. Compared with other state-of-the-art driver gene identification methods, Trans-Driver achieves excellent performance on benchmark datasets such as TCGA and CGC.Among ~20,000 protein-coding genes, Trans-Driver reported 269 candidate driver genes, of which 132 (about 49%) are present in the gold-standard CGC dataset. Further feature analysis demonstrated that integrating multi-omics data significantly improves predictive performance compared to using only somatic mutation data, and many identified candidates are clinically meaningful, confirming the practical utility of Trans-Driver.

Trans-Driver's dataset includes the model's training set, test set, and TCGA, CGC and PCAWG data, which you can find under ./data. Trans-Driver was evaluated using Fisher's test, and results for 33 cancers and comparison methods are in ./results. The Trans-Driver program includes model training and testing in ./program.

## We can run the Trans-Driver model with the following command: 
```Python
python ./program/Trans-Driver.py [--lr LEARNING_RATE] [--batch_size BATCH_SIZE] [--epoch EPOCHS] [--alpha ALPHA] [--gamma GAMMA]
```
You can adjust the training process of Trans-Driver using several command-line arguments: --lr sets the learning rate, --batch_size specifies the number of samples per batch, --epoch controls the number of training epochs, --alpha defines the alpha parameter for the focal loss function (to balance class weights), and --gamma sets the gamma parameter for the focal loss (to focus learning on hard-to-classify samples). These options allow flexible tuning of model training according to your needs.

###Example:
```Python
python ./program/Trans-Driver.py --lr 0.00089 --batch_size 16 --epoch 30
```
## Trans-Driver's performance comparison module is used as follows: 
```Python
python ./program/performance.py
```
## The results analysis module of Trans-Driver is used as follows:
```Bash
./program/analysis.R  
./program/evaluations.R  
./program/fea_imports.R
```
Trans-Driver's model implementation is based on Pytorch. its dependency packages are: Python (3.7.10), PyTorch (1.8.1), NumPy (1.19.5), Pandas (1.2.4), Keras (2.4.3), Scipy(1.6.2). The operating system is windows10. The CPU is an Intel Xeon Platinum 8255C (2.50 GHz), and the GPU is NVIDIA GeForce RTX 3090.
