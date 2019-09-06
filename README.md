## deepIQA model implemented on Pytorch

### **Warning! This code cannot reach the indexes mentioned in the paper!**

#### Unsolved problems:

1. **The model tend to overfit. **I 've tried my best to cover all specifications in the paper, adjusted the weight decay, but the network unfailingly overfits. On TID2013 dataset, if weight decay is not implemented, SROCC on validation set only can reach 0.5

2. Weight distribution is different from the model provided by the original authors.

#### Environment
1. Pytorch 1.1.0+ with GPU
1. tensorboardX

#### HOW TO USE the code:
1. **Generate the metadata for dataset.**
 TID2013 and LIVE2016 dataset are currently supported.
 You are supposed to modify the global vars in `gen_***.py` according to hints, to tell where is the dataset and where to save the metadata.
2. **Train the model.**
 Hyperparameters are stored in `params.yml` , modify the path. Dataset is set to TID2013 by default, if you want to switch to LIVE, uncomment the corresponding part.
 Then run `python3 main.py` The code will give the hint of how to launch Tensorboard to visualize the trainning process. In order to speed up training, **the code will copy the entire dataset to memory**, make sure that the memory is sufficient.
 By default, the code will report validation information every 10 epochs.