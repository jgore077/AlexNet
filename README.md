# AlexNet
This is my attempt at implementing AlexNet using pytorch. The model is then trained for the CIFAR-10 classification task in `train.py`.

# Installation 
You will need all the associated libraries from `torch` and `tqdm`.

# Running
To run the training script you need to provide the number of epochs & batch size (ints) along with the learning rate (float). You also need to provide a path for the model files to be stored after every epoch.
```
python train.py <epochs> <batch_size> <learn_rate> <model_folder>
```
Here's an example of how to run the training script.
```
python train.py 10 64 .001 models/base_model
```
This will train the model for 10 epochs using a batch size of 64 with a learning rate of .001, after each epoch the weights will be saved in models/base_model.
 