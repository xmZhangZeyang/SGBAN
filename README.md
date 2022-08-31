# SGBAN

## Self-Growing Binary-Activation Network (SGBAN)

This is a Tensorflow implementation of SGBAN described in the following papers:
《Self-Growing Binary-Activation Network: a Novel Deep Learning Model with Dynamic Architecture》. 

This article is available at https://doi.org/10.1109/TNNLS.2022.3176027.

## Requirements
The current version of the code has been tested with:

* tensorflow 1.12.0

## Usage

Use this to build the SGBAN:

```
net = SGBAN.Network(input_dimension, output_dimension, '/file_name')
```

Save or restore the model:

```
net.save()
net.load()
```

Training:

```
net.in_memory_data = copy.deepcopy(train_set)
net.in_memory_labels = copy.deepcopy(train_set_labels)
net.out_memory_data = []
net.out_memory_labels = []
net.check_memory()

while len(net.out_memory_data):
  net.training()
```
