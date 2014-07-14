# net-Toolkit

A simple module for [`Torch7`](https://github.com/torch/torch7) and the [`nn` package](https://github.com/torch/nn).

## Installation

```bash
luarocks install --server=https://raw.github.com/Atcold/net-toolkit/master net-toolkit
```

## Description

This package allows to save and retrive to/from disk a lighter version of the network that is being training.

 - [`saveNet()`](#savenet)
 - [`loadNet()`](#loadnet)

### `saveNet()`

`saveNet()` saves a lighter version of your current network, removing all unnecessary data from it (such as *gradients*, *activation units*' state and etc...) and returns a new couple of flattened *weight* and *gradients*. Usage:

```lua
w, dw = saveNet(model, fileName)
```

### `saveNetFields()`

`saveNetFields()` saves your current network, removing all `Tensor` data you don't want to save and returns a new couple of flattened *weight* and *gradients*. Usage:

```lua
w, dw = saveNetFields(model, fileName, {'weight', 'bias'})
```
Only `weight` and `bias` `Tensor`s will be saved and the rest will be discarded.

### `loadNet()`

Let's say we would like to load a network we have previously saved with `saveNet()` for continuing a training session on it. Some inner parameters (something about *gradients*) have to be restored, since `saveNet()` did a pruning operation on the network in order to save space. Here is how we can handle this case:

```lua
model, w, dw = loadNet(fileName)
```

Now we can keep training, perhaps without forgetting to (re-)define a [*criterion* `loss`](https://github.com/torch/nn/blob/master/README.md#criterions) (the criterion is not saved with the network, so we have to re-define it, if we don't already do it somewhere else in the code).
