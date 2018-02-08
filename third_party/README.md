# Third-Party Libraries

## warp-ctc

This is a modification of the Baidu's Warp-CTC binding for PyTorch, originally
created by Sean Naren (https://github.com/SeanNaren/warp-ctc).

The only difference is that his PyTorch binding returned the loss of the whole
batch, while my version returns a vector with the loss of each individual
element in the batch. PyLaia requires this to monitor the loss of the
samples in the batch, and because we actually typically minimize the loss
normalized by the length of the sample.
