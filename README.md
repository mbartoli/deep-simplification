# deep-simplification
Text simplification using RNNs

## Training
Navigate to the RNN directory and run
```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```

## Sampling
Navigate to the RNN directory and run
```
THEANO_FLAGS=device=gpu,floatX=float32 python simplify.py [trained_model] [dict_source] [dict_target] [source] [saveto]
```

