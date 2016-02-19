# deep-simplification
Text simplification using RNNs

## Training
Navigate to the simplify directory and run
```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```

## Sampling
Navigate to the simplify directory and run
```
THEANO_FLAGS=device=cpu,floatX=float32 python translate.py -n -p 1 model_hal.npz ../data/normal-dict.tok.pkl ../data/simple-dict.tok.pkl ../data/testing/testing-normal.tok ../data/testing/testing-simple-out.tok
```
