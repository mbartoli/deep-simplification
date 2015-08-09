Attention-based encoder-decoder model for machine translation

## Training
Change the hard-coded paths to data (if you're not using the wiki data) in `nmt.py` then run
```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```

## Appendix
See http://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/

Thanks to Kyunghyun Cho for the neural machine translation code
