import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=100,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=2000,
                     dispFreq=100,
                     saveFreq=2000,
                     sampleFreq=2000,
                     datasets=['../data/training/training.normal.aligned.tok',
                               '../data/training/training.simple.aligned.tok'],
                     valid_datasets=['../data/validation/validation.normal.aligned.tok',
                                     '../data/validation/validation.simple.aligned.tok'],
                     dictionaries=['../data/dictionaries/simple.wiki.tok.pkl',
                                   '../data/dictionaries/simple.wiki.tok.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_hal.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})
