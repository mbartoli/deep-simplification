perl ./tokenizer.perl -l en < ./training/training.normal.aligned > ./training/training.normal.aligned.tok
perl ./tokenizer.perl -l en < ./training/training.simple.aligned > ./training/training.simple.aligned.tok
perl ./tokenizer.perl -l en < ./validation/validation.normal.aligned > ./validation/validation.normal.aligned.tok
perl ./tokenizer.perl -l en < ./validation/validation.simple.aligned > ./validation/validation.simple.aligned.tok
perl ./tokenizer.perl -l en < ./testing/testing.normal.aligned > ./testing/testing.normal.aligned.tok
perl ./tokenizer.perl -l en < ./testing/testing.simple.aligned > ./testing/testing.simple.aligned.tok
