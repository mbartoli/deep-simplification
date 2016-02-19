perl ./tokenizer.perl -l en < ./training/training-normal > ./training/training-normal.tok
perl ./tokenizer.perl -l en < ./training/training-simple > ./training/training-simple.tok
perl ./tokenizer.perl -l en < ./validation/validation-normal > ./validation/validation-normal.tok
perl ./tokenizer.perl -l en < ./validation/validation-simple > ./validation/validation-simple.tok
