for i in bottom-up pointer-generator fast-abs-RL; do nohup python3 testing.py --document data/$i/articles.txt --summary data/$i/decoded.txt --reference data/$i/reference.txt  --cache-dir experiments/fast-abs-RL/ --rouge --copy --print-scores > $i.out & done
