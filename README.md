# cs287-f
Investigating Factual Accuracy in Abstractive Summarization Models

## Setup
```
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_lg
```

## Preprocessing model outputs
In data/pointer-generator, download data folder "test_output" from [here](https://drive.google.com/file/d/0B7pQmm-OfDv7MEtMVU5sOHc5LTg/view)
```
ls articles | cat > names_articles.txt
python clean_point-gen_data.py names_articles.txt
```

In data/bottom-up, download data files from [here](https://drive.google.com/file/d/1k-LqK3Lt7czIKyVrH_tr3P3Qd_39gLhk/view) and [here](https://drive.google.com/file/d/1EqiEVt3H7z7oCQBKkCO7MXoJkXM7Cipr/view)
```
cp test.txt.src.tagged.shuf.400words articles.txt
cp bottom_up_cnndm_015_threshold.out decoded.txt
cp test.txt.tgt.tagged.shuf.noslash reference.txt
```

In data/fast-abs-RL, download data folder "acl18_results" from [here](https://drive.google.com/file/d/1m1RIc9plJD2g2fhXUvwHLRtAhTVgYFKS/view)
```
ls references | cat > names_references.txt
python clean_fast-abs-RL_data.py names_references.txt
```

