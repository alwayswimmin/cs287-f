# cs287-f

## Setup
```
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Preprocessing model outputs
In data/pointer-generator, download data folder "test_output" from (https://drive.google.com/file/d/0B7pQmm-OfDv7MEtMVU5sOHc5LTg/view "here")
```
ls articles | cat > names_articles.txt
python clean_point-gen_data.py names_articles.txt
```

In data/fast-abs-RL, download data folder "acl18_results" from (https://drive.google.com/file/d/1m1RIc9plJD2g2fhXUvwHLRtAhTVgYFKS/view "here")
```
ls references | cat > names_references.txt
python clean_fast-abs-RL_data.py names_references.txt
```

