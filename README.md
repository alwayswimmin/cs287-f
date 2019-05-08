# cs287-f

## Setup
```
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Preprocessing pointer-generator outputs
In data/pointer-generator, download pointer-generator folder "test_output"
```
ls articles | cat > names_articles.txt
python clean_point-gen_data.py names_articles.txt
```
