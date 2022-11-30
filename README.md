# A comparison of machine learning methods for recommending clinical trials for inclusion in new systematic reviews 

This module facilitates the ranking of candidates given query files according to the predicted probabilities from the model.

This is the demo package for discovering the unestablished links between clnical trial articles and registraions and systematic review andprotocols

This package includes source codes for evaluation (ground truth exists) and inference.

## Packagge Requirements

All the codes are written using Python 3.7.7 and the experiments are run on Linux.
Necessary installed pacakges including:
* numpy
* scikit-learn
* csv

## How to Inference

Please check the manual_evaluation.sh file under the scripts folder

## Data Description
We recommend constructing your own datasets in your applications to avoid time-variance and domain shift bias.

    |-data
        |-clinical_trial_article -> trial articl related files
            |-connections.csv -> the automatically generaget ground truth of clnical trial nctid  and sysmtetic review pmid pairs. This is used for the auto-mined data evaluation. Only connections with pmid in the input systematic review file are triggered.
            |-extended_clinical_trials.csv -> the clinical trial article extracted text
            |-extended_clinical_trials.json -> the clinical trial article with extracted PICO and MetaMap normalized PICO
            |-extended_clinical_trials_metamap.json -> the clinical trial article with MetaMap extracted term
        |-clinical_trial_registration_pico_text -> trial registration related files
            |-clinical_trials_pico_metamap.json -> the clinical trial registration with MetaMap extracted term
            |-clinical_trials_pico_text.csv -> the clinical trial registration extracted text
            |-clinical_trials_pico_text.json -> the clinical trial registration with extracted PICO and MetaMap normalized PICO
        |-prospero -> manually curated systematic review protocol related files
            |-manual_prospero_systematic_reviews.csv -> the systematic review protocol extracted text
            |-manual_prospero_systematic_reviews.json -> the systematic review protocol with extracted PICO and MetaMap normalized PICO
            |-manual_prospero_systematic_reviews_metamap.json -> the systematic review protocol with MetaMap extracted term
            |-systematic_review_crd_num.csv -> a mapping between systematic review pmid and crdid           
        |-systematic_review -> manually curated systematic review related files
            |-manual_pmid2connections.json -> the manually curated connections between systematic review pmid with the linked clinical trial articles and registrations
            |-manual_systematic_reviews.csv -> the systematic review  extracted text
            |-manual_systematic_reviews.json -> the systematic review  with extracted PICO and MetaMap normalized PICO
            |-manual_systematic_reviews_metamap.json -> the systematic review  with MetaMap extracted term

## Source Code Description
    |-src
        |- __init__.py
        |-model
            |- __init__.py
            |-bert_pico_tfidf.py -> the model with BERT extracted PICO or MetaMap normalized BERT extracted PICO
            |-metamap_pico_tfidf.py -> the model with MetaMap extracted terms
            |-tfidf_doc_similarity.py -> the model with text tokens
        |-utils
            |- __init__.py
            |-filters.py -> has filter text by date functions
            |-nlp.py -> has stopwords, tokenization related functions
            |-utils.py -> has flatten array function
    
        |-main_bert_pico_tfidf_doc_sim.py -> apply model from bert_pico_tfidf.py for evaluation
        |-main_bert_pico_tfidf_doc_sim_predict.py -> apply model from bert_pico_tfidf.py for inference
        |-main_metamap_pico_tfidf_doc_sim.py -> apply model from metamap_pico_tfidf.py for evaluation
        |-main_metamap_pico_tfidf_doc_sim_predict.py -> apply model from metamap_pico_tfidf.py for inference
        |-main_tfidf_doc_sim.py -> apply model from tfidf_doc_similarity.py for evaluation
        |-main_tfidf_doc_sim_predict.py -> apply model from tfidf_doc_similarity.py for inference
