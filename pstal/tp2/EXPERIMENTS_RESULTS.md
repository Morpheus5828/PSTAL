## EXPERIMENTS CONFIGS AND RESULTS

### Hidden Markov Model - Viterbi relying on counts

````shell
Predictions file: pstal/tp2/predictions.test.conllu
Accuracy on all parseme:ne: 89.22 ( 8563/ 9598)

Precision, recall, and F-score for parseme:ne:
Exact-EVE  : P= 40.74 (   11/   27) / R= 55.00 (   11/   20) / F= 46.81
Exact-LOC  : P= 76.92 (   30/   39) / R= 39.47 (   30/   76) / F= 52.17
Exact-ORG  : P= 47.83 (   55/  115) / R= 45.08 (   55/  122) / F= 46.41
Exact-PERS : P= 52.38 (   88/  168) / R= 72.13 (   88/  122) / F= 60.69
Exact-PROD : P= 59.46 (   44/   74) / R= 66.67 (   44/   66) / F= 62.86
Exact-nocat: P= 57.45 (  243/  423) / R= 59.85 (  243/  406) / F= 58.62
Fuzzy-nocat: P= 51.90 (  614/ 1183) / R= 76.75 (  614/  800) / F= 61.93
macro-avg  : P= 55.24               / R= 59.28               / F= 57.19

````

### Hidden Markov Model with RNN

Config : D_EMB = 512, D_HIDDEN = 64, BATCH_SIZE = 128,
    NB_EPOCHS = 50,
    MAX_LEN = 50,
    LEARNING_RATE = 0.001,
    WEIGHT_DECAY = 0,
    LABEL_SMOOTHING = 0.00005,
    FEATURE_FUNCTIONS = [
            first_letter_uppercase,
            all_letters_uppercase,
            contains_digits,
            contains_hyphen,
            is_noun,
            is_propn
        ]

````shell
Predictions file: pstal/tp2/predictions.rnn.test.conllu
Accuracy on all parseme:ne: 93.79 ( 9002/ 9598)

Precision, recall, and F-score for parseme:ne:
Exact-EVE  : P= 73.33 (   11/   15) / R= 55.00 (   11/   20) / F= 62.86
Exact-LOC  : P= 60.34 (   35/   58) / R= 46.05 (   35/   76) / F= 52.24
Exact-ORG  : P= 51.82 (   57/  110) / R= 46.72 (   57/  122) / F= 49.14
Exact-PERS : P= 70.37 (   19/   27) / R= 15.57 (   19/  122) / F= 25.50
Exact-PROD : P= 72.22 (   52/   72) / R= 78.79 (   52/   66) / F= 75.36
Exact-nocat: P= 73.76 (  208/  282) / R= 51.23 (  208/  406) / F= 60.47
Fuzzy-nocat: P= 88.76 (  458/  516) / R= 57.25 (  458/  800) / F= 69.60
macro-avg  : P= 70.09               / R= 50.09               / F= 58.42

Predictions file: pstal/tp2/predictions.rnnsf.test.conllu
Accuracy on all parseme:ne: 94.41 ( 9061/ 9598)

Precision, recall, and F-score for parseme:ne:
Exact-EVE  : P= 68.18 (   15/   22) / R= 75.00 (   15/   20) / F= 71.43
Exact-LOC  : P= 61.54 (   32/   52) / R= 42.11 (   32/   76) / F= 50.00
Exact-ORG  : P= 51.79 (   58/  112) / R= 47.54 (   58/  122) / F= 49.57
Exact-PERS : P= 90.00 (   27/   30) / R= 22.13 (   27/  122) / F= 35.53
Exact-PROD : P= 71.83 (   51/   71) / R= 77.27 (   51/   66) / F= 74.45
Exact-nocat: P= 78.05 (  224/  287) / R= 55.17 (  224/  406) / F= 64.65
Fuzzy-nocat: P= 92.62 (  502/  542) / R= 62.75 (  502/  800) / F= 74.81
macro-avg  : P= 73.43               / R= 54.57               / F= 62.61

````

Config: D_EMB = 120,
    D_HIDDEN = 32,
    BATCH_SIZE = 32,
    NB_EPOCHS = 50,
    MAX_LEN = 50,
    LEARNING_RATE = 0.001,
    WEIGHT_DECAY = 0 #1e-5,
    LABEL_SMOOTHING = 0.00005,
    FEATURE_FUNCTIONS = [
        first_letter_uppercase,
        all_letters_uppercase,
        contains_digits,
        contains_hyphen,
        is_noun,
        is_propn
    ]

```shell
Predictions file: pstal/tp2/predictions.rnn2.test.conllu
Accuracy on all parseme:ne: 93.52 ( 8976/ 9598)

Precision, recall, and F-score for parseme:ne:
Exact-EVE  : P= 72.22 (   13/   18) / R= 65.00 (   13/   20) / F= 68.42
Exact-LOC  : P= 22.03 (   26/  118) / R= 34.21 (   26/   76) / F= 26.80
Exact-ORG  : P= 50.88 (   58/  114) / R= 47.54 (   58/  122) / F= 49.15
Exact-PERS : P= 87.50 (    7/    8) / R=  5.74 (    7/  122) / F= 10.77
Exact-PROD : P= 70.15 (   47/   67) / R= 71.21 (   47/   66) / F= 70.68
Exact-nocat: P= 73.85 (  240/  325) / R= 59.11 (  240/  406) / F= 65.66
Fuzzy-nocat: P= 85.67 (  532/  621) / R= 66.50 (  532/  800) / F= 74.88
macro-avg  : P= 66.04               / R= 49.90               / F= 56.85

Predictions file: pstal/tp2/predictions.rnnsf2.test.conllu
Accuracy on all parseme:ne: 94.06 ( 9028/ 9598)

Precision, recall, and F-score for parseme:ne:
Exact-EVE  : P= 81.25 (   13/   16) / R= 65.00 (   13/   20) / F= 72.22
Exact-LOC  : P= 54.12 (   46/   85) / R= 60.53 (   46/   76) / F= 57.14
Exact-ORG  : P= 37.08 (   66/  178) / R= 54.10 (   66/  122) / F= 44.00
Exact-PERS : P= 75.00 (    3/    4) / R=  2.46 (    3/  122) / F=  4.76
Exact-PROD : P= 79.03 (   49/   62) / R= 74.24 (   49/   66) / F= 76.56
Exact-nocat: P= 82.03 (  283/  345) / R= 69.70 (  283/  406) / F= 75.37
Fuzzy-nocat: P= 94.16 (  564/  599) / R= 70.50 (  564/  800) / F= 80.63
macro-avg  : P= 71.81               / R= 56.65               / F= 63.33
```