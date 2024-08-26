### Source of the corpus 

The corpus was downloaded from the Deep Sequoia's git repository:
[Deep Sequoia repository](https://gitlab.inria.fr/sequoia/deep-sequoia)

We obtained the file `trunk/sequoia-ud.parseme.frsemcor` from commit number `ea758e94`
The file is the result of the conversion from Sequoia's source as described on the [documentation](https://deep-sequoia.inria.fr/process/)

We keep the original file in `src` folder to make command line completion faster
The file `tiny.conllu` was manually extracted and simplified, it is used in parsing exercises.

### Simplification

We run `simplify_sequoia.py` to make the file easier to work with. 
Look at the header of the script to understand the simplifications carried out.
Justifications for each simplification are indicated below their description with an arrow "=>"
Here is the log of the simplifications:

```
./bin/simplify_sequoia.py src/sequoia-ud.parseme.frsemcor > sequoia-ud.parseme.frsemcor.simple.full

1952 range tokens removed.

416 discontinuous and overlapping NEs removed.

6578 supersense tags removed (on MWEs or strange POS).
1282 supersense tags modified (complex operators).

66 non-projective sentences removed:
annodis.er_00018, annodis.er_00026, annodis.er_00043, annodis.er_00079, annodis.er_00092, annodis.er_00099, annodis.er_00195, annodis.er_00204, annodis.er_00214, annodis.er_00218, annodis.er_00267, annodis.er_00357, annodis.er_00386, annodis.er_00400, annodis.er_00410, annodis.er_00448, annodis.er_00473, annodis.er_00475, annodis.er_00477, annodis.er_00519, emea-fr-dev_00024, emea-fr-dev_00044, emea-fr-test_00043, emea-fr-test_00207, emea-fr-test_00274, emea-fr-test_00472, emea-fr-test_00499, Europar.550_00004, Europar.550_00005, Europar.550_00029, Europar.550_00041, Europar.550_00043, Europar.550_00209, Europar.550_00257, Europar.550_00276, Europar.550_00289, Europar.550_00295, Europar.550_00303, Europar.550_00307, Europar.550_00335, Europar.550_00355, Europar.550_00367, Europar.550_00473, Europar.550_00476, Europar.550_00488, Europar.550_00518, Europar.550_00544, frwiki_50.1000_00066, frwiki_50.1000_00087, frwiki_50.1000_00091, frwiki_50.1000_00124, frwiki_50.1000_00137, frwiki_50.1000_00215, frwiki_50.1000_00246, frwiki_50.1000_00305, frwiki_50.1000_00318, frwiki_50.1000_00381, frwiki_50.1000_00426, frwiki_50.1000_00455, frwiki_50.1000_00523, frwiki_50.1000_00565, frwiki_50.1000_00621, frwiki_50.1000_00655, frwiki_50.1000_00829, frwiki_50.1000_00843, frwiki_50.1000_00867
```

### Splitting

The files are then split into train, dev and test according to the IDs, following the official UD release.
We got the IDs from [the tools folder](https://gitlab.inria.fr/sequoia/deep-sequoia/tree/master/tools)
The script `conllu_filter.py` was adapted to consider 

```
bin/conllu_filter.py bin/train.ids sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.train
bin/conllu_filter.py bin/dev.ids   sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.dev
bin/conllu_filter.py bin/test.ids  sequoia-ud.parseme.frsemcor.simple.full > sequoia-ud.parseme.frsemcor.simple.test
```

We also generate a toy small version of the corpus that can be useful to test your code during development:
```
NBSENT=30
CORPUS=sequoia-ud.parseme.frsemcor.simple.dev
LASTID=`grep "sent_id" ${CORPUS} | head -n  $((NBSENT+1)) | tail -n 1 | sed 's/.* \([^ ]*\)$/\1/g'`
CUTLINE=`grep -n ${LASTID} ${CORPUS} | sed 's/:.*//g'`
head -n $((CUTLINE-1)) ${CORPUS} > sequoia-ud.parseme.frsemcor.simple.small
```

Finally, we also split the non-simplified version of the corpus into train, dev and test (before simplification).
These files should not be used in your experiments.
```
bin/conllu_filter.py bin/train.ids src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.train
bin/conllu_filter.py bin/dev.ids   src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.dev
bin/conllu_filter.py bin/test.ids  src/sequoia-ud.parseme.frsemcor > src/sequoia-ud.parseme.frsemcor.simple.test
```

### Known compatibility issues

Notice that this version of the corpus is not 100% up-to-date with last UD version of Sequoia.
This can be noticed by the number of non-projective sentences [given by Grew-Match](https://universal.grew.fr/?custom=6697d0b0343b8):
`global { is_not_projective }` on UD_French-Sequoia@2.14 returns 79 occurrences whereas we remove 66 non-projective sentences (see details above)
We suppose that latest UD Sequoia version includes changes that were not reported to DeepSequoia's git repository.
We do not investigate this further, but you can contact the corpus maintainers if you absolutely need to solve this mystery.
