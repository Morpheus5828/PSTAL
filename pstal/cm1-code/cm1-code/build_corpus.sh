#!/usr/bin/bash

> dumas_train.txt
for f in ../../../talia/tp/data/alexandre_dumas/*.train.tok; do 
  cat $f | 
  awk '{if(NF >= 15) print $0}' |
  head -n 1200 |
  sed -E 's@ ?</?s> ?@@g' | 
  awk -v f=`basename ${f%.train.tok}` '{print(f, $0)}' |
  cat >> dumas_train.txt
done

> dumas_dev.txt
for f in ../../../talia/tp/data/alexandre_dumas/*.test.tok; do 
  cat $f |   
  awk '{if(NF >= 15) print $0}' |
  head -n 300 |
  sed -E 's@ ?</?s> ?@@g' | 
  awk -v f=`basename ${f%.test.tok}` '{print(f, $0)}' |
  cat >> dumas_dev.txt
done

> dumas_test.txt
for f in ../../../talia/tp/data/alexandre_dumas/*.test.tok; do 
  cat $f |   
  awk '{if(NF >= 15) print $0}' |
  tail -n 300 |
  sed -E 's@ ?</?s> ?@@g' | 
  awk -v f=`basename ${f%.test.tok}` '{print(f, $0)}' |
  cat >> dumas_test.txt
done
