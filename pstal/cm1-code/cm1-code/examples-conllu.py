#!/usr/bin/env python

import conllu, sys
conllufile = open(sys.argv[1], 'r', encoding='UTF-8')
slens, wlens = [], []
for sent in conllu.parse_incr(conllufile):
  slens.append(len(sent))  
  wlens.extend([len(token['form']) for token in sent])
print("Avg sent len={:.2f}".format(sum(slens)/len(slens)))
print("Avg word len={:.2f}".format(sum(wlens)/len(wlens)))

import matplotlib.pyplot as plt
f,(a1,a2) = plt.subplots(1,2)
a1.hist(slens,bins=20)
a1.set_title("Sentence length")
a2.hist(wlens,bins=20)
a2.set_title("Word length")
plt.show()

