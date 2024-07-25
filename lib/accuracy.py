#!/usr/bin/env python3

import sys
import argparse
import collections
import pdb
from conllulib import CoNLLUReader, Util

################################################################################

parser = argparse.ArgumentParser(description="Calculates the accuracy of a \
prediction with respect to the gold file. By default, uses UPOS, but this can \
be configured with option -c.",  
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-D', "--debug", action="store_true", dest="DEBUG_FLAG", 
        help="""Print debug information (grep it or pipe into `less -SR`)""")
parser.add_argument('-p', "--pred", metavar="FILENAME.conllu", required=True,\
        dest="pred_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Test corpus in CoNLLU with *predicted* tags. (Required)""")
parser.add_argument('-g', "--gold", metavar="FILENAME.conllu", required=True,\
        dest="gold_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Test corpus in CoNLLU with *gold* tags. (Required)""")
parser.add_argument('-t', "--train", metavar="FILENAME.conllu", required=False,\
        dest="train_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Training corpus in CoNLL-U, from which tagger was learnt.""")        
parser.add_argument('-c', "--tagcolumn", metavar="NAME", dest="col_name_tag",
        required=False, type=str, default="upos", help="""Column name of tags, \
        as defined in header. Use lowercase""")   
parser.add_argument('-f', "--featcolumn", metavar="NAME", dest="col_name_feat",
        required=False, type=str, default="form", help="""Column name of input 
        feature, as defined in header. Use lowercase.""")
parser.add_argument('-u', "--upos-filter", metavar="NAME", dest="upos_filter",
        required=False, type=str, nargs='+', default=[], 
        help="""Only calculate accuracy for words with UPOS in this list. \
        Empty list = no filter.""")        
                            
########################################################################

def process_args(parser):  
  args = parser.parse_args()
  Util.DEBUG_FLAG = args.DEBUG_FLAG
  args.col_name_tag = args.col_name_tag.lower()
  args.col_name_feat = args.col_name_feat.lower()
  Util.debug("Command-line arguments and defaults:")
  for (k,v) in vars(args).items():
    Util.debug("  * {}: {}",k,v)    
  gold_corpus = CoNLLUReader(args.gold_filename) 
  pred_corpus = CoNLLUReader(args.pred_filename) 
  train_vocab = None
  if args.train_filename:
    train_corpus = CoNLLUReader(args.train_filename)
    ignoreme, train_vocab = train_corpus.to_int_and_vocab({args.col_name_feat:[]})    
  if args.col_name_tag  not in gold_corpus.header or \
     args.col_name_feat not in gold_corpus.header:
    Util.error("-c and -f names must be valid conllu column among:\n{}", 
               gold_corpus.header)
  return args, gold_corpus, pred_corpus, train_vocab
  
########################################################################
 
if __name__ == "__main__":
  args, gold_corpus, pred_corpus, train_vocab = process_args(parser)
  total_tokens = correct_tokens = 0
  total_oov = correct_oov = 0
  for (sent_gold, sent_pred) in zip(gold_corpus.readConllu(),
                                    pred_corpus.readConllu()):
    for (tok_gold, tok_pred) in zip (sent_gold, sent_pred):
      if not args.upos_filter or tok_gold['upos'] in args.upos_filter :
        if train_vocab :
          train_vocab_feat = train_vocab[args.col_name_feat].keys()
          if tok_gold[args.col_name_feat] not in train_vocab_feat:
            total_oov = total_oov + 1
            oov = True
          else:
            oov = False
        if tok_gold[args.col_name_tag] == tok_pred[args.col_name_tag]:        
          correct_tokens = correct_tokens + 1        
          if train_vocab and oov :
            correct_oov = correct_oov + 1
        total_tokens += 1
  
  print("Pred file: {}".format(pred_corpus.name()))
  if args.upos_filter :
    print("Results focus only on following UPOS: {}".format(" ".join(args.upos_filter)))
  accuracy = (correct_tokens / total_tokens) * 100  
  print("Accuracy on all {}: {:0.2f} ({}/{})".format(args.col_name_tag,accuracy,
                                                 correct_tokens, total_tokens))
  if train_vocab :
    accuracy_oov = (correct_oov / total_oov) * 100
    print("Accuracy on OOV {}: {:0.2f} ({}/{})".format(args.col_name_tag,
                                                 accuracy_oov,
                                                 correct_oov, total_oov))
