#!/usr/bin/env python3

import sys
import conllu
import collections
import pdb

########################################################################
# UTILITY FUNCTIONS
########################################################################

class Util(object):
  
  DEBUG_FLAG = False

########################################################################

  @staticmethod
  def error(msg, *kwargs):
    print("ERROR:", msg.format(*kwargs), file=sys.stderr)
    sys.exit(-1)

########################################################################

  @staticmethod
  def debug(msg, *kwargs):
    if Util.DEBUG_FLAG:
      print(msg.format(*kwargs), file=sys.stderr)
      
  @staticmethod
  def rev_vocab(vocab):
    rev_dict = {y: x for x, y in vocab.items()}
    return [rev_dict[k] for k in range(len(rev_dict))]

########################################################################
# CONLLU FUNCTIONS 
########################################################################

class CoNLLUReader(object):  
 
  ###########################################
  
  def __init__(self, infile):
    self.infile = infile
    DEFAULT_HEADER = "ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC " +\
                     "PARSEME:MWE FRSEMCOR:NOUN PARSEME:NE"
    try:
      first = self.infile.readline().strip() # First line in the file
      globalcolumns = conllu.parse(first)[0].metadata['global.columns']
      self.header = globalcolumns.lower().split(" ")
      self.infile.seek(0) # Rewind open file
    except KeyError:
      self.header = DEFAULT_HEADER.split(" ")
      
  ###########################################
    
  def readConllu(self):    
    for sent in conllu.parse_incr(self.infile):
      yield sent

  ###########################################
  
  def name(self):
    return self.infile.name

  ###########################################

  def to_int_and_vocab(self, col_name_dict):  
    int_list = {}; 
    vocab = {}
    for col_name, special_tokens in col_name_dict.items():  
      int_list[col_name] = []      
      vocab[col_name] = collections.defaultdict(lambda: len(vocab[col_name]))
      for special_token in special_tokens:
        # Simple access to undefined dict key creates new ID (dict length)
        vocab[col_name][special_token]       
    for s in self.readConllu():
      for col_name in col_name_dict.keys():
        int_list[col_name].append([vocab[col_name][tok[col_name]] for tok in s])    
    # vocabs cannot be saved if they have lambda function: erase default_factory
    for col_name in col_name_dict.keys():
      vocab[col_name].default_factory = None    
    return int_list, vocab
     
  ###########################################

  def to_int_from_vocab(self, col_name_dict, unk_token, vocab={}):  
    int_list = {}
    unk_toks = {}
    for col_name, special_tokens in col_name_dict.items():  
      int_list[col_name] = []
      unk_toks[col_name] = vocab[col_name].get(unk_token,None)
    for s in self.readConllu():
      for col_name in col_name_dict.keys():
        id_getter = lambda v,t: v[col_name].get(t[col_name],unk_toks[col_name])
        int_list[col_name].append([id_getter(vocab,tok) for tok in s])        
    return int_list 
      
  ###########################################

  @staticmethod
  def to_int_from_vocab_sent(sent, col_name_dict, unk_token, vocab={}):  
    int_list = {}    
    for col_name in col_name_dict.keys():
      unk_tok_id = vocab[col_name].get(unk_token,None)
      id_getter = lambda v,t: v[col_name].get(t[col_name],unk_tok_id)
      int_list[col_name]=[id_getter(vocab,tok) for tok in sent]
    return int_list 

