#!/usr/bin/env python3

import sys
import conllu
import collections
from torch.utils.data import TensorDataset, DataLoader
import torch
import random
import numpy as np
import pdb

########################################################################
# UTILITY FUNCTIONS
########################################################################

class Util(object):
  
  DEBUG_FLAG = False
  PSEUDO_INF = 9999.0

  ###############################

  @staticmethod
  def error(msg, *kwargs):
    print("ERROR:", msg.format(*kwargs), file=sys.stderr)
    sys.exit(-1)

  ###############################

  @staticmethod
  def warn(msg, *kwargs):
    print("WARNING:", msg.format(*kwargs), file=sys.stderr)    

  ###############################

  @staticmethod
  def debug(msg, *kwargs):
    if Util.DEBUG_FLAG:
      print(msg.format(*kwargs), file=sys.stderr)
      
  ###############################
  
  @staticmethod
  def rev_vocab(vocab):
    rev_dict = {y: x for x, y in vocab.items()}
    return [rev_dict[k] for k in range(len(rev_dict))]
    
  ###############################
  
  @staticmethod
  def dataloader(inputs, outputs, batch_size=16, shuffle=True):
    data_set = TensorDataset(*inputs, *outputs) 
    return DataLoader(data_set, batch_size, shuffle=shuffle)   
    
  ###############################
  
  @staticmethod
  def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
  ###############################
  
  @staticmethod
  def init_seed(seed):
    if seed >= 0:
      random.seed(seed)
      torch.manual_seed(seed)

  ###############################
  
  @staticmethod
  def log_cap(number):
    """Returns the base-10 logarithm of `number`.
    If `number` is negative, stops the program with an error message.
    If `number` is zero returns -9999.0 representing negative pseudo infinity
    This is more convenient than -np.inf returned by np.log10 because :
    inf + a = inf (no difference in sum) but 9999.0 + a != 9999.0"""
    if number < 0 :
      Util.error("Cannot get logarithm of negative number {}".format(number))
    elif number == 0:
      return -Util.PSEUDO_INF
    else :
      return np.log10(number)
    

########################################################################
# CONLLU FUNCTIONS 
########################################################################

class CoNLLUReader(object):  
 
  ###############################
  
  start_tag = "<s>"
  
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
      
  ###############################
    
  def readConllu(self):    
    for sent in conllu.parse_incr(self.infile):
      yield sent

  ###############################
  
  def name(self):
    return self.infile.name
    
  ###############################
  
  def morph_feats(self):
    morph_feats_list = set([])
    for sent in conllu.parse_incr(self.infile):
      for tok in sent :
        if tok["feats"] :
          for key in tok["feats"].keys():
            morph_feats_list.add(key ) 
    self.infile.seek(0) # Rewind open file        
    return list(morph_feats_list)

  ###############################

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
      # IMPORTANT : only works if "col_name" is the same as in lambda function definition!
      for col_name in col_name_dict.keys():
        int_list[col_name].append([vocab[col_name][tok[col_name]] for tok in s])    
    # vocabs cannot be saved if they have lambda function: erase default_factory
    for col_name in col_name_dict.keys():
      vocab[col_name].default_factory = None    
    return int_list, vocab
     
  ###############################

  def to_int_from_vocab(self, col_names, unk_token, vocab={}):  
    int_list = {}
    unk_toks = {}
    for col_name in col_names:  
      int_list[col_name] = []
      unk_toks[col_name] = vocab[col_name].get(unk_token,None)
    for s in self.readConllu():
      for col_name in col_names:
        id_getter = lambda v,t: v[col_name].get(t[col_name],unk_toks[col_name])
        int_list[col_name].append([id_getter(vocab,tok) for tok in s])        
    return int_list 
      
  ###############################

  @staticmethod
  def to_int_from_vocab_sent(sent, col_names, unk_token, vocab={}, 
                             lowercase=False):  
    int_list = {}    
    for col_name in col_names:
      unk_tok_id = vocab[col_name].get(unk_token, None)
      low_or_not = lambda w: w.lower() if lowercase else w
      id_getter = lambda v,t: v[col_name].get(low_or_not(t[col_name]),unk_tok_id)
      int_list[col_name]=[id_getter(vocab,tok) for tok in sent]
    return int_list 

  ###############################
    
  @staticmethod
  def to_bio(sent, bio_style='bio', name_tag='parseme:ne'):
    bio_enc = []
    neindex = 0
    for tok in sent :
      netag = tok[name_tag]
      if netag == '*' :
        cur_tag = 'O'
      elif netag == neindex :
        cur_tag = 'I' + necat
      else :
        neindex, necat = netag.split(":")
        necat = '-' + necat
        if bio_style == 'io' :
          cur_tag = 'I' + necat
        else:
          cur_tag = 'B' + necat
      bio_enc.append(cur_tag)      
    return bio_enc

###############################
    
  @staticmethod
  def from_bio(bio_enc, bio_style='bio', stop_on_error=False):
    """Converst BIO-encoded annotations into Sequoia/parseme format.
    Input `bio_enc` is a list of strings, each corresponding to one BIO tag.
    `bio_style` can be "bio" (default) or "io". Will try to recover encoding
    errors by replacing wrong tags when `stop_on_error` equals False (default),
    otherwise stops execution and shows an error message.  
    Only works for BIO-cat & IO-cat, with -cat appended to both B and I tags.
    Requires adaptations for BIOES, and encoding schemes without "-cat. 
    Examples:
    >>> from_bio(["B-PERS", "I-PERS", "I-PERS", "O", "B-LOC", "I-LOC"], bio_style='bio')
    ['1:PERS', '1', '1', '*', '2:LOC', '2']
    
    >>> from_bio(["B-PERS", "I-PERS", "I-PERS", "O", "B-LOC", "I-LOC"],bio_style='io')
    WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
    WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
    ['1:PERS', '1', '1', '*', '2:LOC', '2']
    
    >>> from_bio(["I-PERS", "B-PERS", "I-PERS", "O", "I-LOC"],bio_style='io')
    WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
    ['1:PERS', '1', '1', '*', '2:LOC']
    
    >>> from_bio(["I-PERS", "I-PERS", "I-PERS", "O", "I-LOC"], bio_style='bio')
    WARNING: Invalid I-initial tag I-PERS converted to B
    WARNING: Invalid I-initial tag I-LOC converted to B
    ['1:PERS', '1', '1', '*', '2:LOC']
    
    >>> from_bio(["I-PERS", "B-PERS", "I-PERS", "O", "I-LOC"], bio_style='bio')
    WARNING: Invalid I-initial tag I-PERS converted to B
    WARNING: Invalid I-initial tag I-LOC converted to B
    ['1:PERS', '2:PERS', '2', '*', '3:LOC']
    
    >>> from_bio(["I-PERS", "B-PERS", "I-EVE", "O", "I-PERS"], bio_style='io')
    ['1:PERS', '2:PERS', '3:EVE', '*', '4:PERS']
    
    >>> from_bio(["I-PERS", "B-PERS", "I-EVE", "O", "I-PERS"], bio_style='bio')
    WARNING: Invalid I-initial tag I-PERS converted to B
    WARNING: Invalid I-initial tag I-EVE converted to B
    WARNING: Invalid I-initial tag I-PERS converted to B
    ['1:PERS', '2:PERS', '3:EVE', '*', '4:PERS']
    """
    # TODO: warning if I-cat != previous I-cat or B-cat
    result = []
    neindex = 0
    prev_bio_tag = 'O'
    prev_cat = None
    for bio_tag in bio_enc :
      if bio_tag == 'O' :
        seq_tag = '*'                  
      elif bio_tag[0] in ['B', 'I'] and bio_tag[1] == '-':
        necat = bio_tag.split("-")[1]
        if bio_tag[0] == 'B' and bio_style == 'bio':
          neindex += 1 # Begining of an entity
          seq_tag = str(neindex) + ":" + necat
        elif bio_tag[0] == 'B' : # bio_style = 'io'
          if  stop_on_error:
            Util.error("B tag not allowed with 'io'")
          else:
            bio_tag = bio_tag.replace("B-", "I-")
            Util.warn("Got B tag in spite of 'io' bio_style: interpreted as I")
        if bio_tag[0] == "I" and bio_style == "io" :
          if necat != prev_cat:
            neindex += 1 # Begining of an entity
            seq_tag = str(neindex) + ":" + necat
          else: 
            seq_tag = str(neindex) # is a continuation
        elif bio_tag[0] == "I" : # tag is "I" and bio_style is "bio"
          if bio_style == 'bio' and prev_bio_tag != 'O' and necat == prev_cat : 
            seq_tag = str(neindex) # is a continuation
          elif stop_on_error : 
            Util.error("Invalid I-initial tag in BIO format: {}".format(bio_tag))
          else:
            neindex += 1 # Begining of an entity
            seq_tag = str(neindex) + ":" + necat
            Util.warn("Invalid I-initial tag {} converted to B".format(bio_tag))
        prev_cat = necat     
      else:
        if stop_on_error:
          Util.error("Invalid BIO tag: {}".format(bio_tag))
        else:
          Util.warn("Invalid BIO tag {} converted to O".format(bio_tag))
          result.append("*")
      result.append(seq_tag)      
      prev_bio_tag = bio_tag
    return result

################################################################################


