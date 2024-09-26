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
  """
  Utility static functions that can be useful (but not required) in any script.
  """
  
  DEBUG_FLAG = False 
  PSEUDO_INF = 9999.0         # Pseudo-infinity value, useful for Viterbi (TP2)

  ###############################

  @staticmethod
  def error(msg, *kwargs):
    """
    Shows an error message `msg` on standard error output, and terminates.
    Any `kwargs` will be forwarded to `msg.format(...)`
    """
    print("ERROR:", msg.format(*kwargs), file=sys.stderr)
    sys.exit(-1)

  ###############################

  @staticmethod
  def warn(msg, *kwargs):
    """
    Shows a warning message `msg` on standard error output.
    Any `kwargs` will be forwarded to `msg.format(...)`
    """
    print("WARNING:", msg.format(*kwargs), file=sys.stderr)    

  ###############################

  @staticmethod
  def debug(msg, *kwargs):
    """
    Shows a message `msg` on standard error output if `DEBUG_FLAG` is true
    Any `kwargs` will be forwarded to `msg.format(...)`
    """
    if Util.DEBUG_FLAG:
      print(msg.format(*kwargs), file=sys.stderr)
      
  ###############################
  
  @staticmethod
  def rev_vocab(vocab):
    """
    Given a dict vocabulary with str keys and unique int idx values, returns a 
    list of str keys ordered by their idx values. The str key can be obtained
    by acessing the reversed vocabulary list in position rev_vocab[idx]. 
    Example:
    >>> print(Util.rev_vocab({"a":0, "b":1,"c":2}))
    ['a', 'b', 'c']
    >>> print(Util.rev_vocab({"a":2, "b":0, "c":1}))
    ['b', 'c', 'a']
    """
    rev_dict = {y: x for x, y in vocab.items()}
    return [rev_dict[k] for k in range(len(rev_dict))]
    
  ###############################
  
  @staticmethod
  def dataloader(inputs, outputs, batch_size=16, shuffle=True):
    """
    Given a **list** of `input` and a list of `output` torch tensors, returns a
    DataLoader where the tensors are shuffled and batched according to `shuffle`
    and `batch_size` parameters. Notice that `inputs` and `outputs` need to be
    aligned, that is, their dimension 0 has identical sizes in all tensors.
    """
    data_set = TensorDataset(*inputs, *outputs) 
    return DataLoader(data_set, batch_size, shuffle=shuffle)   
    
  ###############################
  
  @staticmethod
  def count_params(model):
    """
    Given a class that extends torch.nn.Module, returns the number of trainable
    parameters of that class.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
  ###############################
  
  @staticmethod
  def init_seed(seed):
    """
    Initialise the random seed generator of python (lib random) and torch with
    a single int random seed value. If the value is zero or negative, the random
    seed will not be deterministically initialised. This can be useful to obtain
    reproducible results across runs.
    """
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
    """
    Extract the list of morphological features from the "FEATS" field of the
    CoNLL-U file. At the end, rewinds the file so that it can be read through 
    again. The result is a list of unique strings corresponding to the keys 
    appearing in the FEATS column of the corpus (before the = sign)
    """
    morph_feats_list = set([])
    for sent in conllu.parse_incr(self.infile):
      for tok in sent :
        if tok["feats"] :
          for key in tok["feats"].keys():
            morph_feats_list.add(key ) 
    self.infile.seek(0) # Rewind open file        
    return list(morph_feats_list)

  ###############################

  def to_int_and_vocab(self, col_name_dict, extra_cols_dict={}):  
    int_list = {}; 
    vocab = {}
    for col_name, special_tokens in col_name_dict.items():  
      int_list[col_name] = []      
      vocab[col_name] = collections.defaultdict(lambda: len(vocab[col_name]))
      for special_token in special_tokens:
        # Simple access to undefined dict key creates new ID (dict length)
        vocab[col_name][special_token]       
    for col_name in extra_cols_dict.keys() :
      int_list[col_name] = []
    for s in self.readConllu():
      # IMPORTANT : only works if "col_name" is the same as in lambda function definition!
      for col_name in col_name_dict.keys():
        int_list[col_name].append([vocab[col_name][tok[col_name]] for tok in s]) 
      for col_name, col_fct in extra_cols_dict.items():
        int_list[col_name].append(list(map(col_fct, [tok[col_name] for tok in s])))
    # vocabs cannot be saved if they have lambda function: erase default_factory
    for col_name in col_name_dict.keys():
      vocab[col_name].default_factory = None    
    return int_list, vocab
     
  ###############################

  def to_int_from_vocab(self, col_names, unk_token, vocab={}, extra_cols_dict={}):  
    int_list = {}
    unk_toks = {}
    for col_name in col_names:  
      int_list[col_name] = []
      unk_toks[col_name] = vocab[col_name].get(unk_token,None)
    for col_name in extra_cols_dict.keys() :
      int_list[col_name] = []
    for s in self.readConllu():
      for col_name in col_names:
        id_getter = lambda v,t: v[col_name].get(t[col_name],unk_toks[col_name])
        int_list[col_name].append([id_getter(vocab,tok) for tok in s])   
      for col_name, col_fct in extra_cols_dict.items():
        int_list[col_name].append(list(map(col_fct, [tok[col_name] for tok in s])))
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
    """Given a `sent` represented as a `conllu.TokenList`, returns a list of str
    containing the BIO encoding of the column corresponding to `name_tag`. By
    default, it is the "parseme:ne" column, which uses ConLLU-plus (tokens 
    belonging to the same NE get the same int + first gets ":category" suffix). 
    The output has category appended to 'B' and 'I' tags. The `bio_style` can
    be 'bio' or 'io', the latter has only 'I-category' tags, no 'B's.
    """
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
    """Convert BIO-encoded annotations into Sequoia/parseme format.
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

########################################################################
# PARSING FUNCTIONS 
########################################################################

class TransBasedSent(object): 
  """ 
  Useful functions to build a syntactic transition-based dependency parser.
  Takes as constructor argument a sentence as retrieved by readConllu() above.
  Generates oracle configurations, verifies action validity, etc.
  """
  ###############################

  def __init__(self, sent, actions_only=False):
    """
    `sent`: A `TokenList` as retrieved by the `conllu` library or `readConllu()`
    `actions_only`: affects the way the __str__ function prints this object
    """
    self.sent = sent
    self.actions_only = actions_only

  ###############################

  def __str__(self):
    """
    Sequence of configs and arc-hybrid actions corresponding to the sentence.
    If `self.actions_only=True` prints only sequence of actions
    """
    result = []
    for config, action in self.get_configs_oracle():      
      if not self.actions_only :
        result.append("{} -> {}".format(str(config), action))
      else :
        result.append(action)
    if not self.actions_only :
      result.append("{} -> {}".format(str(config), action))
      return "\n".join(result) 
    else :
      return " ".join(result)
    
    
  ###############################

  def get_configs_oracle(self):
    """
    Generator of oracle arc-hybrid configurations based on gold parsing tree.
    Yields pairs (`TransBasedConfig`, action) where action is a string among:
    - "SHIFT" -> pop buffer into stack
    - "LEFT-ARC-X" -> relation "X" from buffer head to stack head, pop stack
    - "RIGHT-ARC-X" -> relation "X" from stack head to stack second, pop stack
    Notice that RIGHT-ARC is only predicted when all its dependants are attached
    """
    config = TransBasedConfig(self.sent) # initial config
    gold_tree = [(i+1, tok['head']) for (i,tok) in enumerate(self.sent)]
    while not config.is_final():
      action = config.get_action_oracle(gold_tree)        # get next oracle act.
      yield (config, action)                              # yield to caller
      rel = config.apply_action(action, add_deprel=False) # get rel (if any)
      if rel :                                            # remove from gold        
        gold_tree.remove(rel)
      
  ###############################

  def update_sent(self, rels):
    """
    Updates the sentence by removing all syntactic relations and replacing them
    by those encoded as triples in `rels`.  `rels` is a list of syntactic 
    relations of the form (dep, head, label), that is, dep <---label--- head. 
    The function updates words at position (dep-1) by setting its "head"=`head` 
    and "deprel"=`label`
    """
    for tok in self.sent : # Remove existing info to avoid any error in eval
      tok['head']='_'
      tok['deprel']='_'
    for rel in rels :
      (dep, head, label) = rel
      self.sent[dep-1]['head'] = head
      self.sent[dep-1]['deprel'] = label      
      
################################################################################
################################################################################

class TransBasedConfig(object): 
  """ 
  Configuration of a transition-based parser composed of a `TokenList` sentence,
  a stack and a buffer. Both `stack` and `buff` are lists of indices within the
  sentence. Both `stack` and `buff` contain 1-initial indices, so remember to 
  subtract 1 to access `sent`. The class allows converting to/from dependency
  relations to actions.
  """
  
  ###############################  

  def __init__(self, sent): # Initial configuration for a sentence
    """
    Initial stack is an empty list.
    Initial buffer contains all sentence position indices 1..len(sent)    
    Appends 0 (representing root) to last buffer position.
    """
    self.sent = sent
    self.stack = []
    self.buff = [i+1 for (i,w) in enumerate(self.sent)] + [0]
  
  ###############################
  
  def __str__(self):
    """
    Generate a string with explicit buffer and stack words.
    """
    return "{}, {}".format([self.sent[i - 1]['form'] for i in self.stack],
                           [self.sent[i - 1]['form'] for i in self.buff[:-1]] + [0])
    
  ###############################
  
  def is_final(self):
    """
    Returns True if configuration is final, False else.
    A configuration is final if the stack is empty and the buffer contains only
    the root node.
    """
    return len(self.buff) == 1 and len(self.stack) == 0
  
  ###############################
  
  def apply_action(self, next_act, add_deprel=True):
    """
    Updates the configuration's buffer and stack by applying `next_act` action.
    `next_act` is a string among "SHIFT", "RIGHT-ARC-X" or "LEFT-ARC-X" where
    "X" is the name of any valid syntactic relation label (deprel).
    Returns a new syntactic relation added by the action, or None for "SHIFT"        
    Returned relation is a triple (mod, head, deprel) with modifier, head, and 
    deprel label if `add_deprel=True` (default), or a pair (mod, head) if 
    `add_deprel=False`.
    """    
    if next_act == "SHIFT":
      self.stack.append(self.buff.pop(0))
      return None
    else :
      deprel = next_act.split("-")[-1]
      if next_act.startswith("LEFT-ARC-"):
        rel = (self.stack[-1], self.buff[0])      
      else: # RIGHT-ARC-
        rel = (self.stack[-1], self.stack[-2])
      if add_deprel :
        rel = rel + (deprel,)
      self.stack.pop()
      return rel
  
  ###############################
 
  def get_action_oracle(self, gold_tree):       
    """
    Returns a string with the name of the next action to perform given the 
    current config and the gold parsing tree. The gold tree is a list of tuples
    [(mod1, head1), (mod2, head2) ...] with modifier-head pairs in this order.
    """
    if self.stack :
      deprel = self.sent[self.stack[-1] - 1]['deprel']
    if len(self.stack) >= 2 and \
       (self.stack[-1], self.stack[-2]) in gold_tree and \
       self.stack[-1] not in list(map(lambda x:x[1], gold_tree)): # head complete
      return "RIGHT-ARC-" + deprel
    elif len(self.stack) >= 1 and (self.stack[-1], self.buff[0]) in gold_tree:
      return "LEFT-ARC-" + deprel        
    else:        
      return "SHIFT"       
    
  ###############################
  
  def is_valid_act(self, act_cand):
    """
    Given a next-action candidate `act_cand`, returns True if the action is
    valid in the given `stack` and `buff` configuration, and False if the action
    cannot be applied to the current configuration. Constraints taken from
    page 2 of [de Lhoneux et al. (2017)](https://aclanthology.org/W17-6314/)
    """
    return (act_cand == "SHIFT" and len(self.buff)>1) or \
           (act_cand.startswith("RIGHT-ARC-") and len(self.stack)>1) or \
           (act_cand.startswith("LEFT-ARC-") and len(self.stack)>0 and \
                               (len(self.buff)>1 or len(self.stack)==1))
    
