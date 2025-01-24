import os
import pickle
import argparse
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn, relu

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader, Util
from pstal.tp2.NER_RNN import NER_RNN, NER_RNN_with_features


def load_model(model_path: str):
    """Load model parameters from a file."""
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {model_path}")
    return parameters['log_E'], parameters['log_T'], parameters['log_pi']


def initialize_delta_and_psi(n, N, log_pi, log_E, sentence_form, assigned_tags):
    """Initialize delta and psi matrices."""
    delta = np.full((N, n), Util.PSEUDO_INF)
    psi = np.zeros((N, n), dtype=int)

    for i, tag in enumerate(assigned_tags):
        delta[i, 0] = log_pi.get(tag, log_pi.get("OOV")) + \
                      log_E[tag].get(sentence_form[0], log_E[tag].get("OOV"))

    return delta, psi


def viterbi(sentence_form, tags, log_E, log_T, log_pi):
    """Perform the Viterbi algorithm to determine the best tag sequence."""
    assigned_tags = tags
    n = len(sentence_form)  # Sentence length
    N = len(assigned_tags)  # Number of tag types

    delta, psi = initialize_delta_and_psi(n, N, log_pi, log_E, sentence_form, assigned_tags)

    # Recursive step
    for k in range(1, n):
        for j, curr_tag in enumerate(assigned_tags):
            for i, prev_tag in enumerate(assigned_tags):
                prob = (delta[i, k - 1] + log_T[prev_tag].get(curr_tag, log_T[prev_tag].get("OOV"))
                        + log_E[curr_tag].get(sentence_form[k], log_E[curr_tag].get("OOV")))
                if prob < delta[j, k]:
                    delta[j, k] = prob
                    psi[j, k] = i

    # Backtrack to find the best tag sequence
    best_path = backtrack(delta, psi, n, assigned_tags)
    return best_path


def process_and_write_predictions(corpus, output_file, log_E, log_T, log_pi):
    """Process the corpus, predict tags, and write results to an output file."""
    assigned_tags = {}
    for idx, key in enumerate(log_T.keys()):
        if key == "PAD_ID":
            continue
        assigned_tags[key] = idx

    with open(corpus, "r", encoding="UTF-8") as c:
        tokenLists = CoNLLUReader(c).readConllu()
        with open(output_file, "w", encoding="UTF-8") as out:
            for tokenList in tokenLists:
                sentence_form = [token['form'] for token in tokenList]

                best_tag_sequence = viterbi(sentence_form, assigned_tags, log_E, log_T, log_pi)

                # Convert BIO tags to Parseme format and assign to tokens
                parseme_tags = CoNLLUReader.from_bio(best_tag_sequence)

                for token, prediction in zip(tokenList, parseme_tags):
                    token['parseme:ne'] = prediction

                # Write the serialized token list to the output file
                out.write(tokenList.serialize())


################################### PRE TRAINED RNN MODEL AS EMISSION MATRIX ###########################################

def load_rnn_model(pretrained_rnn_model_path):
    """"
        Load a pre-trained RNN model from a .pkl file
        This file contains :
            - the structure of the RNN model
            - the parameters of the RNN model (number of trained epochs, batch_size, saved training history)
            - the parameters of the model's optimizer
            - the loss function
            - the trained corpus and trained tags
    """
    with open(pretrained_rnn_model_path, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {pretrained_rnn_model_path}")
    return parameters


def extract_features(token, upos):
    if token:
        feature_vector = [
            int(token[0].isupper()),  # First letter uppercase
            int(token.isupper()),
            int(any(token.isdigit() for char in token)),  # Contains digits
            int("-" in token),  # Contains hyphen
            int(upos == 'NOUN'),
            int(upos == 'PROPN')
        ]
    else:
        # In case there's an empty token, treat it as all False features
        feature_vector = [0, 0, 0, 0, 0, 0]

    # Convert the feature list into a tensor of shape (max_len, 6)
    features_tensor = torch.tensor(feature_vector, dtype=torch.float)

    return features_tensor


def compute_emission_rnn(sentence, rnn_model, vocabs, upos):
    """
    Compute the emission probabilities for the entire sentence using the RNN model.

    Parameters:
    - sentence (list of str): The input sentence as a list of words.
    - rnn_model (NER_RNN): The trained RNN model for tag prediction.
    - vocabs (dict): A mapping of words to their corresponding IDs, including "UNK_ID" for unknown words.

    Returns:
    - log_probs (torch.Tensor): A tensor of log probabilities for each tag for the entire sentence.
    """
    # Map sentence words to IDs, defaulting to "UNK_ID" for unknown words
    word_ids = [vocabs.get(word, vocabs.get("UNK_ID")) for word in sentence]
    input_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0)  # Batch size = 1
    features = torch.stack([extract_features(word, upos[idx]) for idx, word in enumerate(sentence)])
    features = features.unsqueeze(0)  # Add batch dimension

    # Pass the input through the RNN model
    with torch.no_grad():
        logits = rnn_model(input_tensor, features)  # Shape: (1, sequence_length, num_tags)
        log_probs = torch.log_softmax(logits, dim=-1)  # Log probabilities for stability

    return -log_probs.squeeze(0)  # Shape: (sequence_length, num_tags)


def initialize_delta_and_psi_by_RNN(n, N, log_pi, pretrained_rnn, sentence_form, assigned_tags, vocabs, upos):
    """
    Initialize delta and psi matrices for the Viterbi algorithm.
    """
    delta = np.full((N, n), np.inf)  # Initialize with pseudo-infinity
    psi = np.zeros((N, n), dtype=int)  # Backpointer matrix

    # Compute log emission probabilities for the entire sentence
    log_emission_prob = compute_emission_rnn(sentence_form, pretrained_rnn, vocabs, upos)
    # print(log_emission_prob)
    # Initialize delta for the first word
    for i, tag in enumerate(assigned_tags):
        delta[i, 0] = log_pi.get(tag, 9999.0) + log_emission_prob[0][i].item()
    # print(delta)
    return delta, psi


def viterbi_by_RNN(sentence_form, tags, log_T, log_pi, rnn_model, vocabs, upos):
    """
    Perform the Viterbi algorithm with RNN-based emissions.
    """

    assigned_tags = tags

    n = len(sentence_form)  # Sentence length
    N = len(assigned_tags)  # Number of tag types

    # Initialize delta and psi matrices
    delta, psi = initialize_delta_and_psi_by_RNN(n, N, log_pi, rnn_model, sentence_form, assigned_tags, vocabs, upos)

    # Compute log emission probabilities for the entire sentence
    log_emission_prob = compute_emission_rnn(sentence_form, rnn_model, vocabs, upos)

    # print(log_emission_prob)
    # print(log_emission_prob.shape)

    # Recursive step
    for k in range(1, n):
        for j, curr_tag in enumerate(assigned_tags):
            for i, prev_tag in enumerate(assigned_tags):
                prob = (
                        delta[i, k - 1] +
                        log_T[prev_tag].get(curr_tag, 9999.0) +
                        log_emission_prob[k][j].item()
                )
                if prob < delta[j, k]:
                    delta[j, k] = prob
                    psi[j, k] = i

    # Backtrack to find the best tag sequence
    return backtrack(delta, psi, n, assigned_tags)


def process_and_write_predictions_by_RNN(corpus, output_file, log_T, log_pi, rnn_model, vocabs, trained_tags):
    """
    Process the corpus and write predictions to the output file.
    """

    with open(corpus, "r", encoding="UTF-8") as c:
        token_lists = CoNLLUReader(c).readConllu()
        with open(output_file, "w", encoding="UTF-8") as out:
            for token_list in token_lists:
                sentence_form = [token['form'] for token in token_list]
                bios = CoNLLUReader.to_bio(token_list, bio_style='bio', name_tag='parseme:ne')
                upos = [token['upos'] for token in token_list]
                best_tag_sequence = viterbi_by_RNN(sentence_form, trained_tags, log_T, log_pi, rnn_model, vocabs, upos)

                """
                print(sentence_form)
                print(bios)
                print(best_tag_sequence)
                print()
                """
                # Write predictions
                # Convert BIO tags to Parseme format and assign to tokens
                parseme_tags = CoNLLUReader.from_bio(best_tag_sequence)

                for token, prediction in zip(token_list, parseme_tags):
                    token['parseme:ne'] = prediction

                # Write the serialized token list to the output file
                out.write(token_list.serialize())


def backtrack(delta, psi, n, assigned_tags):
    """
    Backtrack to find the best tag sequence.
    """
    best_path = [0] * n
    best_path[-1] = np.argmin(delta[:, n - 1])  # Best final state
    for k in range(n - 2, -1, -1):
        best_path[k] = psi[best_path[k + 1], k + 1]
    return [tag for state in best_path for tag, idx in assigned_tags.items() if idx == state]


if __name__ == "__main__":
    """Main function to execute the Viterbi prediction."""
    parser = argparse.ArgumentParser(description="Viterbi algorithm for NER")
    parser.add_argument("pre_trained_model", help="Path to pre-trained model")
    parser.add_argument("inputFileName", help="Path for .dev/.test corpus")
    parser.add_argument("predictFileName", help="Path for .conllu output file")
    parser.add_argument("pre_trained_RNN_model", nargs='?', help="Path to pre-trained emission matrix by RNN")
    args = parser.parse_args()

    RNN_model = args.pre_trained_RNN_model
    if RNN_model is None:
        log_E, log_T, log_pi = load_model(args.pre_trained_model)
        process_and_write_predictions(args.inputFileName, args.predictFileName, log_E, log_T, log_pi)

    else:
        log_e, log_T, log_pi = load_model(args.pre_trained_model)
        pre_trained_rnn_params = load_rnn_model(RNN_model)

        print(f"Nb of trained epochs : {len(pre_trained_rnn_params['history']['train_loss'])}")
        for param in pre_trained_rnn_params.keys():
            if param not in ['history', 'vocabulary']:
                print(f"{param}: {pre_trained_rnn_params[param]}")

        pre_trained_rnn_model = pre_trained_rnn_params['model']
        pre_trained_rnn_model.eval()

        train_vocab = pre_trained_rnn_params['vocabulary']
        train_tags = pre_trained_rnn_params['tags']

        process_and_write_predictions_by_RNN(args.inputFileName, args.predictFileName, log_T, log_pi,
                                             pre_trained_rnn_model, train_vocab, train_tags)

    print(f"Predictions written to {args.predictFileName}")
