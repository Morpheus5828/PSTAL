import os
import pickle
import argparse
import sys

import numpy as np
import torch
from torch import nn

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader, Util
from pstal.tp2.NER_RNN import NER_RNN, pad_tensor, extract_ner


def load_model(model_path: str):
    """Load model parameters from a file."""
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {model_path}")
    return parameters['log_E'], parameters['log_T'], parameters['log_pi']


def load_rnn_model(model_path):
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {model_path}")
    rnn_model = parameters['model']
    rnn_model.eval()
    train_vocab = parameters['vocabulary']
    train_tags = parameters['tags']
    history = parameters['history']
    return rnn_model, train_vocab, train_tags, history


def compute_emission_rnn(word, rnn_model, vocabs):
    """
    Compute the emission probabilities using the RNN model.

    Parameters:
    - word (str): The word to compute emissions for.
    - rnn_model (NER_RNN): The trained RNN model for tag prediction.
    - vocabs (dict): A mapping of words to their corresponding IDs, including an "UNK_ID" for unknown words.

    Returns:
    - log_probs (torch.Tensor): A tensor of log probabilities for each tag.
    """
    # Get the word ID from the vocabulary, default to "UNK_ID" for unknown words
    word_id = vocabs.get(word, vocabs.get("UNK_ID"))
    input_tensor = torch.tensor([word_id], dtype=torch.long)  # Convert to tensor

    # Pass the input through the RNN model
    with torch.no_grad():
        logits = rnn_model(input_tensor.unsqueeze(0))  # Add batch dimension
        softmax_probs = torch.softmax(logits, dim=-1)  # Compute probabilities
        log_probs = -torch.log(softmax_probs)  # Compute negative log probabilities
    return log_probs.squeeze(0)  # Remove batch dimension


def initialize_delta_and_psi(n, N, log_pi, log_E, sentence_form, assigned_tags):
    """Initialize delta and psi matrices."""
    delta = np.full((N, n), Util.PSEUDO_INF)
    psi = np.zeros((N, n), dtype=int)

    for i in range(N):
        delta[i, 0] = log_pi.get(assigned_tags[i], log_pi.get("OOV")) + \
                      log_E[assigned_tags[i]].get(sentence_form[0], log_E[assigned_tags[i]].get("OOV"))

    return delta, psi


def initialize_delta_and_psi_by_RNN(n, N, log_pi, pretrained_rnn: NER_RNN, sentence_form, assigned_tags, vocabs, tags):
    """Initialize delta and psi matrices."""
    delta = np.full((N, n), Util.PSEUDO_INF)
    psi = np.zeros((N, n), dtype=int)

    log_emission_prob = compute_emission_rnn(sentence_form[0], pretrained_rnn, vocabs).squeeze()
    for i in range(N):
        delta[i, 0] = log_pi.get(assigned_tags[i], log_pi.get("OOV")) + log_emission_prob[tags[assigned_tags[i]]].item()
    return delta, psi


def viterbi(sentence_form, tags, log_E, log_T, log_pi):
    """Perform the Viterbi algorithm to determine the best tag sequence."""
    assigned_tags = list(set(tags))
    n = len(tags)  # Sentence length
    N = len(assigned_tags)  # Number of tag types

    delta, psi = initialize_delta_and_psi(n, N, log_pi, log_E, sentence_form, assigned_tags)

    # Recursive step
    for k in range(1, n):
        for j in range(N):
            min_prob = Util.PSEUDO_INF
            best_state = 0
            for i in range(N):
                prob = delta[i, k - 1] + log_T[assigned_tags[i]].get(assigned_tags[j],
                                                                     log_T[assigned_tags[i]].get("OOV"))
                if prob < min_prob:
                    min_prob = prob
                    best_state = i
            delta[j, k] = min_prob + log_E[assigned_tags[j]].get(sentence_form[k],
                                                                 log_E[assigned_tags[j]].get("OOV"))
            psi[j, k] = best_state

    # Backtrack to find the best tag sequence
    best_path = backtrack(delta, psi, n, assigned_tags)
    return best_path


def viterbi_by_RNN(sentence_form, tags, log_T, log_pi, rnn_model, vocabs, trained_tags):
    """
    Perform the Viterbi algorithm to determine the best tag sequence using RNN-based emissions.

    Parameters:
    - sentence_form (list of str): The input sentence as a list of words.
    - tags (list of str): The set of all possible tags.
    - log_T (dict): The transition probabilities in log form.
    - log_pi (dict): The initial probabilities in log form.
    - rnn_model (NER_RNN): The trained RNN model for computing emissions.
    - vocabs (dict): A mapping of words to their corresponding IDs, including an "UNK_ID" for unknown words.

    Returns:
    - best_path (list of str): The best sequence of tags for the sentence.
    """
    assigned_tags = list(set(tags))
    n = len(sentence_form)  # Sentence length
    N = len(assigned_tags)  # Number of tag types

    # Initialize delta and psi matrices
    delta, psi = initialize_delta_and_psi_by_RNN(n, N, log_pi=log_pi, pretrained_rnn=rnn_model,
                                                 sentence_form=sentence_form, assigned_tags=assigned_tags, vocabs=vocabs, tags=trained_tags)

    # Recursive step
    for k in range(1, n):
        for j in range(N):
            min_prob = Util.PSEUDO_INF
            best_state = 0
            for i in range(N):
                prob = delta[i, k - 1] + log_T[assigned_tags[i]].get(assigned_tags[j],
                                                                     log_T[assigned_tags[i]].get("OOV"))
                if prob < min_prob:
                    min_prob = prob
                    best_state = i
            log_emission_prob = compute_emission_rnn(sentence_form[k], rnn_model, vocabs).squeeze()

            delta[j, k] = min_prob + log_emission_prob[trained_tags[assigned_tags[j]]].item()
            psi[j, k] = best_state

    # Backtrack to find the best tag sequence
    best_path = backtrack(delta, psi, n, assigned_tags)
    return best_path


def backtrack(delta, psi, n, assigned_tags):
    """Backtrack to find the best tag sequence."""
    last_state = np.argmin(delta[:, n - 1])  # Index of the minimal log-probability in the last column
    best_path = [0] * n
    best_path[-1] = last_state
    for k in range(n - 2, -1, -1):
        best_path[k] = psi[best_path[k + 1], k + 1]

    return [assigned_tags[state] for state in best_path]


def process_and_write_predictions(corpus, output_file, log_E, log_T, log_pi):
    """Process the corpus, predict tags, and write results to an output file."""
    with open(corpus, "r", encoding="UTF-8") as c:
        tokenLists = CoNLLUReader(c).readConllu()
        with open(output_file, "w", encoding="UTF-8") as out:
            for tokenList in tokenLists:
                sentence_form = [token['form'] for token in tokenList]
                tags = CoNLLUReader.to_bio(tokenList, bio_style='bio', name_tag='parseme:ne')

                best_tag_sequence = viterbi(sentence_form, tags, log_E, log_T, log_pi)

                # Convert BIO tags to Parseme format and assign to tokens
                parseme_tags = CoNLLUReader.from_bio(best_tag_sequence)

                for token, prediction in zip(tokenList, parseme_tags):
                    token['parseme:ne'] = prediction

                # Write the serialized token list to the output file
                out.write(tokenList.serialize())


def process_and_write_predictions_by_RNN(corpus, output_file, log_T, log_pi, rnn_model, vocabs, trained_tag):
    with open(corpus, "r", encoding="UTF-8") as c:
        tokenLists = CoNLLUReader(c).readConllu()
        with open(output_file, "w", encoding="UTF-8") as out:
            for tokenList in tokenLists:
                sentence_form = [token['form'] for token in tokenList]
                tags = CoNLLUReader.to_bio(tokenList, bio_style='bio', name_tag='parseme:ne')
                print(sentence_form)
                print(tags)
                best_tag_sequence = viterbi_by_RNN(sentence_form, tags, log_T, log_pi, rnn_model, vocabs, trained_tag)
                print(best_tag_sequence)
                print()
                # Convert BIO tags to Parseme format and assign to tokens
                parseme_tags = CoNLLUReader.from_bio(best_tag_sequence, bio_style='bio')

                for token, prediction in zip(tokenList, parseme_tags):
                    token['parseme:ne'] = prediction

                # Write the serialized token list to the output file
                out.write(tokenList.serialize())


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
        _, log_T, log_pi = load_model(args.pre_trained_model)
        log_E, train_vocab, train_tags, history = load_rnn_model(RNN_model)

        process_and_write_predictions_by_RNN(args.inputFileName, args.predictFileName, log_T, log_pi, log_E, train_vocab, train_tags)

    print(f"Predictions written to {args.predictFileName}")
