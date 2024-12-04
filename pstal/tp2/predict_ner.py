import os
import pickle
import argparse
import sys

import numpy as np

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader, Util


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

    for i in range(N):
        delta[i, 0] = log_pi.get(assigned_tags[i], log_pi.get("OOV")) + \
                      log_E[assigned_tags[i]].get(sentence_form[0], log_E[assigned_tags[i]].get("OOV"))

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
                prob = delta[i, k - 1] + log_T[assigned_tags[i]].get(assigned_tags[j], log_T[assigned_tags[i]].get("OOV"))
                if prob < min_prob:
                    min_prob = prob
                    best_state = i
            delta[j, k] = min_prob + log_E[assigned_tags[j]].get(sentence_form[k],
                                                                 log_E[assigned_tags[j]].get("OOV"))
            psi[j, k] = best_state

    # Backtrack to find the best tag sequence
    best_path = backtrack(delta, psi, n, N, assigned_tags)
    return best_path


def backtrack(delta, psi, n, N, assigned_tags):
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


if __name__ == "__main__":
    """Main function to execute the Viterbi prediction."""
    parser = argparse.ArgumentParser(description="Viterbi algorithm for NER")
    parser.add_argument("pre_trained_model", help="Path to pre-trained model")
    parser.add_argument("inputFileName", help="Path for .dev corpus")
    parser.add_argument("predictFileName", help="Path for .conllu output file")
    args = parser.parse_args()

    log_E, log_T, log_pi = load_model(args.pre_trained_model)
    process_and_write_predictions(args.inputFileName, args.predictFileName, log_E, log_T, log_pi)
    print(f"Predictions written to {args.predictFileName}")
