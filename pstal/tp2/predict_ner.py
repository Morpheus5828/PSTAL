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
    # Load parameters using pickle
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)

    print(f"Parameters loaded from {model_path}")

    log_E = parameters['log_E']
    log_T = parameters['log_T']
    log_pi = parameters['log_pi']

    return parameters, log_E, log_T, log_pi


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Viterbi algorithm for NER")
    parser.add_argument("pre_trained_model", help="Path to pre-trained model")
    parser.add_argument("inputFileName", help="Path for .dev corpus")
    parser.add_argument("predictFileName", help="Path for .conllu output file")

    args = parser.parse_args()
    model = args.pre_trained_model
    corpus = args.inputFileName
    output_file = args.predictFileName

    _, log_E, log_T, log_pi = load_model(model)

    with open(corpus, "r", encoding="UTF-8") as c:
        tokenLists = CoNLLUReader(c).readConllu()
        with open(output_file, "w", encoding="UTF-8") as out:
            for tokenList in tokenLists:
                sentence_form = [token['form'] for token in tokenList]
                tags = CoNLLUReader.to_bio(tokenList, bio_style='bio', name_tag='parseme:ne')
                assigned_tags = list(set(tags))

                n = len(tags)  # Sentence's length
                N = len(assigned_tags)  # Number of tag types

                # Initialize matrix delta(t_j, w_k) \in R^{N x n}
                delta = np.full((N, n), Util.PSEUDO_INF)
                psi = np.zeros((N, n), dtype=int)  # Back pointer matrix

                # Initialize the first column of delta
                for i in range(N):
                    delta[i, 0] = log_pi.get(assigned_tags[i], log_pi.get("OOV")) + \
                                  log_E[assigned_tags[i]].get(sentence_form[0], log_E[assigned_tags[i]].get("OOV"))

                # Recursive step
                for k in range(1, n):
                    for j in range(N):
                        min_prob = Util.PSEUDO_INF
                        best_state = 0

                        for i in range(N):
                            prob = delta[i, k - 1] + \
                                   log_T[assigned_tags[i]].get(assigned_tags[j], log_T[assigned_tags[i]].get("OOV"))
                            if prob < min_prob:
                                min_prob = prob
                                best_state = i
                        delta[j, k] = min_prob + log_E[assigned_tags[j]].get(sentence_form[k],
                                                                             log_E[assigned_tags[j]].get("OOV"))
                        psi[j, k] = best_state

                # Termination
                last_state = np.argmin(delta[:, n - 1])  # Index of the minimal log-probability in the last column

                # Backtracking
                best_path = [0] * n
                best_path[-1] = last_state
                for k in range(n - 2, -1, -1):
                    best_path[k] = psi[best_path[k + 1], k + 1]

                # Convert indices to tags
                best_tag_sequence = [assigned_tags[state] for state in best_path]

                # Convert BIO tags to Parseme format
                parseme_tags = CoNLLUReader.from_bio(best_tag_sequence)

                # Assign the predictions back to the tokenList
                for token, prediction in zip(tokenList, parseme_tags):
                    token['parseme:ne'] = prediction

                out.write(tokenList.serialize())

    print(f"Predictions written to {output_file}")
