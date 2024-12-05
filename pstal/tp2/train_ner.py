import argparse
import os, sys
import pickle
from collections import defaultdict, Counter

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader, Util


def train(file_path: str, alpha=0.1, count_display=True):
    with open(file_path, "r", encoding="UTF-8") as infile:
        tokenLists = CoNLLUReader(infile).readConllu()

        c_w_t = defaultdict(Counter)  # Word-Tag Counts
        c_t_t = defaultdict(Counter)  # Adjacent Tag Counts
        c_t = Counter()  # Tag Counts
        c_s_t = Counter()  # Start-of-Sentence Tag Counts

        for tokenList in tokenLists:
            forms = [token['form'] for token in tokenList]
            BIOs = CoNLLUReader.to_bio(tokenList, bio_style='bio', name_tag='parseme:ne')

            # Update start-of-sentence tag counts
            c_s_t[BIOs[0]] += 1

            # Update word-tag counts
            for form, bio in zip(forms, BIOs):
                c_w_t[form][bio] += 1

            # Update adjacent tag counts
            for tag1, tag2 in zip(BIOs, BIOs[1:]):
                c_t_t[tag1][tag2] += 1

            # Update tag counts
            c_t.update(BIOs)

        S = sum(c_s_t.values())  # Total number of sentence
        V_w = len(c_w_t.keys())
        V_t = len(c_t.keys())

        if count_display:
            # Display results
            print("Word-Tag Counts (c(wj, ti)):")
            print(dict(c_w_t))

            print("\nAdjacent Tag Counts (c(ti, tj)):")
            print(dict(c_t_t))

            print("\nTag Counts (c(ti)):")
            print(dict(c_t))

            print("\nStart-of-Sentence Tag Counts (c(<s>, ti)):")
            print(dict(c_s_t))

            print("\nTotal number of sentences:")
            print(S)

            print("\nSize of vocabulary:")
            print(V_w)

            print("\nNumber of labels:")
            print(V_t)

        # Compute -log probabilities
        log_E = defaultdict(dict)
        for tag in c_t.keys():
            log_E[tag]["OOV"] = Util.log_cap(c_t[tag] + V_w * alpha) - Util.log_cap(0 + alpha)

        log_T = defaultdict(dict)
        for tag in c_t.keys():
            log_T[tag]["OOV"] = Util.log_cap(c_t[tag] + V_t * alpha) - Util.log_cap(0 + alpha)

        log_pi = {"OOV": Util.log_cap(S + V_t * alpha) - Util.log_cap(0 + alpha)}

        # Compute −logE(ti, wj)
        for word, tag_counts in c_w_t.items():
            for tag, count in tag_counts.items():
                log_E[tag][word] = Util.log_cap(c_t[tag] + V_w * alpha) - Util.log_cap(count + alpha)

        # Compute −logT(ti, tj)
        for tag1, tag2_counts in c_t_t.items():
            for tag2, count in tag2_counts.items():
                log_T[tag1][tag2] = Util.log_cap(c_t[tag1] + V_t * alpha) - Util.log_cap(count + alpha)

        # Compute −logπ(ti)
        for tag, count in c_s_t.items():
            log_pi[tag] = Util.log_cap(S + V_t * alpha) - Util.log_cap(count + alpha)

        # Convert defaultdicts to dicts for serialization
        log_E = {word: dict(tags) for word, tags in log_E.items()}
        log_T = {tag1: dict(tag2_counts) for tag1, tag2_counts in log_T.items()}

        return log_E, log_T, log_pi


def save(output_path: str, log_E: dict, log_T: dict, log_pi: dict):
    # Save parameters using pickle
    parameters = {'log_E': log_E, 'log_T': log_T, 'log_pi': log_pi}

    with open(output_path, 'wb') as f:
        pickle.dump(parameters, f)

    print(f"Parameters saved to {output_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for NER")
    parser.add_argument("inputFileName", help="Path to the sequoia input file")
    parser.add_argument("outputFileName", help="Name and path for .pkl output file")

    # TO DO : using optional parameters alpha and countDisplay
    parser.add_argument("alpha", nargs='?', type=float, default=0.1, help="Smoothing rate")
    parser.add_argument("countDisplay", nargs='?', type=str, choices=["True", "False"], default="False", help="")

    args = parser.parse_args()

    # Get the input file name from the command line argument
    input_file_name = args.inputFileName
    output_path = args.outputFileName
    alpha = args.alpha
    display = args.countDisplay == "True"  # Convert to boolean value

    log_E, log_T, log_pi = train(input_file_name, alpha=alpha, count_display=True)

    """
    print(f"Log E: {log_E}\n")
    print(f"Log T: {log_T}\n")
    print(f"Log pi: {log_pi}\n")
    """
    #save(output_path, log_E, log_T, log_pi)
