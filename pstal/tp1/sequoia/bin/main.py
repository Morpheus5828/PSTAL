
import os, sys
from conllu import parse_incr


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_small_path = os.path.join(project_path, "sequoia-ud.parseme.frsemcor.simple.small")

if __name__ == "__main__":
    for sent in parse_incr(open(sequoia_small_path, encoding="UTF-8")):
        print([tok for tok in sent])
        break