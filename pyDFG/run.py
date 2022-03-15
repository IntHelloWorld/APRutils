import logging
import os

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
from DFG import DFG_python, DFG_java
from utils import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, tree_to_variable_index
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)
dfg_function = {"python": DFG_python, "java": DFG_java}

# set work dir
curdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curdir)

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language("my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)  # A list contains the (start point, end point) of each token
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]  # A list contains the tokens
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


def read_example(filename):
    code = ""
    with open(filename, encoding="utf-8") as f:
        code = f.read()
    return code


if __name__ == "__main__":
    example = read_example("example.py")
    code_tokens, dfg = extract_dataflow(example, parsers["python"], "python")
