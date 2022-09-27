import os
import re
from pathlib import Path
from pprint import pprint
from typing import Dict, List

from anytree import AnyNode, RenderTree
from networkx import DiGraph
from tree_sitter import Language, Node, Parser, Tree


class PyAstParser:
    def __init__(self, so, language):
        """Init PyAstParser and build parser.

        Args:
            so (str): The .so file generated for building parser. See https://github.com/tree-sitter/py-tree-sitter.

            language (str): The target language of the parser.
        """
        self.parser = Parser()
        LANGUAGE = Language(so, language)
        self.parser.set_language(LANGUAGE)

    def file2ast(self, file: str) -> Tree:
        """Parse single file to single ast.

        Args:
            file (str): the absolute path of parsed file.

        Return:
            ast (tree_sitter.Tree): the ast of parsed file.

        """
        path = Path(file)
        assert path.is_file()
        with path.open(encoding="utf-8") as f:
            text = f.read()
            try:
                ast = self.parser.parse(bytes(text, "utf-8"))
            except Exception as e:
                print(e)
                print("parsing error, try with wrapped temp class")
                text = "public class TempClass {" + text + "}"
                try:
                    ast = self.parser.parse(bytes(text, "utf-8"))
                except Exception as e2:
                    print(e2)
                    print("parsing error, return None")
                    return None
        return ast

    def dir2asts(self, dir: str, tranverse: bool = False) -> Dict[str, Tree]:
        """Parse all files in the dir to asts dict.

        Args:
            dir (str): the path of the dir contains files to be parsed.
            tranverse (bool): if tranverse all the files in dir.

        Return:
            asts (Dict[str, Tree]): dict where keys are file paths, values are asts.
        """
        asts = dict()

        def all_path(path):
            for sub_path in path.iterdir():
                if sub_path.is_file():
                    yield sub_path
                else:
                    all_path(sub_path)
            return "done"

        if tranverse:
            for path in all_path(Path(dir)):
                ast = self.file2ast(path)
                asts[str(path)] = ast
        else:
            for path in Path(dir).iterdir():
                if path.is_file():
                    ast = self.file2ast(path)
                    asts[str(path)] = ast
        return asts

    @staticmethod
    def get_token(node: Node, lower: bool = False) -> str:
        """Get the token of an ast node, the token of a leaf node is its text in code, the token of a non-leaf node is its ast type.

        Args:
            node (tree_sitter.Node): Ast node.
            lower (bool): If output lower token. Default to False.

        Raises:
            Exception: Get node token error!

        Returns:
            token (str): The token of the input node.
        """
        if not node.is_named:
            token = ""
        else:
            if len(node.children) == 0:
                token = re.sub(r"\s", "", str(node.text, "utf-8"))
            else:
                token = node.type
        if lower:
            return token.lower()
        return token

    @staticmethod
    def get_child(node: Node) -> List[Node]:
        """Get all children of a ast node.

        Args:
            node (tree_sitter.Node): Ast node.

        Returns:
            children (List[Node]): Children list of the input node.
        """
        return node.children

    @staticmethod
    def asts2token_vocab(asts: Dict[str, Tree], statastic: bool = False) -> Dict[str, int]:
        """Transform asts dict to a ast token vocabulary.

        Args:
            asts (Dict[str, Tree]): dict where key is the file path, value is the ast.
            statastic (bool): If print the statastic information. Defaults to False.

        Return:
            token_vocab (Dict[str, int]): dict where keys are ast tokens, values are ids.
        """

        def get_sequence(node, sequence):
            token, children = PyAstParser.get_token(node), PyAstParser.get_child(node)
            if token != "":
                sequence.append(token)
            for child in children:
                get_sequence(child, sequence)

        def token_statistic(all_tokens):
            count = dict()
            for token in all_tokens:
                try:
                    count[token] += 1
                except Exception:
                    count[token] = 1
            return count

        all_tokens = []
        for ast in asts.values():
            get_sequence(ast.root_node, all_tokens)

        # Token statastic
        if statastic:
            count = token_statistic(all_tokens)
            print(f"Tokens quantity: {len(all_tokens)}")
            pprint(count)

        tokens = list(set(all_tokens))
        vocabsize = len(tokens)
        tokenids = range(vocabsize)
        token_vocab = dict(zip(tokens, tokenids))
        return token_vocab

    @staticmethod
    def ast2any_tree(tree: Tree) -> AnyNode:
        """Turn ast to anytree. Using anytree package.

        Args:
            tree (tree_sitter.Tree): The root node of the giving ast tree.

        Returns:
            newtree (AnyNode): The root node of the generated anytree.
        """

        def create_tree(node, id, parent):
            children = PyAstParser.get_child(node)
            token = PyAstParser.get_token(node)
            id += 1
            if id > 1 and token != "":
                newnode = AnyNode(id=id, token=token, data=node, parent=parent)
            for child in children:
                if id > 1:
                    create_tree(child, id, parent=newnode)
                else:
                    create_tree(child, id, parent=parent)

        root_node = tree.root_node
        new_tree = AnyNode(id=0, token=PyAstParser.get_token(root_node), data=root_node)
        id = 0
        create_tree(root_node, id, new_tree)
        return new_tree

    @staticmethod
    def trees2graphs(
        asts: Dict[str, Tree],
        token_vocab: Dict[str, int],
        bidirectional_edge: bool = True,
        ast_only: bool = True,
        next_sib: bool = False,
        block_edge: bool = False,
        next_token: bool = False,
        next_use: bool = False,
        if_edge: bool = False,
        while_edge: bool = False,
        for_edge: bool = False,
        edges_type_idx: Dict[str, int] = {
            "AstEdge": 0,
            "NextSib": 1,
            "Nexttoken": 2,
            "Prevtoken": 3,
            "Nextuse": 4,
            "Prevuse": 5,
            "If": 6,
            "Ifelse": 7,
            "While": 8,
            "For": 9,
            "Nextstmt": 10,
            "Prevstmt": 11,
            "Prevsib": 12,
        },
    ) -> Dict[str, DiGraph]:
        """Turn asts to graphs. Using package 'nextworkx' and 'anytree'.

        Args:
            asts (Dict[str, Tree]): The input ast dict where key is the file path, value is the parsed tree.
            token_vocab (Dict[str, int]): The input token dict where key is the token in ast, value is the token id.
            bidirectional_edge (bool, optional): If add bidirectional edge. Defaults to True.
            ast_only (bool, optional): If only build basic graph bases on origin ast. Defaults to True.
            next_sib (bool, optional): If add next sibling edge. Defaults to False.
            block_edge (bool, optional): If add next statement edge. Defaults to False.
            next_token (bool, optional): If add next token edge. Defaults to False.
            next_use (bool, optional): If add next use edge. Defaults to False.
            if_edge (bool, optional): If add IfStatement control flow edge. Defaults to False.
            while_edge (bool, optional): If add WhileStatement control flow edge. Defaults to False.
            for_edge (bool, optional): If add ForStatement control flow edge. Defaults to False.
            edges_type_idx (Dict[str, int]): The edges_type_idx of each kind of edge. Defaults to
                {
                    "AstEdge": 0,
                    "NextSib": 1,
                    "Nexttoken": 2,
                    "Prevtoken": 3,
                    "Nextuse": 4,
                    "Prevuse": 5,
                    "If": 6,
                    "Ifelse": 7,
                    "While": 8,
                    "For": 9,
                    "Nextstmt": 10,
                    "Prevstmt": 11,
                    "Prevsib": 12,
                }

        Returns:
            graph dict (Dict[str, DiGraph]): The Graph dict where key is the file path, value is the directed graph transform from the ast tree.
        """

        def gen_basic_graph(node, vocab_dict, graph):
            token = node.token
            graph.add_node(vocab_dict[token])
            for child in node.children:
                graph.add_edge(node.id, child.id)
                if bidirectional_edge:
                    graph.add_edge(child.id, node.id)
                if not ast_only:
                    graph[node.id][child.id]["type"] = edges_type_idx["AstEdge"]
                    graph[child.id][node.id]["type"] = edges_type_idx["AstEdge"]
                gen_basic_graph(child, vocab_dict, graph)

        def gen_next_sib_edge(node, graph):
            for i in range(len(node.children) - 1):
                graph.add_edge(node.children[i].id, node.children[i + 1].id, type=edges_type_idx["NextSib"])
                if bidirectional_edge:
                    graph.add_edge(node.children[i + 1].id, node.children[i].id, type=edges_type_idx["Prevsib"])
            for child in node.children:
                gen_next_sib_edge(child, graph)

        def gen_next_stmt_edge(node, graph):
            token = node.token
            if token == "block":
                for i in range(len(node.children) - 1):
                    graph.add_edge(node.children[i].id, node.children[i + 1].id, type=edges_type_idx["Nextstmt"])
                    if bidirectional_edge:
                        graph.add_edge(node.children[i + 1].id, node.children[i].id, type=edges_type_idx["Prevstmt"])
            for child in node.children:
                gen_next_stmt_edge(child, graph)

        def gen_next_token_edge(node, graph):
            def get_leaf_node_list(node, token_list):
                if len(node.children) == 0:
                    token_list.append(node.id)
                for child in node.children:
                    get_leaf_node_list(child, token_list)

            token_list = []
            get_leaf_node_list(node, token_list)
            for i in range(len(token_list) - 1):
                graph.add_edge(token_list[i], token_list[i + 1], type=edges_type_idx["Nexttoken"])
                if bidirectional_edge:
                    graph.add_edge(token_list[i + 1], token_list[i], type=edges_type_idx["Prevtoken"])

        def gen_next_use_edge(node, graph):
            def get_vars(node, var_dict):
                if node.data.type == "identifier":
                    var = str(node.data.text)
                    if not var_dict.__contains__(var):
                        var_dict[var] = [node.id]
                    else:
                        var_dict[var].append(node.id)
                for child in node.children:
                    get_vars(child, var_dict)

            var_dict = {}
            get_vars(node, var_dict)
            for v in var_dict:
                for i in range(len(var_dict[v]) - 1):
                    graph.add_edge(var_dict[v][i], var_dict[v][i + 1], type=edges_type_idx["Nextuse"])
                    if bidirectional_edge:
                        graph.add_edge(var_dict[v][i + 1], var_dict[v][i], type=edges_type_idx["Prevuse"])

        def gen_control_flow_edge(node, graph):
            def distinguish_for(node):
                """
                As for_statement has different types base on the presence of init, condition and update statements, this function distinguish the type of a for_statement according to its named children amount and semicolon children amount (under tree_sitter package).
                """
                semicolon = 0
                named_children = 0
                for c in node.children:
                    if c.type == ";":
                        semicolon += 1
                    if c.is_named:
                        named_children += 1
                if named_children == 1:
                    return "only_block"
                elif named_children == 2:
                    if semicolon == 1:
                        return "init_block"
                    else:
                        return "con/up_block"
                elif named_children == 3:
                    if semicolon == 1:
                        return "con_up_block"
                    else:
                        return "init_con/up_block"
                else:
                    return "init_con_up_block"

            token = node.token
            if while_edge:
                if token == "while_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["While"])
                    graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["While"])
            if for_edge:
                if token == "for_statement":
                    for_type = distinguish_for(node.data)
                    if for_type == "only_block":
                        pass
                    elif for_type == "init_block":
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[1].id, type=edges_type_idx["For"])
                    elif for_type == "con/up_block":
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["For"])
                    elif for_type == "con_up_block":
                        graph.add_edge(node.children[0].id, node.children[2].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[2].id, node.children[1].id, type=edges_type_idx["For"])
                    elif for_type == "init_con/up_block":
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[2].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[2].id, node.children[1].id, type=edges_type_idx["For"])
                    else:
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[3].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[2].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[3].id, node.children[2].id, type=edges_type_idx["For"])
                if token == "enhanced_for_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                    graph.add_edge(node.children[1].id, node.children[2].id, type=edges_type_idx["For"])
                    graph.add_edge(node.children[1].id, node.children[3].id, type=edges_type_idx["For"])
                    graph.add_edge(node.children[3].id, node.children[1].id, type=edges_type_idx["For"])
            if if_edge:
                if token == "if_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["If"])
                    if bidirectional_edge:
                        graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["If"])
                    if len(node.children) == 3:  # has else statement
                        graph.add_edge(node.children[0].id, node.children[2].id, type=edges_type_idx["Ifelse"])
                        if bidirectional_edge:
                            graph.add_edge(node.children[2].id, node.children[0].id, type=edges_type_idx["Ifelse"])
            for child in node.children:
                gen_control_flow_edge(child, graph)

        # print mode
        if ast_only:
            print("trees2graphs Mode: astonly")
        else:
            print(
                f"trees2graphs Mode: astonly=False, nextsib={next_sib}, ifedge={if_edge}, whileedge={while_edge}, foredge={for_edge}, blockedge={block_edge}, nexttoken={next_token}, nextuse={next_use}"
            )

        graph_dict = {}
        for fpath, ast in asts.items():
            new_tree = PyAstParser.ast2any_tree(ast)
            DiG = DiGraph()
            gen_basic_graph(new_tree, token_vocab, DiG)
            if next_sib:
                gen_next_sib_edge(new_tree, DiG)
            if block_edge:
                gen_next_stmt_edge(new_tree, DiG)
            if next_token:
                gen_next_token_edge(new_tree, DiG)
            if next_use:
                gen_next_use_edge(new_tree, DiG)
            gen_control_flow_edge(new_tree, DiG)
            graph_dict[fpath] = DiG
        return graph_dict


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    my_parser = PyAstParser("/home/qyh/Desktop/github/GitHub/MTN-cue/tree-sitter/python-java-c-languages.so", "java")
    # my_parser = PyAstParser()
    tree = my_parser.file2ast("./test.java")
    anytree = my_parser.ast2any_tree(tree)
    print(RenderTree(anytree))
    token_vocab = my_parser.asts2token_vocab({"1": tree})
    graph = my_parser.trees2graphs({"1": tree}, token_vocab)
    print(graph["1"].nodes)
    vocab = my_parser.asts2token_vocab({"1": tree})
    print(1)
