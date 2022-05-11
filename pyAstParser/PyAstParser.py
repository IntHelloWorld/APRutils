import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import javalang
import javalang.ast
import javalang.tree
import javalang.util
from anytree import AnyNode, RenderTree
from javalang.ast import Node
from javalang.tree import CompilationUnit
from networkx import DiGraph


class PyAstParser:
    def __init__(self) -> None:
        pass

    def file2ast(self, file: str) -> CompilationUnit:
        """Parse single file to single ast.

        Args:
            file (str): the absolute path of parsed file.

        Return:
            ast (CompilationUnit): the ast of parsed file.

        """
        path = Path(file)
        assert path.is_file()
        with path.open(encoding="utf-8") as f:
            text = f.read()
            try:
                tokens = javalang.tokenizer.tokenize(text)
                ast = javalang.parser.parse(tokens)
            except Exception:
                text = "public class TempClass {" + text + "}"
                tokens = javalang.tokenizer.tokenize(text)
                ast = javalang.parser.parse(tokens)
        return ast

    def dir2asts(self, dir: str) -> Dict[str, CompilationUnit]:
        """Parse all files in the dir to asts dict.

        Args:
            dir (str): the absolute path of the dir contains files to be parsed.

        Return:
            asts (Dict[str, CompilationUnit]): dict where keys are file paths, values are asts.
        """
        asts = dict()
        for path in Path(dir).iterdir():
            if path.is_file():
                ast = self.file2ast(path)
                asts[str(path)] = ast
        return asts

    def get_token(self, node: Node) -> str:
        """Get token of an ast node.

        Args:
            node (ast node): Ast node.

        Raises:
            Exception: Get node token error!

        Returns:
            token (str): The token of the input node.
        """
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = "Modifier"
        elif isinstance(node, Node):
            token = node.__class__.__name__
        else:
            raise Exception("Get node token error!")
        return token

    def get_child(self, node: Node) -> List[Node]:
        """Get all children of a ast node.

        Args:
            node (ast node): Ast node.

        Returns:
            children (List[Node]): Children list of the input node.
        """
        # print(root)
        if isinstance(node, Node):
            children = node.children
        elif isinstance(node, set):
            children = list(node)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        # print(sub_item)
                        yield sub_item
                elif item:
                    # print(item)
                    yield item

        return list(expand(children))

    def asts2token_vocab(
        self, asts: Dict[str, CompilationUnit], print_statastic: bool = False
    ) -> Dict[str, int]:
        """Transform asts dict to a ast token vocabulary.

        Args:
            asts (Dict[str, CompilationUnit]): dict where key is the file path, value is the ast.
            print_statastic (bool): If print the statastic information. Defaults to False.

        Return:
            token_vocab (Dict[str, int]): dict where keys are ast tokens, values are ids.
        """

        def get_sequence(node, sequence):
            token, children = self.get_token(node), self.get_child(node)
            sequence.append(token)
            # print(len(sequence), token)
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
            get_sequence(ast, all_tokens)

        # Token statastic
        if print_statastic:
            count = token_statistic(all_tokens)
            print(f"Tokens quantity: {len(all_tokens)}")
            pprint(count)

        tokens = list(set(all_tokens))
        vocabsize = len(tokens)
        tokenids = range(vocabsize)
        token_vocab = dict(zip(tokens, tokenids))
        return token_vocab

    def ast2any_tree(self, node: Node) -> AnyNode:
        """Turn ast to anytree. Using anytree package.

        Args:
            node (Complaition): The root node of the giving ast tree.

        Returns:
            newtree (AnyNode): The root node of the generated anytree.
        """

        def create_tree(node, id, parent):
            token, children = self.get_token(node), self.get_child(node)
            id += 1
            if id > 1:
                newnode = AnyNode(id=id, token=token, data=node, parent=parent)
            for child in children:
                if id > 1:
                    create_tree(child, id, parent=newnode)
                else:
                    create_tree(child, id, parent=parent)

        new_tree = AnyNode(id=0, token=self.get_token(node), data=node)
        id = 0
        create_tree(node, id, new_tree)
        return new_tree

    def trees2graphs(
        self,
        asts: Dict[str, CompilationUnit],
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
        """Turn asts to graphs. Using package nextworkx and anytree.

        Args:
            asts (Dict[str, CompilationUnit]): The input ast dict where key is the file path, value is the parsed tree.
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
                graph.add_edge(
                    node.children[i].id,
                    node.children[i + 1].id,
                    type=edges_type_idx["NextSib"],
                )
                if bidirectional_edge:
                    graph.add_edge(
                        node.children[i + 1].id,
                        node.children[i].id,
                        type=edges_type_idx["Prevsib"],
                    )
            for child in node.children:
                gen_next_sib_edge(child, graph)

        def gen_next_stmt_edge(node, graph):
            token = node.token
            if token == "BlockStatement":
                for i in range(len(node.children) - 1):
                    graph.add_edge(
                        node.children[i].id,
                        node.children[i + 1].id,
                        type=edges_type_idx["Nextstmt"],
                    )
                    if bidirectional_edge:
                        graph.add_edge(
                            node.children[i + 1].id,
                            node.children[i].id,
                            type=edges_type_idx["Prevstmt"],
                        )
            for child in node.children:
                gen_next_stmt_edge(child, graph)

        def gen_next_token_edge(node, graph):
            def get_token_list(node, token_list):
                if len(node.children) == 0:
                    token_list.append(node.id)
                for child in node.children:
                    get_token_list(child, token_list)

            token_list = []
            get_token_list(node, token_list)
            for i in range(len(token_list) - 1):
                graph.add_edge(
                    token_list[i], token_list[i + 1], type=edges_type_idx["Nexttoken"]
                )
                if bidirectional_edge:
                    graph.add_edge(
                        token_list[i + 1],
                        token_list[i],
                        type=edges_type_idx["Prevtoken"],
                    )

        def gen_next_use_edge(node, graph):
            def get_vars(node, var_dict):
                token = node.token
                if token == "MemberReference":
                    for child in node.children:
                        if child.token == node.data.member:
                            variable = child.token
                            var_node = child
                    if not var_dict.__contains__(variable):
                        var_dict[variable] = [var_node.id]
                    else:
                        var_dict[variable].append(var_node.id)
                for child in node.children:
                    get_vars(child, var_dict)

            var_dict = {}
            get_vars(node, var_dict)
            for v in var_dict:
                for i in range(len(var_dict[v]) - 1):
                    graph.add_edge(
                        var_dict[v][i],
                        var_dict[v][i + 1],
                        type=edges_type_idx["Nextuse"],
                    )
                    if bidirectional_edge:
                        graph.add_edge(
                            var_dict[v][i + 1],
                            var_dict[v][i],
                            type=edges_type_idx["Prevuse"],
                        )

        def gen_control_flow_edge(node, graph):
            token = node.token
            if while_edge:
                if token == "WhileStatement":
                    graph.add_edge(
                        node.children[0].id,
                        node.children[1].id,
                        type=edges_type_idx["While"],
                    )
                    graph.add_edge(
                        node.children[1].id,
                        node.children[0].id,
                        type=edges_type_idx["While"],
                    )
            if for_edge:
                if token == "ForStatement":
                    graph.add_edge(
                        node.children[0].id,
                        node.children[1].id,
                        type=edges_type_idx["For"],
                    )
                    graph.add_edge(
                        node.children[1].id,
                        node.children[0].id,
                        type=edges_type_idx["For"],
                    )
            if if_edge:
                if token == "IfStatement":
                    graph.add_edge(
                        node.children[0].id,
                        node.children[1].id,
                        type=edges_type_idx["If"],
                    )
                    if bidirectional_edge:
                        graph.add_edge(
                            node.children[1].id,
                            node.children[0].id,
                            type=edges_type_idx["If"],
                        )
                    if len(node.children) == 3:
                        graph.add_edge(
                            node.children[0].id,
                            node.children[2].id,
                            type=edges_type_idx["Ifelse"],
                        )
                        if bidirectional_edge:
                            graph.add_edge(
                                node.children[2].id,
                                node.children[0].id,
                                type=edges_type_idx["Ifelse"],
                            )
            for child in node.children:
                gen_control_flow_edge(child, graph)

        # print mode
        if ast_only:
            print("Func trees2graphs Mode: astonly")
        else:
            print(
                f"Func trees2graphs Mode: astonly=False, nextsib={next_sib}, ifedge={if_edge}, whileedge={while_edge}, foredge={for_edge}, blockedge={block_edge}, nexttoken={next_token}, nextuse={next_use}"
            )

        graph_dict = {}
        for fpath, ast in asts.items():
            new_tree = self.ast2any_tree(ast)
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
    my_parser = PyAstParser()
    tree = my_parser.file2ast("./test.java")
    anytree = my_parser.ast2any_tree(tree)
    # print(RenderTree(anytree))
    token_vocab = my_parser.asts2token_vocab({"1": tree})
    graph = my_parser.trees2graphs({"1": tree}, token_vocab)
    print(graph["1"].nodes)
    vocab = my_parser.asts2token_vocab({"1": tree})
    print(1)
