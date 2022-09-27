import os
import re
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

from tqdm import tqdm
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
            ast = self.parser.parse(bytes(text, "utf-8"))
            if ast.root_node.has_error:
                print("try with wrapped temp class")
                text = "public class TempClass {" + text + "}"
                ast = self.parser.parse(bytes(text, "utf-8"))
                if ast.root_node.has_error:
                    print(f"    parsing error, return None")
                    print(f"        {str(path)}")
                    return None
        return ast

    def dir2asts(self, dir: str, tranverse: bool = False) -> Dict[str, Dict[str, Tree]]:
        """Parse all files in the dir to asts dict.

        Args:
            dir (str): the path of the dir contains files to be parsed.
            tranverse (bool): if tranverse all the files in dir.

        Return:
            asts (Dict[str, Dict[str, Tree]]): asts with form Dict[dir_name, Dict[file_name, Tree]].
        """
        asts = dict()

        def all_path(path):
            for sub_path in path.iterdir():
                if sub_path.is_file():
                    yield sub_path
                else:
                    for sub_sub_path in all_path(sub_path):
                        yield sub_sub_path
            return "done"

        if tranverse:
            for path in tqdm(all_path(Path(dir)), desc="itering dirs and parsing files..."):
                ast = self.file2ast(str(path))
                file_name = str(path.name)
                dir_name = str(path.parent)
                try:
                    asts[dir_name][file_name] = ast
                except Exception:
                    asts[dir_name] = {file_name: ast}
        else:
            for path in tqdm(Path(dir).iterdir(), desc="parsing files..."):
                if path.is_file():
                    ast = self.file2ast(str(path))
                    file_name = str(path.name)
                    dir_name = str(path.parent)
                    try:
                        asts[dir_name][file_name] = ast
                    except Exception:
                        asts[dir_name] = {file_name: ast}
        return asts

    @staticmethod
    def asts2sequences(asts: Dict[str, Dict[str, Tree]], lower: bool = False) -> Dict[str, List[str]]:
        """Turning parsed asts to token sequences, also known as 'flatterd, can be used for training woed2vec.

        Args:
            asts (Dict[str, Dict[str, Tree]]): Parsed asts.
            lower (bool, optional): If lower the tokens. Defaults to False.

        Returns:
            Dict[str, List[str]]: Sequences with form Dict[file_path, List[tokens]].
        """

        def get_sequence(node, sequence):
            token, children = PyAstParser.get_token(node, lower=lower), node.children
            if token != "":
                sequence.append(token)
            for child in children:
                if "comment" in child.type:
                    continue
                get_sequence(child, sequence)

        sequences = {}
        for dir_name in tqdm(asts, desc="asts to sequences"):
            for fname, ast in asts[dir_name].items():
                seq = []
                get_sequence(ast.root_node, seq)
                fpath = os.path.join(dir_name, fname)
                sequences[fpath] = seq

        return sequences

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
            if len(node.children) == 0:  # leaf node
                if node.type == "string_literal":
                    token = "string_literal"
                else:
                    token = re.sub(r"\s", "", str(node.text, "utf-8"))
            else:
                token = node.type
        if lower:
            return token.lower()
        return token

    @staticmethod
    def get_child_with_type(node: Node, type: str, vague=False) -> Tuple[Node, int]:
        """Get the child and its index in all of the children with given type. Notice that the comment children are ignored.

        Args:
            node (tree_sitter.Node): Ast node.
            type (str): Expect type name. The pattern if vague is True.
            vague (bool): If vague mode. Default to False.

        Returns:
            child (tree_sitter.Node): Child of the node with given type. Return None if did not find.
            id (int): Child index. Return None if did not find.
        """
        id = 0
        if vague:
            for child in node.children:
                if "comment" in child.type:
                    continue
                else:
                    if re.search(type, child.type) is not None:
                        return child, id
                    id += 1
        else:
            for child in node.children:
                if "comment" in child.type:
                    continue
                else:
                    if child.type == type:
                        return child, id
                    id += 1
        return None, None

    @staticmethod
    def get_named_children(node: Node) -> List[Node]:
        """Get all named children of a ast node. Notice that the comment children are ignored.

        Args:
            node (tree_sitter.Node): Ast node.

        Returns:
            children (List[tree_sitter.Node]): Named children list of the input node.
        """
        named_children = []
        for child in node.children:
            if child.is_named and "comment" not in child.type:
                named_children.append(child)
        return named_children

    @staticmethod
    def distinguish_for(node: Node) -> str:
        """
        As for_statement has different types base on the presence of init, condition and update statements, this function distinguish the type of a for_statement.

        Args:
            node (tree_sitter.Node): input node, must be for_statement.

        Returns:
            type (str): type of the for_statement node, one of {"","i","ic","iu","cu","icu"}, i, c, u are abbrevations of init, condition and update respectively.
        """
        assert node.type == "for_statement"
        res = ""
        for child in node.children:
            if child.type == "(":
                if child.next_sibling.type != ";":
                    res += "i"
                if child.next_sibling.next_sibling.type != ";":
                    res += "c"
            elif child.type == ")":
                if child.prev_sibling.type != ";":
                    res += "u"
        return res

    @staticmethod
    def distinguish_if(node: Node) -> str:
        """
        As if_statement has different types base on the presence of 'else if' and 'else', this function distinguish the type of a if_statement.

        Args:
            node (tree_sitter.Node): input node, must be if_statement.

        Returns:
            type (str): type of the for_statement node, one of {"if","if_elif","if_else"}.
        """
        assert node.type == "if_statement"
        for child in node.children:
            if child.type == "else":
                if child.next_sibling.type == "if_statement":
                    return "if_elif"
                else:
                    return "if_else"
        return "if"

    @staticmethod
    def asts2token_vocab(asts: Dict[str, Dict[str, Tree]], lower: bool = False, statastic: bool = False) -> Dict[str, int]:
        """Transform asts dict to a ast token vocabulary, ignore comments.

        Args:
            asts (Dict[str, Dict[str, Tree]]): dict with form Dict[dir_name, Dict[file_name, Tree]].
            lower (bool): If token lower.
            statastic (bool): If print the statastic information. Defaults to False.

        Return:
            token_vocab (Dict[str, int]): dict where keys are ast tokens, values are ids.
        """

        def get_sequence(node, sequence):
            token = PyAstParser.get_token(node, lower=lower)
            children = PyAstParser.get_named_children(node)
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
        for dir_name in tqdm(asts, desc="Get token sequence"):
            for file_name, ast in asts[dir_name].items():
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
    def ast2any_tree(tree: Tree, lower: bool = False):
        """Turn ast to anytree.  Require package 'anytree'.

        Args:
            tree (tree_sitter.Tree): The root node of the giving ast tree.
            lower (bool): If token lower. Default to False.

        Returns:
            newtree (AnyNode): The root node of the generated anytree.
        """
        from anytree import AnyNode

        global id
        id = 0

        def create_tree(node, parent):
            children = PyAstParser.get_named_children(node)
            token = PyAstParser.get_token(node, lower=lower)
            global id
            if id > 0:
                newnode = AnyNode(id=id, token=token, data=node, parent=parent)
            else:
                newnode = parent
            id += 1
            for child in children:
                # if parent.id == 0:
                #     create_tree(child, id, parent=parent)
                # else:
                create_tree(child, parent=newnode)

        root_node = tree.root_node
        new_tree = AnyNode(id=id, token=PyAstParser.get_token(root_node, lower=lower), data=root_node)
        create_tree(root_node, new_tree)
        return new_tree

    @staticmethod
    def trees2DGLgraphs(asts: Dict[str, Dict[str, Tree]], token_vocab: Dict[str, int]):
        """Turn asts to DGLgraphs. Require package 'dgl'.

        Args:
            asts (Dict[str, Dict[str, Tree]]): The input ast dict with form Dict[dir_name, Dict[file_name, Tree]].
            token_vocab (Dict[str, int]): The input token dict where key is the token in ast, value is the token id.

        Returns:
            Dict[str, Dict[str, info_dict]]: The Graph dict with form Dict[dir_name, Dict[file_name, info_dict]],
                    info_dict is {"n_layers": int, "graph": dgl.DGLgraph, "node_types": List[str]}.
        """
        import torch
        from dgl import add_self_loop, graph

        def gen_basic_graph(u, v, feats, node_types, node, vocab_dict):
            feat = torch.LongTensor([vocab_dict[node.token]])
            feats.append(feat)
            node_types.append(node.data.type)
            for child in node.children:
                v.append(node.id)
                u.append(child.id)
                gen_basic_graph(u, v, feats, node_types, child, vocab_dict)

        print("Turn ast trees into DGLgraphs...")
        graphs = {}
        for dir_name in tqdm(asts, desc="transform trees to DGLgraphs..."):
            for f_name, ast in asts[dir_name].items():
                new_tree = PyAstParser.ast2any_tree(ast)
                n_layers = new_tree.height
                u, v, feats, node_types = [], [], [], []
                gen_basic_graph(u, v, feats, node_types, new_tree, token_vocab)
                g = graph((u, v))
                g.ndata["token_id"] = torch.stack(feats)
                g = add_self_loop(g)
                try:
                    graphs[dir_name][f_name] = {"n_layers": n_layers, "graph": g, "node_types": node_types}
                except Exception:
                    graphs[dir_name] = {f_name: {"n_layers": n_layers, "graph": g, "node_types": node_types}}
        print("finished!")

        return graphs

    @staticmethod
    def trees2graphs(
        asts: Dict[str, Dict[str, Tree]],
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
    ):
        """Turn asts to graphs. Require package 'nextworkx' and 'anytree'.

        Args:
            asts (Dict[str, Dict[str, Tree]]): The input ast dict with form Dict[dir_name, Dict[file_name, Tree]].
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
            Dict[str, Dict[str, networkx.DiGraph]]: The Graph dict with form Dict[dir_name, Dict[file_name, DiGraph]].
        """
        from networkx import DiGraph

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
            token = node.token
            if while_edge:
                if token == "while_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["While"])
                    graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["While"])
            if for_edge:
                if token == "for_statement":
                    for_type = PyAstParser.distinguish_for(node.data)
                    if for_type == "":
                        pass
                    elif for_type == "i":
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[1].id, type=edges_type_idx["For"])
                    elif for_type in {"c", "u"}:
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["For"])
                    elif for_type == "cu":
                        graph.add_edge(node.children[0].id, node.children[2].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[0].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[2].id, node.children[1].id, type=edges_type_idx["For"])
                    elif for_type in {"ic", "iu"}:
                        graph.add_edge(node.children[0].id, node.children[1].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[1].id, node.children[2].id, type=edges_type_idx["For"])
                        graph.add_edge(node.children[2].id, node.children[1].id, type=edges_type_idx["For"])
                    else:  # "icu"
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
        for dir_name in tqdm(asts, desc="transform trees to graphs..."):
            for f_name, ast in asts[dir_name].items():
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
                try:
                    graph_dict[dir_name][f_name] = DiG
                except Exception:
                    graph_dict[dir_name] = {f_name: DiG}
        return graph_dict
