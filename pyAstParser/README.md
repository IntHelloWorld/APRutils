# PyAstParser

A tool write in python for parsing java code snippet to ast (tree_sitter.Tree) or other data structure (e.g. anytree, graph).

Function:

1. file2ast(self, file: str) -> tree_sitter.Tree:

   Parse single file to single ast. Using tree_sitter package.

2. dir2asts(self, dir: str) -> Dict[str, tree_sitter.Tree]:

   Parse all files in the dir to asts dict.

3. get_token(self, node: tree_sitter.Node) -> str:

   Get token of an ast node.

4. get_child(self, node: tree_sitter.Node) -> List[tree_sitter.Node]:

   Get all children of a ast node.

5. asts2token_vocab(self, asts: Dict[str, tree_sitter.Tree], print_statastic: bool = False) -> Dict[str, int]:

   Transform asts dict to a ast token vocabulary. Print token vocabulary statastic information (optional).

6. ast2any_tree(self, node: tree_sitter.Node) -> anytree.AnyNode:

   Turn ast to anytree. Using anytree package.

7. trees2graphs(
   self,
   asts: Dict[str, tree_sitter.Tree],
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
   ) -> Dict[str, networkx.DiGraph]:

   Turn asts first to anytrees and end with directed graphs. Using package anytree and nextworkx respectively. Argument "edges_type_idx" defines the label to each edge type.
