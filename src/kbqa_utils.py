import re
import os
import json
import urllib
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, AnyStr, List, Union, Tuple, Any, Optional, Set
from enum import Enum
from collections import defaultdict

from SPARQLWrapper import SPARQLWrapper, JSON
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy


class Expr:
    
    def __init__(
        self, 
        arg0: int,
        tokens: List[str]
    ) -> None:
        self.arg0 = arg0
        self.tokens = tokens

    def __str__(self) -> str:
        return "([v{}] = {})".format(self.arg0, " ".join(self.tokens))


class KBQADataInstance:
    
    def __init__(
        self,
        qid: str,
        query: str,
        S_expr: str,  # target query graph
        relations: List[str],
        answer: List[str]
    ) -> None:
        self.qid = qid
        self.query = query
        self.relations = relations
        self.answer = answer
        
        self.parse(S_expr)

    def parse(self, S_expr: str):
        exprs, ents = parse_S_expr(S_expr)
        self.exprs: List[Expr] = exprs
        self.ents: List[str] = ents


class TrainDataInstance:
    
    def __init__(self,
        query: str,
        prefix: List[Expr],
        target: Expr,
        rels: List[str],
        rev_rels: List[str]
    ) -> None:
        self.query = query
        self.prefix = prefix
        self.target = target
        self.rels = rels
        self.rev_rels = rev_rels


class DataBatch:
    
    def __init__(
        self,
        batch: List[TrainDataInstance]
    ) -> None:
        self.batch = batch
        
        self.parse()
    
    def __len__(self):
        return len(self.batch)

    def parse(self):
        self.batch_query = [I.query for I in self.batch]
        self.batch_rels = [I.rels for I in self.batch]


class KBQADataset(Dataset):
    
    def __init__(
        self, 
        data: List[Union[KBQADataInstance, TrainDataInstance]], 
        **kwargs
    ) -> None:
        # instances
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> KBQADataInstance:
        return self.data[index]


def is_entity(token: str) -> bool:
    return re.match("(?:m|g)\.", token) or token == 'Country'


def parse_S_expr(S_expr: str) -> List[Expr]:
    exprs: List[Expr] = []
    tokens = [tok for tok in re.split(r"([\(\) ])", S_expr) if tok.strip() != ""]
    m_vars = {}
    ents = []
    stk  = []
    for tok in tokens:
        if is_entity(tok):
            m_vars[tok] = len(m_vars)
            ents.append(tok)
    for tok in tokens:
        tok = "{}".format(tok.strip())
        # if re.fullmatch("(?:JOIN)|(?:AND)|(?:CONS)|(?:TC)|(?:R)|(?:ARGMAX)|(?:ARGMIN)", tok):
        #     tok = "[{}]".format(tok)
        if tok == '(':
            stk.append(tok)
        elif tok == ')':
            expr_tokens = []
            while stk[-1] != '(':
                expr_tokens.append(stk.pop())
            stk.pop()
            expr_tokens.reverse()
            expr_str = " ".join(expr_tokens)
            if expr_tokens[0] != 'R':
                if expr_str not in m_vars:
                    m_vars[expr_str] = len(m_vars)
                arg0 = m_vars[expr_str]
                stk.append("[v{}]".format(arg0))
                exprs.append(Expr(arg0, expr_tokens))
            else:
                stk.extend(expr_tokens)
        else:
            if tok not in m_vars:
                stk.append(tok)
            else:
                stk.append("[v{}]".format(m_vars[tok]))

    return exprs, ents


def build_extra_tokens(
    dataset_dict: Dict[str, KBQADataset], 
    rel_dict:  Dict[str, Set[str]],
    type_dict: Dict[str, Set[str]],
    threshold: int = 5
) -> Set[str]:
    tok_dict = defaultdict(int)
    for key in ["train", "dev"]:
        dataset = dataset_dict[key]
        for item in dataset.data:
            for expr in item.exprs:
                for tok in expr.tokens:
                    if tok in rel_dict["train"] or tok in type_dict["train"]:
                        continue
                    tok_dict[tok] += 1
    extra_tokens = [k for k, v in tok_dict.items() if v >= threshold]
    return sorted(extra_tokens)


class DBClient:

    def __init__(self, url: str = "http://10.77.110.128:3001/sparql") -> None:
        self.sparql = SPARQLWrapper(url)
        self.sparql.setReturnFormat(JSON)
    
    def execute(self, lisp_program: str) -> List[str]:
        sparql_program = lisp_to_sparql(lisp_program)
        self.sparql.setQuery(sparql_program)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err
        ...

 
def lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            # subp[2] = subp[2].split("^^")[0] + '-08:00^^' + subp[2].split("^^")[1]
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?sk0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]