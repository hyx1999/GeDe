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


class RawDataInstance:
    
    def __init__(
        self,
        qid: str,
        query: str,
        S_expr: str,  # target query graph
        answer: List[str]
    ) -> None:
        self.qid = qid
        self.query = query
        self.answer = answer
        
        self.parse(S_expr)

    def parse(self, S_expr: str):
        exprs, ents = parse_S_expr(S_expr)
        self.exprs: List[Expr] = exprs
        self.ents: List[str] = ents
    
    def check_exprs(self) -> bool:
        pat = re.compile('(?:g|m)\.|#\d+')
        for expr in self.exprs:
            if expr.tokens[0] == "JOIN":
                if not pat.match(expr.tokens[-1]):
                    return False
        return True


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
        data: List[Union[RawDataInstance, TrainDataInstance]], 
        **kwargs
    ) -> None:
        # instances
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> RawDataInstance:
        return self.data[index]


def parse_S_expr(S_expr: str) -> List[Expr]:
    exprs: List[Expr] = []
    tokens = [tok for tok in re.split(r"([\(\) ])", S_expr) if tok.strip() != ""]
    m_vars = {}
    ents = []
    stk  = []
    for tok in tokens:
        if re.match("(?:m|g)\.", tok):
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
                stk.append("#{}".format(arg0))
                exprs.append(Expr(arg0, expr_tokens))
            else:
                stk.extend(expr_tokens)
        else:
            if tok not in m_vars:
                stk.append(tok)
            else:
                stk.append("#{}".format(m_vars[tok]))

    return exprs, ents


def build_extra_tokens(
    dataset_dict: Dict[str, KBQADataset], 
    rel_dict: Dict[str, Set[str]],
    threshold: int = 5
) -> Set[str]:
    tok_dict = defaultdict(int)
    for key in ["train", "dev"]:
        dataset = dataset_dict[key]
        for item in dataset.data:
            for expr in item.exprs:
                for tok in expr.tokens:
                    if tok in rel_dict["train"]:
                        continue
                    tok_dict[tok] += 1
    for i in range(10):
        tok_dict[f"[v{i}]"] += threshold
    extra_tokens = [k for k, v in tok_dict.items() if v >= threshold]
    return sorted(extra_tokens)


def build_domain_info(
    rel_dict: Dict[str, Set[str]]
) -> Set[str]:
    domain_info = set()
    for rel in rel_dict["train"]:
        prefix = rel.split(".")[0]
        domain_info.add(prefix)
    return domain_info


class KBClient:

    def __init__(self, url: str = "http://10.77.110.128:3001/sparql") -> None:
        self.sparql = SPARQLWrapper(url)
        self.sparql.setReturnFormat(JSON)
        self.max_ents_num = 100
    

    def execute_join(self, sparql_tokens: List[str], heads: List[str]) -> List[str]:
        rel = sparql_tokens[-1]
        if sparql_tokens[2] == "[R]":
            return self.query_ent(heads, rel, rev=True)
        else:
            return self.query_ent(heads, rel, rev=False)


    def execute_and(self, 
        lisp_tokens: List[str], 
        ents_left:  Optional[List[str]] = None,
        ents_right: Optional[List[str]] = None,
    ):
        pat = re.compile("#\d*]")
        if not pat.match(lisp_tokens[2]):
            t1, t2 = lisp_tokens[1], lisp_tokens[2]
            lisp_tokens[1] = t2
            lisp_tokens[2] = t1
            ents_left, ents_right = ents_right, ents_left
        if not pat.match(lisp_tokens[1]):
            ents = " ".join([f"ns:{e}" for e in ents_right])
            class_name = "ns:{}".format(lisp_tokens[1])
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x WHERE {{
                    ?x ns:type.object.type {class_name} .
                    VALUES ?x{{ {ents} }}
                }}
            """
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError as err:
                raise err

            return [i['x']['value'] for i in results['results']['bindings']]
        else:
            return list(set(ents_left) & set(ents_right))


    def execute_cons(
        self,
        lisp_tokens: List[str],
        ents: List[str]
    ):
        ents = " ".join([f"ns:{e}" for e in ents])
        rel  = "ns:{}".format(lisp_tokens[2])
        cons = "ns:{}".format(lisp_tokens[3])
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x WHERE {{
                ?x {rel} {cons} .
                VALUES ?x{{ {ents} }}
            }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err
        return [i['x']['value'] for i in results['results']['bindings']]


    def execute_tc(
        self,
        lisp_tokens: List[str],
        ents: List[str]
    ):
        ents = " ".join([f"ns:{e}" for e in ents])
        rel = "ns:{}".format(lisp_tokens[2])
        year = lisp_tokens[3]
        if year == "NOW":
            from_para = '"2015-08-10"^^xsd:dateTime'
            to_para = '"2015-08-10"^^xsd:dateTime'
        else:
            from_para = f'"{year}-12-31"^^xsd:dateTime'
            to_para = f'"{year}-01-01"^^xsd:dateTime'
        filter_clauses0 = f'FILTER(NOT EXISTS {{ ?x {rel} ?sk0 }} || EXISTS {{ ?x {rel} ?sk1 . FILTER(xsd::datetime(?sk1) <= {from_para}) }})'
        if rel.endswith('from'):
            rel = rel[:-4] + "to"  # from -> to
            filter_clauses1 = f'FILTER(NOT EXISTS {{ ?x {rel} ?sk2 }} || EXISTS {{ ?x {rel} ?sk3 . FILTER(xsd::datetime(?sk3) >= {to_para}) }})'
        else:
            rel = rel[:-9] + "to_date"  # from_date -> to_date
            filter_clauses1 = f'FILTER(NOT EXISTS {{ ?x {rel} ?sk2 }} || EXISTS {{ ?x {rel} ?sk3 . FILTER(xsd::datetime(?sk1) >= {to_para}) }})'
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x WHERE {{
                VALUES ?x{{ {ents} }}
                {filter_clauses0}
                {filter_clauses1}
            }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err
        return [i['x']['value'] for i in results['results']['bindings']]


    def execute_argmax(
        self,
        lisp_tokens: List[str],
        ents: List[str]
    ):
        rel = "ns:{}".format(lisp_tokens[-1])
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x WHERE {{
                ?x {rel} ?sk0 .
                VALUES ?x{{ {ents} }}
            }}
            ORDER BY DESC(?sk0)
            LIMIT 1
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err
        return [i['x']['value'] for i in results['results']['bindings']]


    def execute_argmin(
        self,
        lisp_tokens: List[str],
        ents: List[str]
    ):
        rel = "ns:{}".format(lisp_tokens[-1])
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x WHERE {{
                ?x {rel} ?sk0 .
                VALUES ?x{{ {ents} }}
            }}
            ORDER BY ?sk0
            LIMIT 1
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err
        return [i['x']['value'] for i in results['results']['bindings']]


    def query_rel(self, ents: List[str], rev: bool) -> List[str]:
        ents = sorted(ents)
        if len(ents) == 0:
            return [], []
        ents = " ".join([f"ns:{e}" for e in ents])
        if not rev:
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {{
                    ?src ?r0_ ?t0 .
                    VALUES ?src {{ {ents} }}
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(ns:)) as ?r0)
                }}
            """
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError as err:
                raise err
            rels = [i['r0']['value'] for i in results['results']['bindings'] if i['r0']['value'] != 'type.object.type']
            return rels
        else:
            rev_query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {{
                    ?src ?r0_ ?t0 .
                    VALUES ?src {{ {ents} }}
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(ns:)) as ?r0)
                }}
            """
            self.sparql.setQuery(rev_query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError as err:
                raise err            
            rev_rels = [i['r0']['value'] for i in results['results']['bindings'] if i['r0']['value'] != 'type.object.type']
            return rev_rels           


    def query_ent(self, heads: List[str], rel: str, rev: bool) -> List[str]:
        heads = sorted(heads)
        if len(heads) == 0:
            return []
        if rev:
            heads = " ".join([f"ns:{h}" for h in heads])
            rel = f"ns:{rel}"
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {{
                    ?t0_ {rel} ?src .
                    VALUES ?src {{ {heads} }}
                    FILTER regex(?t0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?t0_),str(ns:)) as ?t0)
                }}
            """
        else:
            heads = " ".join([f"ns:{h}" for h in heads])
            rel = f"ns:{rel}"
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {{
                    ?src {rel} ?t0_ .
                    VALUES ?src {{ {heads} }}
                    FILTER regex(?t0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?t0_),str(ns:)) as ?t0)
                }}
            """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err

        return [i['t0']['value'] for i in results['results']['bindings']][:self.max_ents_num]


    def query_value(self, heads: List[str], rel: str, rev: bool) -> List[str]:
        heads = sorted(heads)
        if len(heads) == 0:
            return []
        if rev:
            heads = " ".join([f"ns:{h}" for h in heads])
            rel = f"ns:{rel}"
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {{
                    ?t0 {rel} ?src .
                    VALUES ?src {{ {heads} }}
                    FILTER (!regex(?t0, "http://rdf.freebase.com/ns/"))
                }}
            """
        else:
            heads = " ".join([f"ns:{h}" for h in heads])
            rel = f"ns:{rel}"
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {{
                    ?src {rel} ?t0 .
                    VALUES ?src {{ {heads} }}
                    FILTER (!regex(?t0, "http://rdf.freebase.com/ns/"))
                }}
            """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError as err:
            raise err

        return [i['t0']['value'] for i in results['results']['bindings']]
