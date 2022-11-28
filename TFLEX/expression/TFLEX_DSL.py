"""
@date: 2022/2/21
@description: null
"""
import random
from math import pi
from typing import List, Union

from TFLEX.expression.ParamSchema import get_param_name_list
from TFLEX.expression.symbol import Interpreter

query_structures = {
    # 1. 1-hop Pe and Pt, manually
    # 2. entity multi-hop
    "Pe2": "def Pe2(e1, r1, t1, r2, t2): return Pe(Pe(e1, r1, t1), r2, t2)",  # 2p
    "Pe3": "def Pe3(e1, r1, t1, r2, t2, r3, t3): return Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)",  # 3p
    # 3. time multi-hop
    "Pt_lPe": "def Pt_lPe(e1, r1, t1, r2, e2): return Pt(Pe(e1, r1, t1), r2, e2)",  # l for left (as head entity)
    "Pt_rPe": "def Pt_rPe(e1, r1, e2, r2, t1): return Pt(e1, r1, Pe(e2, r2, t1))",  # r for right (as tail entity)
    "Pe_Pt": "def Pe_Pt(e1, r1, e2, r2, e3): return Pe(e1, r1, Pt(e2, r2, e3))",  # at
    "Pe_aPt": "def Pe_aPt(e1, r1, e2, r2, e3): return Pe(e1, r1, after(Pt(e2, r2, e3)))",  # a for after
    "Pe_bPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))",  # b for before
    "Pe_nPt": "def Pe_nPt(e1, r1, e2, r2, e3): return Pe(e1, r1, next(Pt(e2, r2, e3)))",  # n for next
    # 4. entity and & time and
    "e2i": "def e2i(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2i
    "e3i": "def e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))",  # 3i
    "t2i": "def t2i(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2i
    "t3i": "def t3i(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), Pt(e5, r3, e6))",  # t-3i
    # 5. complex time and
    "e2i_Pe": "def e2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))",  # pi
    "Pe_e2i": "def Pe_e2i(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(e2i(e1, r1, t1, e2, r2, t2), r3, t3)",  # ip
    "Pt_le2i": "def Pt_le2i(e1, r1, t1, e2, r2, t2, r3, e3): return Pt(e2i(e1, r1, t1, e2, r2, t2), r3, e3)",  # mix ip
    "Pt_re2i": "def Pt_re2i(e1, r1, e2, r2, t1, e3, r3, t2): return Pt(e1, r1, e2i(e2, r2, t1, e3, r3, t2))",  # mix ip
    "t2i_Pe": "def t2i_Pe(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), Pt(e3, r3, e4))",  # t-pi
    "Pe_t2i": "def Pe_t2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, t2i(e2, r2, e3, e4, r3, e5))",  # t-ip
    "Pe_at2i": "def Pe_at2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, after(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_bt2i": "def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, before(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_nt2i": "def Pe_nt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, next(t2i(e2, r2, e3, e4, r3, e5)))",
    "between": "def between(e1, r1, e2, e3, r2, e4): return TimeAnd(after(Pt(e1, r1, e2)), before(Pt(e3, r2, e4)))",  # between(t1, t2) == after t1 and before t2
    # 5. entity not
    "e2i_NPe": "def e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))",  # pni = e2i_N(Pe(e1, r1, t1), r2, t2, e2, r3, t3)
    "e2i_PeN": "def e2i_PeN(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))",  # pin
    "Pe_e2i_Pe_NPe": "def Pe_e2i_Pe_NPe(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)",  # inp
    "e2i_N": "def e2i_N(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))",  # 2in
    "e3i_N": "def e3i_N(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))",  # 3in
    # 6. time not
    "t2i_NPt": "def t2i_NPt(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(TimeNot(Pt(Pe(e1, r1, t1), r2, e2)), Pt(e3, r3, e4))",  # t-pni
    "t2i_PtN": "def t2i_PtN(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), TimeNot(Pt(e3, r3, e4)))",  # t-pin
    "Pe_t2i_PtPe_NPt": "def Pe_t2i_PtPe_NPt(e1, r1, e2, r2, t2, r3, e3, e4, r4, e5): return Pe(e1, r1, TimeAnd(Pt(Pe(e2, r2, t2), r3, e3), TimeNot(Pt(e4, r4, e5))))",  # t-inp
    "t2i_N": "def t2i_N(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), TimeNot(Pt(e3, r2, e4)))",  # t-2in
    "t3i_N": "def t3i_N(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), TimeNot(Pt(e5, r3, e6)))",  # t-3in
    # 7. entity union & time union
    "e2u": "def e2u(e1, r1, t1, e2, r2, t2): return Or(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2u
    "Pe_e2u": "def Pe_e2u(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)",  # up
    "t2u": "def t2u(e1, r1, e2, e3, r2, e4): return TimeOr(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2u
    "Pe_t2u": "def Pe_t2u(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeOr(Pt(e2, r2, e3), Pt(e4, r3, e5)))",  # t-up
    # 8. union-DM
    "e2u_DM": "def e2u_DM(e1, r1, t1, e2, r2, t2): return Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2))))",  # 2u-DM
    "Pe_e2u_DM": "def Pe_e2u_DM(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2)))), r3, t3)",  # up-DM
    "t2u_DM": "def t2u_DM(e1, r1, e2, e3, r2, e4): return TimeNot(TimeAnd(TimeNot(Pt(e1, r1, e2)), TimeNot(Pt(e3, r2, e4))))",  # t-2u-DM
    "Pe_t2u_DM": "def Pe_t2u_DM(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeNot(TimeAnd(TimeNot(Pt(e2, r2, e3)), TimeNot(Pt(e4, r3, e5)))))",  # t-up-DM
    # 9. union-DNF
    "e2u_DNF": "def e2u_DNF(e1, r1, t1, e2, r2, t2): return Pe(e1, r1, t1), Pe(e2, r2, t2)",  # 2u_DNF
    "Pe_e2u_DNF": "def Pe_e2u_DNF(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Pe(e1, r1, t1), r3, t3), Pe(Pe(e2, r2, t2), r3, t3)",  # up_DNF
    "t2u_DNF": "def t2u_DNF(e1, r1, e2, e3, r2, e4): return Pt(e1, r1, e2), Pt(e3, r2, e4)",  # t-2u_DNF
    "Pe_t2u_DNF": "def Pe_t2u_DNF(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, Pt(e2, r2, e3)), Pe(e1, r1, Pt(e4, r3, e5))",  # t-up_DNF
}
union_query_structures = [
    "e2u", "Pe_e2u",  # 2u, up
    "t2u", "Pe_t2u",  # t-2u, t-up
]
train_query_structures = [
    # entity
    "Pe", "Pe2", "Pe3", "e2i", "e3i",  # 1p, 2p, 3p, 2i, 3i
    "e2i_NPe", "e2i_PeN", "Pe_e2i_Pe_NPe", "e2i_N", "e3i_N",  # npi, pni, inp, 2in, 3in
    # time
    "Pt", "Pt_lPe", "Pt_rPe", "Pe_Pt", "Pe_aPt", "Pe_bPt", "Pe_nPt",  # t-1p, t-2p
    "t2i", "t3i", "Pt_le2i", "Pt_re2i", "Pe_t2i", "Pe_at2i", "Pe_bt2i", "Pe_nt2i", "between",  # t-2i, t-3i
    "t2i_NPt", "t2i_PtN", "Pe_t2i_PtPe_NPt", "t2i_N", "t3i_N",  # t-npi, t-pni, t-inp, t-2in, t-3in
]
test_query_structures = train_query_structures + [
    # entity
    "e2i_Pe", "Pe_e2i",  # pi, ip
    "e2u", "Pe_e2u",  # 2u, up
    # time
    "t2i_Pe", "Pe_t2i",  # t-pi, t-ip
    "t2u", "Pe_t2u",  # t-2u, t-up
    # union-DM
    "e2u_DM", "Pe_e2u_DM",  # 2u-DM, up-DM
    "t2u_DM", "Pe_t2u_DM",  # t-2u-DM, t-up-DM
]


def is_to_predict_entity_set(query_name) -> bool:
    return query_name.startswith("e") or query_name.startswith("Pe")


def query_contains_union_and_we_should_use_DNF(query_name) -> bool:
    return query_name in union_query_structures


# def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5):
#     if t1 in before(t2):
#         return t2i(e2, r2, e3, e4, r3, e5)
#     else:
#         return t2i(e2, r2, e3, e4, r3, e5)

class BasicParser(Interpreter):
    """
    abstract class
    """

    def __init__(self, variables, neural_ops):
        alias = {
            "Pe": neural_ops["EntityProjection"],
            "Pt": neural_ops["TimeProjection"],
            "before": neural_ops["TimeBefore"],
            "after": neural_ops["TimeAfter"],
            "next": neural_ops["TimeNext"],
        }
        predefine = {
            "pi": pi,
        }
        functions = dict(**neural_ops, **alias, **predefine)
        super().__init__(usersyms=dict(**variables, **functions))
        self.func_cache = {}
        for _, qs in query_structures.items():
            self.eval(qs)

    def fast_function(self, func_name):
        if func_name in self.func_cache:
            return self.func_cache[func_name]
        func = self.eval(func_name)
        self.func_cache[func_name] = func
        return func

    def fast_args(self, func_name) -> List[str]:
        return get_param_name_list(self.fast_function(func_name))


class NeuralParser(BasicParser):
    def __init__(self, neural_ops, variables=None):
        if variables is None:
            variables = {}
        must_implement_neural_ops = [
            "And",
            "And3",
            "Or",
            "Not",
            "EntityProjection",
            "TimeProjection",
            "TimeAnd",
            "TimeAnd3",
            "TimeOr",
            "TimeNot",
            "TimeBefore",
            "TimeAfter",
            "TimeNext",
        ]
        for op in must_implement_neural_ops:
            if op not in neural_ops:
                raise Exception(f"{op} Not Found! You MUST implement neural operation '{op}'")
        super().__init__(variables=variables, neural_ops=neural_ops)
