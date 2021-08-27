#!/usr/bin/env python
from etamp.constraint_graph import Constraint, reset_constraint_dict
import pickle as pk
import numpy as np
from run_branch_p1 import main as main_1
from run_branch_p2 import main as main_2
from run_branch_p3 import main as main_3
from run_branch_p4 import main as main_4
import os

#######################################################


if __name__ == '__main__':

    fname = 'ctype_to_constraints_p14.pk'
    with open(fname, 'rb') as f:
        Constraint.ctype_to_constraints = pk.load(f)
        for ctype, cs in Constraint.ctype_to_constraints.items():
            print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")

    constraint_data = Constraint.ctype_to_constraints.copy()

    list_visits = []

    for i in range(200):
        print(f'exp {i} -------------------------------------------------------------------')

        sk_visits = main_3(use_bo=1, visualization=0, new_problem=0)
        for ctype, cs in Constraint.ctype_to_constraints.items():
            print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")

        list_visits.append(int(sk_visits))
        Constraint.ctype_to_constraints = constraint_data.copy()

        print(list_visits)
        a = np.asarray(list_visits).astype(int)
        np.savetxt("list_visits_transfer.csv", a, delimiter=",")
