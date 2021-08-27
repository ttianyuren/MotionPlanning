from etamp.constraint_graph import Constraint
import pickle as pk

with open('ctype_to_constraints.pk', 'rb') as f:
    ctype_to_constriants = pk.load(f)

list_cs = []

for ctype, cs in ctype_to_constriants.items():
    if len(cs) > 10:
        list_cs.append(cs)

    print(f"#{ctype}# {cs[0]},{cs[0].culprits}->cs[0].victim: {len(cs)}")

"""Generate data for BO"""
list_cs.sort(key=len, reverse=True)
data = []
for cs in list_cs:
    points = []  # for one constraint
    for c in cs:
        point = [None, None, None, None, c.yg]
        for _, v in c.culprits.items():
            c_step, decision = v
            if c_step == 1:
                point[0] = decision
            elif c_step == 7:
                point[1] = decision
            elif c_step == 13:
                point[2] = decision
            elif c_step == 19:
                point[3] = decision
        points.append(point)
    data.append(points)

with open('data.pk', 'wb') as f:
    pk.dump(data, f)
