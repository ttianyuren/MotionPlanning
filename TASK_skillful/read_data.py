import pickle as pk

with open('data.pk', 'rb') as f:
    data = pk.load(f)

g_a, g_b, g_c, g_d, g_e, g_f = data

for point in g_f:
    x_1, x_2, x_3, x_4, y = point

    print(
        (f'x_1: {x_1} ' if x_1 else '') + (f' x_2: {x_2} ' if x_2 else '') + (f' x_3: {x_3} ' if x_3 else '') + (
            f' x_4: {x_4} ' if x_4 else '') + f' y: {y}')
