import numpy as np
from collections import deque


def starting_temperature(x, y):
    return np.exp(-(x+1)**2 - (y+1)**2)


wymiary_x = [[-1, 0], [0, 1]]
wymiary_y = [[-1, 1], [-1, 1/2]]

h = 0.1
ht = 0.001
T = 1
n = int(T/ht)
moc_grzejnika = 1500


def celsius_to_kelvin(temp):
    return temp + 273.15


def kelvin_to_celsius(temp):
    return temp - 273.15


def calculate_outdoor_temperature():
    return celsius_to_kelvin(5)


def gestosc(temp_w_kelwinach):
    return 101300 * 0.0289 / 8.31446261815324 / temp_w_kelwinach
    #      ciśnienie, efektywna masa molowa, uniwersalna stała gazowa


max_temp = celsius_to_kelvin(22)


def calculate_doors(x, y, plan, t):
    vis = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'door' and not vis[i][j]:
                l = [[i, j]]
                q = deque()
                q.append([i, j])
                s = t[i][j]
                vis[i][j] = True
                while len(q):
                    a = q.pop()
                    for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                        b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                        if b[0] < 0 or b[0] >= x or b[1] < 0 or b[1] >= y:
                            continue
                        if plan[b[0]][b[1]] == 'door' and not vis[b[0]][b[1]]:
                            q.append((b[0], b[1]))
                            l.append((b[0], b[1]))
                            s += t[b[0]][b[1]]
                            vis[b[0]][b[1]] = True
                s = s / len(l)
                for elem in l:
                    t[elem[0]][elem[1]] = s
    return t


def calculate_windows(x, y, plan, t):
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'window':
                t[i][j] = calculate_outdoor_temperature()
    return t


def calculate_room_temperature(x, y, plan, t, i, j):
    vis = np.zeros((x, y))
    q = deque()
    q.append([i, j])
    s = t[i][j]
    l = 1
    vis[i][j] = True
    while len(q):
        a = q.pop()
        for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            b = [a[0] + neighbour[0], a[1] + neighbour[1]]
            if b[0] < 0 or b[0] >= x or b[1] < 0 or b[1] >= y:
                continue
            if (plan[b[0]][b[1]] == 'heater' or plan[b[0]][b[1]] == 'room') and not vis[b[0]][b[1]]:
                q.append((b[0], b[1]))
                l += 1
                s += t[b[0]][b[1]]
                vis[b[0]][b[1]] = True
    s = s / l
    return s


def calculate_heater(x, y, plan, t):
    vis = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'heater' and not vis[i][j]:
                if calculate_room_temperature(x, y, t, plan, i, j) > max_temp:
                    continue
                l = [[i, j]]
                q = deque()
                q.append([i, j])
                s = t[i][j]
                vis[i][j] = True
                while len(q):
                    a = q.pop()
                    for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                        b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                        if b[0] < 0 or b[0] >= x or b[1] < 0 or b[1] >= y:
                            continue
                        if plan[b[0]][b[1]] == 'heater' and not vis[b[0]][b[1]]:
                            q.append((b[0], b[1]))
                            l.append((b[0], b[1]))
                            s += t[b[0]][b[1]]
                            vis[b[0]][b[1]] = True
                s = s / len(l)
                for elem in l:
                    t[elem[0]][elem[1]] += moc_grzejnika / (gestosc(celsius_to_kelvin(s)) * len(l) * 1005)
    return t


def calculate_room(x, y, plan, t):
    new = t
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'room':
                new[i, j] = t[i, j] + ht / (h**2)*(t[i+1, j] + t[i-1, j] + t[i, j+1] + t[i, j-1] - 4*t[i, j])
    return new


def calculate_wall(x, y, plan, t):
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'wall':
                for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    if plan[i + neigh[0]][j + neigh[1]] == "room" or plan[i + neigh[0]][j + neigh[1]] == "heater":
                        t[i][j] = t[i + neigh[0]][j + neigh[1]]
    return t
