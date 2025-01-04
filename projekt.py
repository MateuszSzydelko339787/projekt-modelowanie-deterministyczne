import numpy as np
import matplotlib.pyplot as plt
from collections import deque

h = 0.1
ht = 0.001
T = 1
n = int(T/ht)
moc_grzejnika = 1500
rows, cols = 121, 111


def celsius_to_kelvin(temp):
    return temp + 273.15


def kelvin_to_celsius(temp):
    return temp - 273.15


def starting_temperature(_, __):
    return celsius_to_kelvin(21)


def calculate_outdoor_temperature():
    return celsius_to_kelvin(5)


def create_plan():
    tab = np.zeros((rows, cols))
    arr = np.zeros((rows, cols), dtype='<U10')
    for j in range(cols):
        tab[0][j] = 1
        tab[1][j] = 1
    for j in range(32, 54):
        tab[22][j] = 1
        tab[23][j] = 1
    for j in range(33):
        tab[39][j] = 1
        tab[40][j] = 1
    for j in range(53, cols):
        tab[52][j] = 1
        tab[53][j] = 1
    for j in range(13, 33):
        tab[54][j] = 1
        tab[55][j] = 1
    for j in range(13, 33):
        tab[69][j] = 1
        tab[70][j] = 1
    for j in range(53, cols):
        tab[71][j] = 1
        tab[72][j] = 1
    for j in range(13, cols):
        tab[84][j] = 1
        tab[85][j] = 1
    for j in range(13, cols):
        tab[119][j] = 1
        tab[120][j] = 1
    for i in range(40):
        tab[i][0] = 1
        tab[i][1] = 1
    for i in range(11):
        tab[i][53] = 1
        tab[i][54] = 1
    for i in range(21, 72):
        tab[i][53] = 1
        tab[i][54] = 1
    for i in range(rows):
        tab[i][109] = 1
        tab[i][110] = 1
    for i in range(52, 72):
        tab[i][74] = 1
        tab[i][75] = 1
    for i in range(71, 85):
        tab[i][90] = 1
        tab[i][91] = 1
    for i in range(84, rows):
        tab[i][45] = 1
        tab[i][46] = 1
    for i in range(84, rows):
        tab[i][77] = 1
        tab[i][78] = 1
    for i in range(85):
        tab[i][32] = 1
        tab[i][33] = 1
    for i in range(39, rows):
        tab[i][13] = 1
        tab[i][14] = 1

    for i in range(29, 38):
        tab[i][32] = 2
        tab[i][33] = 2
    for i in range(58, 67):
        tab[i][32] = 2
        tab[i][33] = 2
    '''
    for i in range(58, 67):
        tab[i][13] = 2
        tab[i][14] = 2
    '''
    for i in range(73, 82):
        tab[i][32] = 2
        tab[i][33] = 2
    for i in range(34, 50):
        tab[i][53] = 2
        tab[i][54] = 2
    for j in range(19, 28):
        tab[39][j] = 2
        tab[40][j] = 2
    for j in range(35, 44):
        tab[84][j] = 2
        tab[85][j] = 2
    for j in range(67, 76):
        tab[84][j] = 2
        tab[85][j] = 2
    for j in range(80, 89):
        tab[84][j] = 2
        tab[85][j] = 2
    for j in range(92, 109):
        tab[84][j] = 2
        tab[85][j] = 2
    for j in range(64, 73):
        tab[71][j] = 2
        tab[72][j] = 2
    for j in range(80, 89):
        tab[71][j] = 2
        tab[72][j] = 2

    for j in range(36, 51):
        tab[0][j] = 3
        tab[1][j] = 3
    for j in range(57, 72):
        tab[0][j] = 3
        tab[1][j] = 3
    for i in range(47, 52):
        tab[i][13] = 3
        tab[i][14] = 3
    for i in range(73, 78):
        tab[i][13] = 3
        tab[i][14] = 3
    for i in range(13, 28):
        tab[i][0] = 3
        tab[i][1] = 3
    for j in range(25, 43):
        tab[119][j] = 3
        tab[120][j] = 3
    for j in range(61, 75):
        tab[119][j] = 3
        tab[120][j] = 3
    for i in range(12, 36):
        tab[i][109] = 3
        tab[i][110] = 3
    for i in range(57, 63):
        tab[i][109] = 3
        tab[i][110] = 3
    for i in range(90, 114):
        tab[i][109] = 3
        tab[i][110] = 3
    '''
    for j in range(38, 49):
        tab[2][j] = 4
    for j in range(59, 70):
        tab[2][j] = 4
    for i in range(15, 26):
        tab[i][2] = 4
    for j in range(29, 39):
        tab[118][j] = 4
    for j in range(65, 74):
        tab[118][j] = 4
    for i in range(93, 105):
        tab[i][108] = 4
    for j in range(26, 30):
        tab[56][j] = 4
    for j in range(26, 30):
        tab[71][j] = 4
    for i in range(65, 70):
        tab[i][76] = 4
    '''

    for i in range(41, rows):
        for j in range(13):
            tab[i][j] = -1

    for i in range(rows):
        for j in range(cols):
            if tab[i][j] == -1:
                arr[i][j] = "outside"
            elif tab[i][j] == 0:
                arr[i][j] = "room"
            elif tab[i][j] == 1:
                arr[i][j] = "wall"
            elif tab[i][j] == 2:
                arr[i][j] = "door"
            elif tab[i][j] == 3:
                arr[i][j] = "window"
            elif tab[i][j] == 4:
                arr[i][j] = "heater"
    return arr


plan = create_plan()


def create_tab():
    t = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if plan[i][j] == "room" or plan[i][j] == "wall" or plan[i][j] == "heater":
                t[i][j] = starting_temperature(i, j)
            elif plan[i][j] == "window" or plan[i][j] == "outside":
                t[i][j] = calculate_outdoor_temperature()
    return t


X = create_tab()


def gestosc(temp_w_kelwinach):
    return 101300 * 0.0289 / 8.31446261815324 / temp_w_kelwinach
    #      ciśnienie, efektywna masa molowa, uniwersalna stała gazowa


def calculate_doors(x, y, t):
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


def calculate_windows(x, y, t):
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'window':
                t[i][j] = calculate_outdoor_temperature()
    return t


def calculate_room_temperature(x, y, t, i, j):
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


def calculate_heater(x, y, t):
    vis = np.zeros((x, y))
    max_temp = celsius_to_kelvin(22)
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'heater' and not vis[i][j]:
                if calculate_room_temperature(x, y, t, i, j) > max_temp:
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


def calculate_room(x, y, t):
    new = t
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'room':
                new[i, j] = t[i, j] + ht / (h**2)*(t[i+1, j] + t[i-1, j] + t[i, j+1] + t[i, j-1] - 4*t[i, j])
    return new


def calculate_wall(x, y, t):
    for i in range(x):
        for j in range(y):
            if plan[i][j] == 'wall' or plan[i][j] == 'door':
                for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    if i + neigh[0] < 0 or i + neigh[0] >= x or j + neigh[1] < 0 or j + neigh[1] >= y:
                        continue
                    if plan[i + neigh[0]][j + neigh[1]] == "room" or plan[i + neigh[0]][j + neigh[1]] == "heater":
                        t[i][j] = t[i + neigh[0]][j + neigh[1]]
    return t


def main(t):
    for step in range(n):
        t = calculate_room(rows, cols, t)
        t = calculate_wall(rows, cols, t)
        t = calculate_doors(rows, cols, t)
        t = calculate_windows(rows, cols, t)
        t = calculate_heater(rows, cols, t)
    return t


def change_to_celsius(t):
    for i in range(rows):
        for j in range(cols):
            t[i][j] = kelvin_to_celsius(t[i][j])
    return t


X = main(X)
X = change_to_celsius(X)
plt.imshow(X)
plt.show()
