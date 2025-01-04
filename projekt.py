import numpy as np
import matplotlib.pyplot as plt
from collections import deque

h = 0.1
ht = 0.001
T = 1
n = int(T/ht)
moc_grzejnika = 1500
rows, cols = 121, 111
D = 1


def celsius_to_kelvin(temp):
    return temp + 273.15


def kelvin_to_celsius(temp):
    return temp - 273.15


def starting_temperature(_, __):
    return celsius_to_kelvin(19)


def calculate_outdoor_temperature():
    return celsius_to_kelvin(5)


windows = []
walls = []
doors = []
heaters = []
outside = []
rooms = []


def create_plan():
    tab = np.zeros((rows, cols))
    arr = np.zeros((rows, cols), dtype='<U10')
    for j in range(cols):
        tab[0][j] = 1
        tab[1][j] = 1
        walls.append([0, j])
        walls.append([1, j])
    for j in range(32, 54):
        tab[22][j] = 1
        tab[23][j] = 1
        walls.append([22, j])
        walls.append([23, j])
    for j in range(33):
        tab[39][j] = 1
        tab[40][j] = 1
        walls.append([39, j])
        walls.append([40, j])
    for j in range(53, cols):
        tab[52][j] = 1
        tab[53][j] = 1
        walls.append([52, j])
        walls.append([53, j])
    for j in range(13, 33):
        tab[54][j] = 1
        tab[55][j] = 1
        walls.append([54, j])
        walls.append([55, j])
    for j in range(13, 33):
        tab[69][j] = 1
        tab[70][j] = 1
        walls.append([69, j])
        walls.append([70, j])
    for j in range(53, cols):
        tab[71][j] = 1
        tab[72][j] = 1
        walls.append([71, j])
        walls.append([72, j])
    for j in range(13, cols):
        tab[84][j] = 1
        tab[85][j] = 1
        walls.append([84, j])
        walls.append([85, j])
    for j in range(13, cols):
        tab[119][j] = 1
        tab[120][j] = 1
        walls.append([119, j])
        walls.append([120, j])
    for i in range(40):
        tab[i][0] = 1
        tab[i][1] = 1
        walls.append([i, 0])
        walls.append([i, 1])
    for i in range(11):
        tab[i][53] = 1
        tab[i][54] = 1
        walls.append([i, 53])
        walls.append([i, 54])
    for i in range(21, 72):
        tab[i][53] = 1
        tab[i][54] = 1
        walls.append([i, 53])
        walls.append([i, 54])
    for i in range(rows):
        tab[i][109] = 1
        tab[i][110] = 1
        walls.append([i, 109])
        walls.append([i, 110])
    for i in range(52, 72):
        tab[i][74] = 1
        tab[i][75] = 1
        walls.append([i, 74])
        walls.append([i, 75])
    for i in range(71, 85):
        tab[i][90] = 1
        tab[i][91] = 1
        walls.append([i, 90])
        walls.append([i, 91])
    for i in range(84, rows):
        tab[i][45] = 1
        tab[i][46] = 1
        walls.append([i, 45])
        walls.append([i, 46])
    for i in range(84, rows):
        tab[i][77] = 1
        tab[i][78] = 1
        walls.append([i, 77])
        walls.append([i, 78])
    for i in range(85):
        tab[i][32] = 1
        tab[i][33] = 1
        walls.append([i, 32])
        walls.append([i, 33])
    for i in range(39, rows):
        tab[i][13] = 1
        tab[i][14] = 1
        walls.append([i, 13])
        walls.append([i, 14])

    for i in range(29, 38):
        tab[i][32] = 2
        tab[i][33] = 2
        doors.append([i, 32])
        doors.append([i, 33])
    for i in range(58, 67):
        tab[i][32] = 2
        tab[i][33] = 2
        doors.append([i, 32])
        doors.append([i, 33])
    for i in range(73, 82):
        tab[i][32] = 2
        tab[i][33] = 2
        doors.append([i, 32])
        doors.append([i, 33])
    for i in range(34, 50):
        tab[i][53] = 2
        tab[i][54] = 2
        doors.append([i, 53])
        doors.append([i, 54])
    for j in range(19, 28):
        tab[39][j] = 2
        tab[40][j] = 2
        doors.append([39, j])
        doors.append([40, j])
    for j in range(35, 44):
        tab[84][j] = 2
        tab[85][j] = 2
        doors.append([84, j])
        doors.append([85, j])
    for j in range(67, 76):
        tab[84][j] = 2
        tab[85][j] = 2
        doors.append([84, j])
        doors.append([85, j])
    for j in range(80, 89):
        tab[84][j] = 2
        tab[85][j] = 2
        doors.append([84, j])
        doors.append([85, j])
    for j in range(92, 109):
        tab[84][j] = 2
        tab[85][j] = 2
        doors.append([84, j])
        doors.append([85, j])
    for j in range(64, 73):
        tab[71][j] = 2
        tab[72][j] = 2
        doors.append([71, j])
        doors.append([72, j])
    for j in range(80, 89):
        tab[71][j] = 2
        tab[72][j] = 2
        doors.append([71, j])
        doors.append([72, j])

    for j in range(36, 51):
        tab[0][j] = 3
        tab[1][j] = 3
        windows.append([0, j])
        windows.append([1, j])
    for j in range(57, 72):
        tab[0][j] = 3
        tab[1][j] = 3
        windows.append([0, j])
        windows.append([1, j])
    for i in range(47, 52):
        tab[i][13] = 3
        tab[i][14] = 3
        windows.append([i, 13])
        windows.append([i, 14])
    for i in range(73, 78):
        tab[i][13] = 3
        tab[i][14] = 3
        windows.append([i, 13])
        windows.append([i, 14])
    for i in range(13, 28):
        tab[i][0] = 3
        tab[i][1] = 3
        windows.append([i, 0])
        windows.append([i, 1])
    for j in range(25, 43):
        tab[119][j] = 3
        tab[120][j] = 3
        windows.append([119, j])
        windows.append([120, j])
    for j in range(61, 75):
        tab[119][j] = 3
        tab[120][j] = 3
        windows.append([119, j])
        windows.append([120, j])
    for i in range(12, 36):
        tab[i][109] = 3
        tab[i][110] = 3
        windows.append([i, 109])
        windows.append([i, 110])
    for i in range(57, 63):
        tab[i][109] = 3
        tab[i][110] = 3
        windows.append([i, 109])
        windows.append([i, 110])
    for i in range(90, 114):
        tab[i][109] = 3
        tab[i][110] = 3
        windows.append([i, 109])
        windows.append([i, 110])

    for i in range(15, 26):
        tab[i][2] = 4
        heaters.append([i, 2])
    for i in range(65, 70):
        tab[i][76] = 4
        heaters.append([i, 76])
    for i in range(93, 105):
        tab[i][108] = 4
        heaters.append([i, 108])
    for j in range(38, 49):
        tab[2][j] = 4
        heaters.append([2, j])
    for j in range(59, 70):
        tab[2][j] = 4
        heaters.append([2, j])
    for j in range(26, 30):
        tab[56][j] = 4
        heaters.append([56, j])
    for j in range(26, 30):
        tab[71][j] = 4
        heaters.append([71, j])
    for j in range(29, 39):
        tab[118][j] = 4
        heaters.append([118, j])
    for j in range(65, 74):
        tab[118][j] = 4
        heaters.append([118, j])

    for i in range(41, rows):
        for j in range(13):
            tab[i][j] = -1
            outside.append([i, j])

    for i in range(rows):
        for j in range(cols):
            if ([i, j] not in walls and [i, j] not in doors and [i, j] not in windows
                    and [i, j] not in heaters and [i, j] not in outside):
                rooms.append([i, j])

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
    return 101325 / (287.05 * temp_w_kelwinach)
    #      ciśnienie, efektywna masa molowa, uniwersalna stała gazowa


def calculate_doors(x, y):
    vis = np.zeros((x, y))
    for elem in doors:
        if not vis[elem[0]][elem[1]]:
            l = [[elem[0], elem[1]]]
            q = deque()
            q.append([elem[0], elem[1]])
            s = X[elem[0]][elem[1]]
            vis[elem[0]][elem[1]] = True
            while len(q):
                a = q.pop()
                for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                    if b[0] < 0 or b[0] >= x or b[1] < 0 or b[1] >= y:
                        continue
                    if plan[b[0]][b[1]] == 'door' and not vis[b[0]][b[1]]:
                        q.append((b[0], b[1]))
                        l.append((b[0], b[1]))
                        s += X[b[0]][b[1]]
                        vis[b[0]][b[1]] = True
            s = s / len(l)
            for e in l:
                X[e[0]][e[1]] = s


def calculate_windows():
    for elem in windows:
        X[elem[0]][elem[1]] = calculate_outdoor_temperature()


def calculate_room_temperature(x, y, i, j):
    vis = np.zeros((x, y))
    q = deque()
    q.append([i, j])
    s = X[i][j]
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
                s += X[b[0]][b[1]]
                vis[b[0]][b[1]] = True
    s = s / l
    return s


def calculate_heater(x, y):
    vis = np.zeros((x, y))
    max_temp = celsius_to_kelvin(20)
    for elem in heaters:
        if not vis[elem[0]][elem[1]]:
            if calculate_room_temperature(x, y, elem[0], elem[1]) < max_temp:
                l = [[elem[0], elem[1]]]
                q = deque()
                q.append([elem[0], elem[1]])
                s = X[elem[0]][elem[1]]
                vis[elem[0]][elem[1]] = True
                while len(q):
                    a = q.pop()
                    for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                        b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                        if b[0] < 0 or b[0] >= x or b[1] < 0 or b[1] >= y:
                            continue
                        if plan[b[0]][b[1]] == 'heater' and not vis[b[0]][b[1]]:
                            q.append((b[0], b[1]))
                            l.append((b[0], b[1]))
                            s += X[b[0]][b[1]]
                            vis[b[0]][b[1]] = True
                s = s / len(l)
                for e in l:
                    X[e[0]][e[1]] += moc_grzejnika / (gestosc(s) * len(l) * 1005)


def calculate_room():
    new = X
    for elem in rooms + heaters:
        new[elem[0], elem[1]] = X[elem[0], elem[1]] + D * ht / (h**2)*(X[elem[0]+1, elem[1]] + X[elem[0]-1, elem[1]] +
                                                                       X[elem[0], elem[1]+1] + X[elem[0], elem[1]-1] -
                                                                       4*X[elem[0], elem[1]])
    for elem in rooms + heaters:
        X[elem[0]][elem[1]] = new[elem[0]][elem[1]]


def calculate_wall(x, y):
    for e in walls + doors:
        for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            if e[0] + neigh[0] < 0 or e[0] + neigh[0] >= x or e[1] + neigh[1] < 0 or e[1] + neigh[1] >= y:
                continue
            if plan[e[0] + neigh[0]][e[1] + neigh[1]] == "room" or plan[e[0] + neigh[0]][e[1] + neigh[1]] == "heater":
                X[e[0]][e[1]] = X[e[0] + neigh[0]][e[1] + neigh[1]]


def main():
    for step in range(n):
        calculate_room()
        calculate_wall(rows, cols)
        calculate_doors(rows, cols)
        calculate_windows()
        calculate_heater(rows, cols)


def change_to_celsius(t):
    for i in range(rows):
        for j in range(cols):
            X[i][j] = kelvin_to_celsius(X[i][j])
    return t


main()
X = change_to_celsius(X)
plt.imshow(X)
plt.show()
