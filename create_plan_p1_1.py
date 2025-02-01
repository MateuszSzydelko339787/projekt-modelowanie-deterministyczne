import numpy as np
import matplotlib.pyplot as plt

rows, cols = 60, 55


def create_plan():
    tab = np.zeros((rows, cols))
    arr = np.zeros((rows, cols), dtype='<U10')
    for j in range(cols):
        tab[0][j] = 1
    for j in range(15, 28):
        tab[11][j] = 1
        tab[12][j] = 1
    for j in range(17):
        tab[19][j] = 1
        tab[20][j] = 1
    for j in range(26, cols):
        tab[24][j] = 1
        tab[25][j] = 1
    for j in range(6, 17):
        tab[26][j] = 1
        tab[27][j] = 1
    for j in range(6, 17):
        tab[34][j] = 1
        tab[35][j] = 1
    for j in range(26, cols):
        tab[35][j] = 1
        tab[36][j] = 1
    for j in range(6, cols):
        tab[42][j] = 1
        tab[43][j] = 1
    for j in range(6, cols):
        tab[59][j] = 1
    for i in range(21):
        tab[i][0] = 1
    for i in range(37):
        tab[i][26] = 1
        tab[i][27] = 1
    for i in range(rows):
        tab[i][54] = 1
    for i in range(24, 36):
        tab[i][37] = 1
        tab[i][38] = 1
    for i in range(35, 44):
        tab[i][44] = 1
        tab[i][45] = 1
    for i in range(83, rows):
        tab[i][22] = 1
        tab[i][23] = 1
    for i in range(83, rows):
        tab[i][38] = 1
        tab[i][39] = 1
    for i in range(44):
        tab[i][15] = 1
        tab[i][16] = 1
    for i in range(19, rows):
        tab[i][7] = 1
    for i in range(42, rows):
        tab[i][22] = 1
        tab[i][23] = 1
    for i in range(42, rows):
        tab[i][38] = 1
        tab[i][39] = 1

    for i in range(14, 19):
        tab[i][15] = 2
        tab[i][16] = 2
    for i in range(29, 34):
        tab[i][15] = 2
        tab[i][16] = 2
    for i in range(37, 41):
        tab[i][15] = 2
        tab[i][16] = 2
    for i in range(6, 11):
        tab[i][26] = 2
        tab[i][27] = 2
    for i in range(15, 23):
        tab[i][26] = 2
        tab[i][27] = 2
    for j in range(10, 15):
        tab[19][j] = 2
        tab[20][j] = 2
    for j in range(18, 22):
        tab[42][j] = 2
        tab[43][j] = 2
    for j in range(33, 38):
        tab[42][j] = 2
        tab[43][j] = 2
    for j in range(40, 44):
        tab[42][j] = 2
        tab[43][j] = 2
    for j in range(46, 54):
        tab[42][j] = 2
        tab[43][j] = 2
    for j in range(32, 36):
        tab[35][j] = 2
        tab[36][j] = 2
    for j in range(40, 44):
        tab[35][j] = 2
        tab[36][j] = 2

    for j in range(18, 25):
        tab[0][j] = 3
    for j in range(29, 37):
        tab[0][j] = 3
    for j in range(13, 21):
        tab[59][j] = 3
    for j in range(30, 37):
        tab[59][j] = 3
    for i in range(7, 14):
        tab[i][0] = 3
    for i in range(24, 26):
        tab[i][7] = 3
    for i in range(37, 39):
        tab[i][7] = 3
    for i in range(6, 18):
        tab[i][54] = 3
    for i in range(29, 32):
        tab[i][54] = 3
    for i in range(45, 57):
        tab[i][54] = 3

    for i in range(8, 13):
        tab[i][1] = 4
    for i in range(30, 32):     # tu była zmiana
        tab[i][53] = 4
    for i in range(48, 54):
        tab[i][53] = 4
    for i in range(1, 6):
        tab[i][53] = 4
    for j in range(19, 24):
        tab[1][j] = 4
    for j in range(30, 36):
        tab[1][j] = 4
    for j in range(13, 15):
        tab[28][j] = 4
    for i in range(37, 39):     # tu była zmiana
        tab[i][8] = 4
    for j in range(15, 20):
        tab[58][j] = 4
    for j in range(32, 36):
        tab[58][j] = 4

    for i in range(20, rows):
        for j in range(7):
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
    # plt.imshow(tab)
    # plt.show()
    return arr


# plan = create_plan()
