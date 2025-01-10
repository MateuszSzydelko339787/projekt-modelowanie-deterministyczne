import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


h = 0.1
ht = 0.001
T = 1
n = int(T/ht)
moc_grzejnika = 1500
rows, cols = 121, 111
D = 2.5
total_energy = 0


def calculate_time(t):
    t = t % (60 * 24)
    return t, t//60, t % 60  # czas w minutach, godzina, minuty


def celsius_to_kelvin(temp):
    return temp + 273.15


def kelvin_to_celsius(temp):
    return temp - 273.15


def starting_temperature(_, __):
    return celsius_to_kelvin(19)


def setup_temperature():
    """
    kod i dane ze strony:
    https://open-meteo.com/en/docs/historical-weather-api#start_date=2024-12-01&end_date=2024-12-31
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
        "hourly": "temperature_2m"
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = dict(date=pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ))
    hourly_data["temperature_2m"] = hourly_temperature_2m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    tab = np.array(hourly_dataframe)
    for elem in tab:
        elem[1] = celsius_to_kelvin(elem[1])
    return tab


outdoors_temperature = setup_temperature()


def calculate_outdoor_temperature(times):
    return outdoors_temperature[times//60][1]


def change_to_celsius(t):
    for i in range(rows):
        for j in range(cols):
            t[i][j] = kelvin_to_celsius(X[i][j])
    return t


def change_to_kelwin(t):
    for i in range(rows):
        for j in range(cols):
            t[i][j] = celsius_to_kelvin(X[i][j])
    return t


windows = []
walls = []
doors = []
heaters = []
heaters_use = []
outside = []
rooms = []
animations = []


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
    for i in range(12, 21):
        tab[i][53] = 2
        tab[i][54] = 2
        doors.append([i, 53])
        doors.append([i, 54])
    for i in range(32, 48):
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
    heaters_use.append([15, 2])
    for i in range(65, 70):
        tab[i][76] = 4
        heaters.append([i, 76])
    heaters_use.append([65, 76])
    for i in range(93, 105):
        tab[i][108] = 4
        heaters.append([i, 108])
    heaters_use.append([93, 108])
    for j in range(38, 49):
        tab[2][j] = 4
        heaters.append([2, j])
    heaters_use.append([2, 38])
    for j in range(59, 70):
        tab[2][j] = 4
        heaters.append([2, j])
    heaters_use.append([2, 59])
    for j in range(26, 30):
        tab[56][j] = 4
        heaters.append([56, j])
    heaters_use.append([56, 26])
    for j in range(26, 30):
        tab[71][j] = 4
        heaters.append([71, j])
    heaters_use.append([71, 26])
    for j in range(29, 39):
        tab[118][j] = 4
        heaters.append([118, j])
    heaters_use.append([118, 29])
    for j in range(65, 74):
        tab[118][j] = 4
        heaters.append([118, j])
    heaters_use.append([118, 65])

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


def is_closed(times):
    t, hours, minutes = calculate_time(times)
    if hours in [1, 8, 18, 22]:
        return False
    return True


def is_working(times):
    t, hours, minutes = calculate_time(times)
    if hours in [1, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21]:
        return True
    return False


plan = create_plan()


def create_tab():
    t = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if plan[i][j] == "room" or plan[i][j] == "wall" or plan[i][j] == "heater" or plan[i][j] == "door":
                t[i][j] = starting_temperature(i, j)
            elif plan[i][j] == "outside":
                t[i][j] = calculate_outdoor_temperature(0)
            elif is_closed(0):
                t[i][j] = starting_temperature(i, j)
            else:
                t[i][j] = calculate_outdoor_temperature(0)
    return t


X = create_tab()
vis = np.zeros((rows, cols))
animations.append(change_to_celsius(X.copy()))


def gestosc(temp_w_kelwinach):
    return 101325 / (287.05 * temp_w_kelwinach)
    #      ciśnienie, efektywna masa molowa * uniwersalna stała gazowa


def calculate_doors():
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
                    if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                        continue
                    if plan[b[0]][b[1]] == 'door' and not vis[b[0]][b[1]]:
                        q.append((b[0], b[1]))
                        l.append((b[0], b[1]))
                        s += X[b[0]][b[1]]
                        vis[b[0]][b[1]] = True
            s = s / len(l)
            for e in l:
                X[e[0]][e[1]] = s
    for elem in doors:
        vis[elem[0], elem[1]] = 0


def calculate_windows(times):
    if is_closed(times):
        trans = 0
        for elem in windows:
            for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                b = [elem[0] + neighbour[0], elem[1] + neighbour[1]]
                if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                    continue
                if plan[b[0]][b[1]] == 'heater' or plan[b[0]][b[1]] == 'room':
                    X[elem[0]][elem[1]] = trans * calculate_outdoor_temperature(times) + (1 - trans) * X[b[0]][b[1]]
    else:
        for elem in windows:
            X[elem[0]][elem[1]] = calculate_outdoor_temperature(times)


def calculate_room_temperature(i, j):
    q = deque()
    q.append([i, j])
    s = X[i][j]
    l = [(i, j)]
    vis[i][j] = 1
    while len(q):
        a = q.pop()
        for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            b = [a[0] + neighbour[0], a[1] + neighbour[1]]
            if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                continue
            if (plan[b[0]][b[1]] == 'heater' or plan[b[0]][b[1]] == 'room') and not vis[b[0]][b[1]]:
                q.append((b[0], b[1]))
                l.append((b[0], b[1]))
                s += X[b[0]][b[1]]
                vis[b[0]][b[1]] = 1
    s = s / len(l)
    for elem in l:
        vis[elem[0]][elem[1]] = 0
    return s


def calculate_heater(times):
    if is_working(times):
        global total_energy
        max_temp = celsius_to_kelvin(20)
        for elem in heaters_use:
            if calculate_room_temperature(elem[0], elem[1]) < max_temp:
                l = [[elem[0], elem[1]]]
                q = deque()
                q.append([elem[0], elem[1]])
                s = X[elem[0]][elem[1]]
                while len(q):
                    a = q.pop()
                    for neighbour in [[1, 0], [0, 1]]:
                        b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                        if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                            continue
                        if plan[b[0]][b[1]] == 'heater':
                            q.append((b[0], b[1]))
                            l.append((b[0], b[1]))
                            s += X[b[0]][b[1]]
                s = s / len(l)
                for e in l:
                    X[e[0]][e[1]] += moc_grzejnika / (gestosc(s) * len(l) * 1005 / 10)
                    total_energy += moc_grzejnika / (gestosc(s) * len(l) * 1005 / 10)


def calculate_room():
    new = X.copy()
    for elem in rooms + heaters:
        new[elem[0], elem[1]] = X[elem[0], elem[1]] + D * ht / (h**2)*(X[elem[0]+1, elem[1]] + X[elem[0]-1, elem[1]] +
                                                                       X[elem[0], elem[1]+1] + X[elem[0], elem[1]-1] -
                                                                       4*X[elem[0], elem[1]])
    for elem in rooms + heaters:
        X[elem[0]][elem[1]] = new[elem[0]][elem[1]]


def calculate_wall():
    for e in walls + doors:
        for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            if e[0] + neigh[0] < 0 or e[0] + neigh[0] >= rows or e[1] + neigh[1] < 0 or e[1] + neigh[1] >= cols:
                continue
            if plan[e[0] + neigh[0]][e[1] + neigh[1]] == "room" or plan[e[0] + neigh[0]][e[1] + neigh[1]] == "heater":
                X[e[0]][e[1]] = X[e[0] + neigh[0]][e[1] + neigh[1]]


def calculate_outdoors(times):
    for elem in outside:
        X[elem[0]][elem[1]] = calculate_outdoor_temperature(times)


def main():
    for step in range(n):
        calculate_room()
        calculate_heater(step)
        calculate_wall()
        calculate_windows(step)
        calculate_outdoors(step)
        calculate_doors()
        animations.append(change_to_celsius(X.copy()))


def show():
    def update(frame):
        for ind, heatmap in enumerate(heatmaps):
            heatmap.set_array(animations[frame])
        return heatmaps
    plt.imshow(change_to_celsius(X))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    heatmaps = [axs.imshow(animations[0], extent=(10, -10, -10, 10), origin="upper", animated=True, vmin=5, vmax=50)]

    anim = animation.FuncAnimation(fig, update, frames=n-1, blit=True, interval=1)
    plt.tight_layout()
    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save('animacja.gif', writer=writer)
    plt.show()


main()
show()
print(total_energy)
