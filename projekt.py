import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import importlib.util
import time
start_time = time.time()


h = 0.2
ht = 0.4            # 0.4 sekundy
T = 3600 * 24       # dzień
n = int(T/ht)

moc_grzejnika = 2000
rows, cols = 60, 55
D = 0.025
total_energy = 0
start_hour = 12


def gestosc(_):
    return 1.1225


def calculate_time(t):
    t = t % int((60 * 60 * 24) / ht)
    return int(t*ht), t//(int(3600 / ht)), int(int(((t * ht) % 3600) / 60))  # czas w sekundach, godzinach, minutach


def calculate_outdoor_temperature(times):
    return outdoors_temperature[times//int(3600/ht)][1]


def is_closed(times):
    t, hours, minutes = calculate_time(times)
    if (hours == 6 and minutes < 5) or (hours == 16 and minutes >= 55) or (hours == 20 and minutes >= 55):
        return False
    return True


def is_working(times):
    t, hours, minutes = calculate_time(times)
    if hours in [6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        return True
    return False


class Tile:
    def __init__(self, x, y, temp):
        self.x = x
        self.y = y
        self.temp = temp

    def calculate_tile(self, times):
        pass


class Room(Tile):
    def calculate_tile(self, times):
        new[self.x, self.y] = (X[self.x, self.y] + D * ht / (h ** 2) *
                               (X[self.x + 1, self.y] + X[self.x - 1, self.y] + X[self.x, self.y + 1] +
                                X[self.x, self.y - 1] - 4 * X[self.x, self.y]))


class Wall(Tile):
    def calculate_tile(self, times):
        for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            if self.x + neigh[0] < 0 or self.x + neigh[0] >= rows or self.y + neigh[1] < 0 or self.y + neigh[1] >= cols:
                continue
            if plan[self.x+neigh[0]][self.y+neigh[1]] in ["room", "heater"]:
                new[self.x][self.y] = X[self.x + neigh[0]][self.y + neigh[1]]
                return
        for neigh in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            if self.x + neigh[0] < 0 or self.x + neigh[0] >= rows or self.y + neigh[1] < 0 or self.y + neigh[1] >= cols:
                continue
            if plan[self.x+neigh[0]][self.y+neigh[1]] in ["door"]:
                new[self.x][self.y] = X[self.x + neigh[0]][self.y + neigh[1]]
                return
        for neigh in [[1, 0], [0, 1], [0, -1], [-1, 0]]:
            if self.x + neigh[0] < 0 or self.x + neigh[0] >= rows or self.y + neigh[1] < 0 or self.y + neigh[1] >= cols:
                continue
            if plan[self.x+neigh[0]][self.y+neigh[1]] in ["wall"]:
                new[self.x][self.y] = X[self.x + neigh[0]][self.y + neigh[1]]
                return


class Door(Tile):
    def calculate_tile(self, times):
        if not vis[self.x][self.y]:
            l = [[self.x, self.y]]
            q = deque()
            q.append([self.x, self.y])
            s = 0
            vis[self.x][self.y] = 1
            new[self.x, self.y] = (X[self.x, self.y] + D * ht / (h ** 2) *
                                   (X[self.x + 1, self.y] + X[self.x - 1, self.y] + X[self.x, self.y + 1] +
                                    X[self.x, self.y - 1] - 4 * X[self.x, self.y]))
            while len(q):
                a = q.pop()
                s += new[a[0]][a[1]]
                for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                    if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                        continue
                    if plan[b[0]][b[1]] == 'door' and not vis[b[0]][b[1]]:
                        q.append((b[0], b[1]))
                        l.append((b[0], b[1]))
                        vis[b[0]][b[1]] = 1
                        new[b[0], b[1]] = (X[b[0], b[1]] + D * ht / (h ** 2) *
                                           (X[b[0] + 1, b[1]] + X[b[0] - 1, b[1]] + X[b[0], b[1] + 1] +
                                            X[b[0], b[1] - 1] - 4 * X[b[0], b[1]]))
            s = s / len(l)
            for [i, j] in l:
                new[i][j] = s

        X[self.x, self.y] = (X[self.x, self.y] + D * ht / (h ** 2) *
                             (X[self.x + 1, self.y] + X[self.x - 1, self.y] + X[self.x, self.y + 1] +
                              X[self.x, self.y - 1] - 4 * X[self.x, self.y]))


class Window(Tile):
    def calculate_tile(self, times):
        if is_closed(times):
            trans = 0
            for neighbour in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                b = [self.x + neighbour[0], self.y + neighbour[1]]
                if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                    continue
                if plan[b[0]][b[1]] == 'heater' or plan[b[0]][b[1]] == 'room':
                    X[self.x][self.y] = trans * calculate_outdoor_temperature(times) + (1 - trans) * X[b[0]][b[1]]
        else:
            X[self.x][self.y] = calculate_outdoor_temperature(times)


class Heater(Tile):
    def calculate_room_temperature(self):
        q = deque()
        q.append([self.x, self.y])
        s = X[self.x][self.y]
        l = [(self.x, self.y)]
        vis[self.x][self.y] = 1
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
            vis[elem[0]][elem[1]] = s
            pom[elem[0]][elem[1]] = len(l)
        return s

    def calculate_tile(self, times):
        if is_working(times) and not vis[self.x][self.y]:
            global total_energy
            max_temp = celsius_to_kelvin(20)
            average_temperature = vis[self.x][self.y] if vis[self.x][self.y] else self.calculate_room_temperature()
            if average_temperature < max_temp:
                l = [[self.x, self.y]]
                q = deque()
                q.append([self.x, self.y])
                while len(q):
                    a = q.pop()
                    for neighbour in [[1, 0], [0, 1]]:
                        b = [a[0] + neighbour[0], a[1] + neighbour[1]]
                        if b[0] < 0 or b[0] >= rows or b[1] < 0 or b[1] >= cols:
                            continue
                        if plan[b[0]][b[1]] == 'heater':
                            q.append((b[0], b[1]))
                            l.append((b[0], b[1]))
                for e in l:
                    total_energy += ht * moc_grzejnika / (1.1225 * pom[e[0]][e[1]] * h**2 * 1005)
                    pom2[e[0]][e[1]] = ht * moc_grzejnika / (1.1225 * pom[e[0]][e[1]] * h**2 * 1005)
        new[self.x, self.y] = (X[self.x, self.y] + D * ht / (h ** 2) *
                               (X[self.x + 1, self.y] + X[self.x - 1, self.y] + X[self.x, self.y + 1] +
                                X[self.x, self.y - 1] - 4 * X[self.x, self.y]) + pom2[self.x][self.y])


class Outdoors(Tile):
    def calculate_tile(self, times):
        X[self.x][self.y] = calculate_outdoor_temperature(times)


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
        "latitude": 49.53,
        "longitude": 19.08,
        "start_date": "2024-12-01",
        "end_date": "2024-12-02",
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


def change_to_celsius(t):
    for i in range(rows):
        for j in range(cols):
            t[i][j] = kelvin_to_celsius(X[i][j])
    return t


windows = []
walls = []
doors = []
heaters = []
outside = []
rooms = []
animations = []
outdoors_temperature = setup_temperature()
X_class = [[Tile(i, j, celsius_to_kelvin(starting_temperature(i, j))) for i in range(cols)] for j in range(rows)]
vis = np.zeros((rows, cols))
pom = [[0 for i in range(cols)] for j in range(rows)]
pom2 = [[0 for i in range(cols)] for j in range(rows)]


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


def main():
    animations.append(change_to_celsius(X.copy()))
    for step in range(int(3600 * start_hour / ht), int(3600 * start_hour / ht) + n):
        for i in range(rows):
            for j in range(cols):
                X_class[i][j].calculate_tile(step)
        for [i, j] in rooms + heaters + doors + walls:
            X[i, j] = new[i, j]
        for [i, j] in rooms + doors + heaters:
            vis[i, j] = 0
            pom[i][j] = 0
        for [i, j] in heaters:
            pom2[i][j] = 0
        if step % 100 == 0:
            animations.append(change_to_celsius(X.copy()))


def show(save=""):
    def update(frame):
        if frame >= len(animations):
            print(f"Błąd: Próba dostępu do klatki {frame}, która nie istnieje (długość animations: {len(animations)})")
            return
        for ind, heatmap in enumerate(heatmaps):
            heatmap.set_array(animations[frame])  # Aktualizacja danych w heatmapie
        return heatmaps

    # print("Czas generowania wykresu:", time.time() - start_time)
    plt.figure()
    plt.imshow(change_to_celsius(X))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.draw()
    plt.pause(0.001)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    heatmaps = [axs.imshow(animations[0], extent=(10, -10, -10, 10), origin="upper", animated=True, vmin=13, vmax=22)]
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel("")
    axs.set_ylabel("")

    anim = animation.FuncAnimation(fig, update, frames=len(animations), blit=True, interval=100)  # 100 ms na klatkę
    # writer = animation.PillowWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save(f'animacja_{save}.gif', writer=writer)
    plt.tight_layout()
    plt.pause(0.001)
    return anim


plt.ion()
for address in ["main"]:
    animations = []
    file_path = f"./create_plan_{address}.py"
    """ Ważne aby wszystkie pliki były w jednym folderze"""
    spec = importlib.util.spec_from_file_location("create_plan", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.create_plan()
    for i in range(rows):
        for j in range(cols):
            if plan[i][j] == "outside":
                outside.append([i, j])
                X_class[i][j] = Outdoors(i, j, celsius_to_kelvin(starting_temperature(i, j)))
            elif plan[i][j] == "room":
                rooms.append([i, j])
                X_class[i][j] = Room(i, j, celsius_to_kelvin(starting_temperature(i, j)))
            elif plan[i][j] == "wall":
                walls.append([i, j])
                X_class[i][j] = Wall(i, j, celsius_to_kelvin(starting_temperature(i, j)))
            elif plan[i][j] == "door":
                X_class[i][j] = Door(i, j, celsius_to_kelvin(starting_temperature(i, j)))
                doors.append([i, j])
            elif plan[i][j] == "window":
                windows.append([i, j])
                X_class[i][j] = Window(i, j, celsius_to_kelvin(starting_temperature(i, j)))
            elif plan[i][j] == "heater":
                heaters.append([i, j])
                X_class[i][j] = Heater(i, j, celsius_to_kelvin(starting_temperature(i, j)))

    X = create_tab()
    new = X.copy()

    main()
    anim = show()
    plt.pause(0.1)
    print("Energia: ", total_energy)
    total_energy = 0
    avg = 0
    for tile_x, tile_y in rooms + heaters:
        avg += X[tile_x][tile_y]

    avg = avg / len(rooms + heaters)
    print("Średnia temperatura: ", avg)

plt.ioff()
plt.show()
