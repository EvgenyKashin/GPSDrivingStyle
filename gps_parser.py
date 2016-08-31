import pynmea2
import matplotlib.pyplot as plt
from datetime import datetime, date
import numpy as np
import seaborn as sns
import pandas as pd
import config
import sys


def speedtime_from_nmea_VTG(filename):
    """NMEA:
    GPVTG speed, GPRMC short all info (but knots).
    This method don't contain the iformation about time."""

    nmea_file = open(filename)
    lines = nmea_file.readlines()
    parsed_lines = list(map(pynmea2.parse, lines))
    vtg_lines = filter(lambda x: isinstance(x, pynmea2.VTG),
                       parsed_lines)
    vtg_lines = list(filter(lambda x: x.spd_over_grnd_kmph is not None,
                            vtg_lines))  # clear data
    speeds = [x.spd_over_grnd_kmph for x in vtg_lines]
    if len(speeds) < config.MIN_POINTS:
        raise Exception('Too many points')
    times = list(range(len(speeds)))
    nmea_file.close()
    return speeds, times


def speed_time_from_nmea_RMC(filename):
    """NMEA:
    GPVTG speed, GPRMC short all info (but knots)"""

    with open(filename) as nmea_file:
        lines = nmea_file.readlines()
        parsed_lines = list(map(pynmea2.parse, lines))
        rmc_lines = filter(lambda x: isinstance(x, pynmea2.RMC), parsed_lines)
        # data cleaning
        rmc_lines = list(filter(lambda x: x.spd_over_grnd is not None,
                                rmc_lines))
        # from knots to km/h
        speeds = [x.spd_over_grnd * 1.852 for x in rmc_lines]
        if len(speeds) < config.MIN_POINTS:
            raise Exception('Too many points')
        start_time = datetime.combine(date.today(), rmc_lines[0].timestamp)
        times = [(datetime.combine(date.today(),
                                   x.timestamp) - start_time).seconds for
                 x in rmc_lines]

        """
        If there are two speed value in one second, move time value
        for one speed by 0.5 seconds
        """
        for i in range(1, len(times)):
            if times[i - 1] == times[i]:
                times[i] += 0.5
        fix_data_error(speeds, times, config.EMPTY_INTERVAL,
                       config.PRINT_ERROR_POINT)
        # delete_stops(speeds, times)
        # print(times)
        # print('error count: ', error_count)
    return speeds, times


def fix_data_error(speeds, times, empty_interval, print_err):
    """
    If there is big range without data, we move times value to left
    and leave only empty second wich serve indicator of error in
    get_acceleration method
    """
    offset = 0
    i = 1
    l = len(times)
    error_count = 0
    while i < l:
        if (times[i] + offset) - times[i - 1] > empty_interval:
            if print_err:
                print("Error in: ", times[i - 1], "sec")
            error_count += 1
            offset -= (times[i] + offset) - times[i - 1] - (empty_interval + 1)
        times[i] += offset
        i += 1
    return error_count


def get_acceleration(speeds, times, delete_noise, noise_interval):
    acceleration = []
    i = 1
    l = len(times)
    err = 0
    while i < l:
        # delete one value of noise to avoid high values in acceleration
        if delete_noise and times[i] - times[i - 1] > noise_interval:
            time = times.pop(i - 1)
            speed = speeds.pop(i - 1)
            if config.PRINT_NOISE_POINT:
                print('Noise. Time: {}s and speed: {:.2f}'.format(time, speed))
            l -= 1
            err += 1
            continue
        acceleration.append((speeds[i] - speeds[i-1])/(times[i] - times[i-1]))
        i += 1
    speeds.pop(0)
    times.pop(0)
    return acceleration, err


def get_distance(speeds, times):
    distance = 0
    for i in range(1, len(times)):
        distance += speeds[i] * (times[i] - times[i - 1])
    distance *= 10/36  # from km/h to m/c
    return distance


def get_total_time(times):
    total_time = times[-1] - times[0]
    return total_time


def get_stops_info(speeds, stop_time, with_info=False):
    # if with_info=True return additional information about stops:
    # list of tuples of index stop_start, stop_end
    # else only stop_count

    stop_count = 0
    stop_start = 0
    stop_length = 0
    info = []

    for i in range(len(speeds)):
        if speeds[i] == 0:
            if stop_length == 0:
                stop_start = i
            stop_length += 1
        elif stop_length > stop_time:
            stop_count += 1
            stop_length = 0
            info.append((stop_start, i))
    if with_info:
        return stop_count, info
    else:
        return stop_count


def delete_stops(speeds, times, stoptime=2):
    stop_count = 0
    stop_start = 0
    stop_length = 0
    i = 0
    l = len(speeds)
    offset = 0
    while i < l:
        if speeds[i] == 0:
            if stop_length == 0:
                stop_start = i
            stop_length += 1
        else:
            if stop_length > stoptime:
                stop_count += 1
                for i in range(stop_start + 1, stop_start + stop_length):
                    speeds.pop(stop_start + 1)
                    times.pop(stop_start + 1)
                i = stop_start + 1
                l -= stop_length - 1
                offset -= stop_length - 1
                stop_length = 0
                continue
            stop_length = 0
        times[i] += offset
        i += 1
    print("Stop count: ", stop_count)
    return stop_count


def acceleration_features(acceleration):
    # return mean of positive and negative acceleration
    pos_acc = filter(lambda x: x > 0, acceleration)
    neg_acc = filter(lambda x: x < 0, acceleration)
    pos_acc = np.array(list(pos_acc))
    neg_acc = np.array(list(neg_acc))
    return np.mean(pos_acc), np.mean(neg_acc)


def get_acceleration_maxmin(acceleration):
    return np.max(acceleration), np.min(acceleration)


def get_avg_speed(speeds):
    # only positive values counted
    clear_speed = np.array(list(filter(lambda x: x > 0, speeds)))
    return np.mean(clear_speed)


def plot_data(speeds, times, acceleration):
    f, plts = plt.subplots(2, sharex=True)
    plts[0].plot(times, speeds, label='Speed')
    plts[0].set_title("Speed")
    plts[0].set_xlabel('$sec$')
    plts[0].set_ylabel('$km/hour$')
    plts[1].plot(times, acceleration, label='Acceleration')
    plts[1].set_title("Acceleration")
    plts[1].set_xlabel('$sec$')
    plts[1].set_ylabel('$km/hour^2$')


def process_data(id):
    filename = './' + str(id) + '/data.txt'
    speeds, times = speed_time_from_nmea_RMC(filename)
    acceleration, err = get_acceleration(speeds, times, config.DELETE_NOISE,
                                         config.NOISE_INTERVAL)
    print('acc mean: pos {0[0]:.2f}, neg {0[1]:2f}'
          .format(acceleration_features(acceleration)))
    print('dist: {:.2f}'.format(get_distance(speeds, times)))
    print('time: {:.2f}'.format(get_total_time(times)))
    print('acceleration_maxmin: {0[0]:.2f}, {0[1]:.2f}'
          .format(get_acceleration_maxmin(acceleration)))
    print('gps_err: ', err)
    print('stop_count: ', get_stops_info(speeds, config.STOP_TIME))
    print('avg speed: {:.2f}'.format(get_avg_speed(speeds)))
    plot_data(speeds, times, acceleration)


def add_if_not_in(df, row):
    is_in_df = False
    for i in range(len(df)):
        if df.loc[i].id == row.id:
            is_in_df = True
            print('Row {} already in DataFrame'.format(row.id))
            break
    if not is_in_df:
        print('added id', row.id)
        df.loc[len(df)] = row


def value_from_file(id):
    filename = './' + str(id) + '/desk.txt'
    with open(filename) as f:
        s = f.readline().lower()
    if s.startswith('aggr'):
        return 1
    else:
        return 0


def add_to_csv(id):
    filename = './' + str(id) + '/data.txt'
    value = value_from_file(id)
    try:
        df = pd.read_csv(config.CSV_PATH)
        print('Loaded DataFrame')
    except Exception:
        df = pd.DataFrame(columns=config.FEATURES_COLUMNS)
        print('Created new DataFrame')
    speeds, times = speed_time_from_nmea_RMC(filename)
    acc, err = get_acceleration(speeds, times, config.DELETE_NOISE,
                                config.NOISE_INTERVAL)
    acc_feat = acceleration_features(acc)
    acc_maxmin = get_acceleration_maxmin(acc)
    distance = get_distance(speeds, times)
    total_time = get_total_time(times)
    stops = get_stops_info(speeds, config.STOP_TIME)
    avg_speed = get_avg_speed(speeds)
    row = pd.Series((id, acc_feat[0], acc_feat[1], distance, total_time,
                     acc_maxmin[0], acc_maxmin[1], err, stops, avg_speed,
                     value), index=config.FEATURES_COLUMNS)
    row = row.round(config.ROUND_NDIGITS)
    add_if_not_in(df, row)
    df.to_csv(config.CSV_PATH, index=None)


def add_to_csv_range(r):
    # r - python range, should include id for processing
    for i in r:
        try:
            add_to_csv(i)
        except:
            print('id {} not in range'.format(i))


def pairplot_csv():
    try:
        df = pd.read_csv(config.CSV_PATH)
    except Exception:
        df = pd.DataFrame(columns=config.FEATURES_COLUMNS)
    sns.pairplot(df, vars=['pos_acc_mean', 'neg_acc_mean', 'min_acc',
                           'max_acc', 'stop_count'], hue='aggressive')


def scatter_csv():
    try:
        df = pd.read_csv(config.CSV_PATH)
    except Exception:
        df = pd.DataFrame(columns=config.FEATURES_COLUMNS)
    x_aggressive = []
    y_aggressive = []
    x_no_aggressive = []
    y_no_aggressive = []
    for val in df.values:
        if int(val[len(config.FEATURES_COLUMNS) - 1]) == 0:
            x_no_aggressive.append(val[1])
            y_no_aggressive.append(val[2])
        else:
            x_aggressive.append(val[1])
            y_aggressive.append(val[2])
    plt.scatter(x_aggressive, y_aggressive, c='red', label='aggressive')
    plt.scatter(x_no_aggressive, y_no_aggressive, c='green',
                label='No aggressive')
    plt.xlabel('Positive acceleration')
    plt.ylabel('Negative acceletation')
    plt.gca().invert_yaxis()
    plt.legend()


# process_data(22)  # show caclulated features
# add_to_csv_range(range(23))  # parsed all 23 data file and added to csv
# scatter_csv()  # show scatter plot by acceleration

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            id = int(sys.argv[1])
            process_data(id)
            print('Adding track to data.csv')
            add_to_csv(id)
        except:
            print('Wrong id! Create folder with integer id.')
