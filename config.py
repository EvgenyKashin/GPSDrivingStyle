# for better result
DELETE_NOISE = True

# if time step between two point greater this value, this point will not be
# inclute to calclulating acceleration
NOISE_INTERVAL = 2

# for debug information about deleted noise point
PRINT_NOISE_POINT = False

# if speed equal to 0 this value time in a row, it will be count as a stop
STOP_TIME = 2

# if time step without data will be greater this value, all data will move to
# left on this value
EMPTY_INTERVAL = 3

# for debug information
PRINT_ERROR_POINT = False

# all final parameters will be rounded
ROUND_NDIGITS = 3

# min data points with speed value to parse, if less - raise exception
MIN_POINTS = 20

# columns of csv file
FEATURES_COLUMNS = ['id', 'pos_acc_mean', 'neg_acc_mean', 'distance',
                    'total_time', 'max_acc', 'min_acc', 'gps_err',
                    'stop_count', 'avg_speed', 'aggressive']

CSV_PATH = './data.csv'
