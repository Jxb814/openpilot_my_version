import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
import os
from tqdm.contrib.concurrent import process_map
import seaborn as sns
import statsmodels.api as sm
from scipy import signal

def process_csv_filt(csv_file): # withfiltering
    df = pd.read_csv(csv_file)
    df_not_null_CAN = df[df['LAT_ACCEL'].notnull()].copy()
    df_not_null_car = df[df['vEgo'].notnull()].copy()
    df_not_null_roll = df[df['roll'].notnull()].copy()
    df_not_null_control = df[df['currentCurvature'].notnull()].copy()
    df_not_null_driving = df[df['desiredCurvature'].notnull()].copy()

    order = 2 # filtfilt butterfilter order 2, 4, no significant effect
    fs = 100  # sampling frequency
    cutoff_freq = 1 # cutoff frequency 1,2,5 hz, lower the smoother and fitting is better
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    columns_to_filter = ['LAT_ACCEL','YAW_RATE','LONG_ACCEL']
    df_not_null_CAN.loc[:,columns_to_filter] = df_not_null_CAN[columns_to_filter].apply(lambda x: signal.filtfilt(b, a, x.values))

    columns_to_filter = ['vEgo','yawRate','steeringAngleDeg']
    df_not_null_car.loc[:,columns_to_filter] = df_not_null_car[columns_to_filter].apply(lambda x: signal.filtfilt(b, a, x.values))
    df_not_null_control.loc[:,'currentCurvature'] = signal.filtfilt(b, a, df_not_null_control['currentCurvature'].values)

    fs = 20  # Sampling frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    df_not_null_roll.loc[:,'roll'] = signal.filtfilt(b, a, df_not_null_roll['roll'].values)
    df_not_null_driving.loc[:,'desiredCurvature'] = signal.filtfilt(b, a, df_not_null_driving['desiredCurvature'].values)

    df_not_null_roll=df_not_null_roll.dropna(axis=1)
    df_not_null_car=df_not_null_car.dropna(axis=1)
    df_not_null_CAN=df_not_null_CAN.dropna(axis=1)
    df_not_null_control=df_not_null_control.dropna(axis=1)
    df_not_null_driving=df_not_null_driving.dropna(axis=1)

    df_combined = pd.concat([df_not_null_CAN.set_index('timestamp').copy(), \
                            df_not_null_car.set_index('timestamp').copy(), \
                            df_not_null_roll.set_index('timestamp').copy(), \
                            df_not_null_control.set_index('timestamp').copy(), \
                            df_not_null_driving.set_index('timestamp').copy()], axis=1)
    return df_combined

def process_csv(csv_file):  # no filtering
    df = pd.read_csv(csv_file)
    # Merge horizontally
    df_combined = pd.concat([df.set_index('timestamp')], axis=1, join='outer')
    return df_combined

sns.set_theme()
ACC_G = 9.81
wheelbase = 2.90  # Hyundai Palisade
steering_ratio = 15.6*1.15  # Hyundai Palisade ? not the same in the comma device
# calibration
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000026--c7b3f58ee5/"
# output_csv = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000026--c7b3f58ee5/combined.csv"
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000027--0019206325/"
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000028--a23c9fe785/"

# without trailer
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/"
# output_csv = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/combined.csv"

# # with trailer
input_folder = "./tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/"
output_csv = "./tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/combined.csv"

# Collect all CSV paths
csv_files = glob.glob(os.path.join(input_folder, "*_selected_data.csv"))

# Process in parallel with a progress bar
all_combined = process_map(process_csv, csv_files, max_workers=os.cpu_count())
# all_combined = process_map(process_csv_filt, csv_files, max_workers=os.cpu_count())

# # Merge all CSVs vertically, filtering first and then interpolate data
df_all = pd.concat(all_combined).sort_index()
df = df_all.interpolate(method='linear')


# df = df[(df.index-df.index[0]>110) & (df.index-df.index[0]<180)] # calibration
# df = df[(df.index>2579.439430264) & (df.index<2639.438230084)]  # towing: right and left segment 42
# df = df[(df.index>1319.40718814) & (df.index<1379.399647336)]  # towing: right-turn on ramp segment 21
# df = df[(df.index>101895) & (df.index<101938)] # no-towing: right-turn on ramp segment 53-54

# fig, ax = plt.subplots(3,1,figsize = [6,8])
# ax[0].plot(df.index-df.index[0],df['steeringAngleDeg'], color = 'orange', label = 'steeringAngle')
# ax[0].set_ylabel('steering [deg]')
# ax[0].set_xlabel('time [s]')

# ax[1].plot(df.index-df.index[0],df['vEgo'], color = 'black', label = 'vEgo')
# ax[1].set_ylabel('vEgo [m/s]')
# ax[1].set_xlabel('time [s]')

# ax[2].plot(df.index-df.index[0],df['roll']*57.3, color = 'blue', label = 'roll')
# ax[2].set_ylabel('roll [deg]')
# ax[2].set_xlabel('time [s]')

# fig.tight_layout()
# plt.show()


# fig, ax = plt.subplots(5,1,figsize = [10,10])
# ax[0].plot(df.index-df.index[0],df['LAT_ACCEL'], color = 'red', label = 'LAT_ACCEL')
# ax[0].set_ylabel('LAT_ACCEL [m/s^2]')
# ax[0].set_xlabel('time [s]')

# ax[1].plot(df.index-df.index[0],df['yawRate'], color = 'green', label = 'yawRate (car)')
# ax[1].plot(df.index-df.index[0],df['YAW_RATE'], color = 'purple', label = 'YAW_RATE (CAN)')
# ax[1].legend()
# ax[1].set_ylabel('yaw rate [deg/s]')
# ax[1].set_xlabel('time [s]')

# ax[2].plot(df.index-df.index[0],df['steeringAngleDeg'], color = 'orange', label = 'steeringAngle')
# ax[2].set_ylabel('steering [deg]')
# ax[2].set_xlabel('time [s]')

# ax[3].plot(df.index-df.index[0],df['vEgo'], color = 'black', label = 'vEgo')
# ax[3].set_ylabel('vEgo [m/s]')
# ax[3].set_xlabel('time [s]')

# ax[4].plot(df.index-df.index[0],df['roll']*57.3, color = 'blue', label = 'roll')
# ax[4].set_ylabel('roll [deg]')
# ax[4].set_xlabel('time [s]')

# fig.tight_layout()
# plt.show()

# df.to_csv(output_csv)
# print("✅ All CSVs combined and interpolated")


df['steeringRate'] = np.gradient(df['steeringAngleDeg'],df.index)
df['yawAccel'] = np.gradient(df['YAW_RATE'],df.index)
df = df[(np.abs(df['steeringRate'])<1) & (20<df['vEgo']) & (np.abs(df['LONG_ACCEL'])<0.1) ]
# df = df[(np.abs(df['steeringRate'])<1) & (20<df['vEgo']) & (np.abs(df['LONG_ACCEL'])<0.1) & (np.abs(df['yawAccel'])<0.1)]

# adjusted_steering = ACC_G*57.3*(df['steeringAngleDeg']/steering_ratio/57.3-wheelbase/df['vEgo']/df['vEgo']*(df['LAT_ACCEL']-ACC_G*np.sin(df['roll'])))
adjusted_steering = ACC_G*(df['steeringAngleDeg']/steering_ratio-wheelbase/df['vEgo']*df['YAW_RATE'])
# adjusted_steering = ACC_G*(df['steeringAngleDeg']/steering_ratio+57.3*wheelbase*df['currentCurvature'])
# adjusted_steering = ACC_G*(df['steeringAngleDeg']/steering_ratio+57.3*wheelbase*df['desiredCurvature'])

# LS fit
x = df['LAT_ACCEL']
# x = df['YAW_RATE']*df['vEgo']/57.3     # Namyang

coef = np.polyfit(x,adjusted_steering,1)
print(coef)
fitted_line = np.polyval(coef,x)

# Calculate the correlation matrix
corr_matrix = np.corrcoef(x, adjusted_steering)

# The R-value is the off-diagonal element
r_value = corr_matrix[0, 1]

print(f"R-value: {r_value}")

matplotlib.use("TkAgg")
plt.figure(figsize = [6,5])
plt.scatter(x/ACC_G, adjusted_steering/ACC_G, s = 2)
plt.plot(x/ACC_G,fitted_line/ACC_G, color = 'red', label = 'Linear Fit')
plt.xlabel('lateral acceleration [g]')
plt.ylabel('adjusted steering angle [deg]')
plt.ylim(-1.5,1.5)
plt.xlim(-0.5,0.5)
plt.show()

# # RLS fit
# df['adjustedSteer'] = adjusted_steering
# y = df['adjustedSteer']/ACC_G
# X = sm.add_constant(df['LAT_ACCEL'])/ACC_G
# X = X.rename(columns={'LAT_ACCEL': 'K_us'})

# # X = sm.add_constant(df['YAW_RATE']*df['vEgo']/57.3)   # Namyang
# mod = sm.RecursiveLS(y,X)
# res = mod.fit()
# print(res.summary())
# res.plot_recursive_coefficient([0,1],alpha = None, figsize = (6,4),legend_loc='lower right')
# plt.show()
