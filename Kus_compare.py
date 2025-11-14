import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
from tqdm.contrib.concurrent import process_map
import seaborn as sns
import statsmodels.api as sm
from scipy import signal

def comma_process_csv_filt(csv_file): # withfiltering
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


def process_csv_filt(csv_file): # withfiltering
    df = pd.read_csv(csv_file)
    #df.set_index('time')

    order = 2 # filtfilt butterfilter order 2, 4, no significant effect
    fs = 100  # sampling frequency
    cutoff_freq = 1 # cutoff frequency 1,2,5 hz, lower the smoother and fitting is better
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    columns_to_filter = ['v_CAN','SteerAg_CAN','Yaw_rate_IMU','Long_Accel_CAN','Lat_Accel_CAN']
    df.loc[:,columns_to_filter] = df[columns_to_filter].apply(lambda x: signal.filtfilt(b, a, x.values))
    df['v_CAN'] = df['v_CAN']*0.277778 # km/h to m/s
    df['steeringRate'] = np.gradient(df['SteerAg_CAN'],df['time'])
    df['yawAccel'] = np.gradient(df['Yaw_rate_IMU'],df['time'])
    df_combined = pd.concat([df], axis=1, join='outer')
    return df_combined

def process_csv(csv_file):  # no filtering
    df = pd.read_csv(csv_file)
    df['v_CAN'] = df['v_CAN']*0.277778 # km/h to m/s
    df['steeringRate'] = np.gradient(df['SteerAg_CAN'],df['time'])
    df['yawAccel'] = np.gradient(df['Yaw_rate_IMU'],df['time'])
    # Merge horizontally
    df_combined = pd.concat([df], axis=1, join='outer')
    return df_combined

sns.set_theme()
wheelbase = 2.90
steering_ratio = 15.6
ACC_G = 9.81

input_folder ="/home/hatci/openpilot/Telluride_test_data/CCW"   # first
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
all_combined = process_map(comma_process_csv_filt, csv_files, max_workers=os.cpu_count())
df_all = pd.concat(all_combined).sort_index()
df_CCW = df_all.interpolate(method='linear')
df_CCW = df_CCW[(18240 <df_CCW.index) & (df_CCW.index<19155)]     #  segment 28 has a bad point!!! interpolation made it worse
df_CCW = df_CCW[np.abs(1/df_CCW['currentCurvature'])<25]  # gear shift changed sth which affected the curvature estimation
df_CCW['steeringRate'] = np.gradient(df_CCW['steeringAngleDeg'],df_CCW.index)
df_CCW['yawAccel'] = np.gradient(df_CCW['YAW_RATE'],df_CCW.index)

input_folder ="/home/hatci/openpilot/Telluride_test_data/CW"
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
all_combined = process_map(comma_process_csv_filt, csv_files, max_workers=os.cpu_count())
df_all = pd.concat(all_combined).sort_index()
df_CW = df_all.interpolate(method='linear')
df_CW = df_CW[(df_CW.index>19572) & (df_CW.index<20312)]
df_CW['steeringRate'] = np.gradient(df_CW['steeringAngleDeg'],df_CW.index)
df_CW['yawAccel'] = np.gradient(df_CW['YAW_RATE'],df_CW.index)
df_comma = pd.concat([df_CCW,df_CW])

# df_comma = df_comma[np.abs(df_comma['vEgo'])>1]

plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],-1/df_CCW['currentCurvature'], color = 'red', label = 'from curvature')
plt.plot(df_CW.index-df_CCW.index[0],-1/df_CW['currentCurvature'], color = 'red')
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['vEgo']/df_CCW['YAW_RATE']*57.3, color = 'blue', label = 'from yawrate')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['vEgo']/df_CW['YAW_RATE']*57.3,  color = 'blue')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Radius [m]')
plt.tight_layout()

# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['vEgo'], label = 'Comma 3x')
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['vEgo'])
# plt.legend()
# plt.xlabel('time')
# plt.ylabel('velocity [m/s]')
# plt.tight_layout()

# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['LAT_ACCEL']/ACC_G)
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['LAT_ACCEL']/ACC_G)
# plt.xlabel('time')
# plt.ylabel('lateral acceleration [g]')
# plt.tight_layout()


# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['steeringAngleDeg']/steering_ratio)
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['steeringAngleDeg']/steering_ratio)
# plt.xlabel('time')
# plt.ylabel('steering angle [deg]')
# plt.tight_layout()

# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['YAW_RATE'])
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['YAW_RATE'])
# plt.xlabel('time')
# plt.ylabel('yaw rate [deg/s]')
# plt.tight_layout()

# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['YAW_RATE']/df_CCW['vEgo']*wheelbase)
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['YAW_RATE']/df_CW['vEgo']*wheelbase)
# plt.plot(df_CCW.index-df_CCW.index[0],-57.3*df_CCW['currentCurvature']*wheelbase)
# plt.plot(df_CW.index-df_CCW.index[0],-57.3*df_CW['currentCurvature']*wheelbase)
# plt.xlabel('time')
# plt.ylabel('required steering angle [deg]')
# plt.tight_layout()
# plt.show()

# df_CCW = df_CCW[np.abs(df_CCW['vEgo'])>7]
# df_CW = df_CW[np.abs(df_CW['vEgo'])>7]
# df_comma = df_comma[np.abs(df_comma['vEgo'])>7]
# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],57.3*wheelbase*df_CCW['LAT_ACCEL']/df_CCW['vEgo']/df_CCW['vEgo'])
# plt.plot(df_CW.index-df_CCW.index[0],57.3*wheelbase*df_CW['LAT_ACCEL']/df_CW['vEgo']/df_CW['vEgo'])
# plt.xlabel('time')
# plt.ylabel('required steering angle [deg]')
# plt.tight_layout()

# plt.figure(figsize = [6,3])
# plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['roll']*57.3)
# plt.plot(df_CW.index-df_CCW.index[0],df_CW['roll']*57.3)
# plt.xlabel('time [s]')
# plt.ylabel('roll [deg]')
# plt.tight_layout()

# LS fit
adjusted_steering = ACC_G*(df_comma['steeringAngleDeg']/steering_ratio-wheelbase/df_comma['vEgo']*df_comma['YAW_RATE'])
x1 = df_CCW['LAT_ACCEL']
y1 = ACC_G*(df_CCW['steeringAngleDeg']/steering_ratio-wheelbase/df_CCW['vEgo']*df_CCW['YAW_RATE'])
x2 = df_CW['LAT_ACCEL']
y2 = ACC_G*(df_CW['steeringAngleDeg']/steering_ratio-wheelbase/df_CW['vEgo']*df_CW['YAW_RATE'])

# adjusted_steering = ACC_G*(df_comma['steeringAngleDeg']/steering_ratio+57.3*wheelbase*df_comma['currentCurvature'])
# x1 = df_CCW['LAT_ACCEL']
# y1 = ACC_G*(df_CCW['steeringAngleDeg']/steering_ratio+57.3*wheelbase*df_CCW['currentCurvature'])
# x2 = df_CW['LAT_ACCEL']
# y2 = ACC_G*(df_CW['steeringAngleDeg']/steering_ratio+57.3*wheelbase*df_CW['currentCurvature'])

# adjusted_steering = ACC_G*57.3*(df_comma['steeringAngleDeg']/steering_ratio/57.3-wheelbase/df_comma['vEgo']/df_comma['vEgo']*(df_comma['LAT_ACCEL']-ACC_G*np.sin(df_comma['roll'])))
# x1 = df_CCW['LAT_ACCEL']
# y1 = ACC_G*57.3*(df_CCW['steeringAngleDeg']/steering_ratio/57.3-wheelbase/df_CCW['vEgo']/df_CCW['vEgo']*(df_CCW['LAT_ACCEL']-ACC_G*np.sin(df_CCW['roll'])))
# x2 = df_CW['LAT_ACCEL']
# y2 = ACC_G*57.3*(df_CW['steeringAngleDeg']/steering_ratio/57.3-wheelbase/df_CW['vEgo']/df_CW['vEgo']*(df_CW['LAT_ACCEL']-ACC_G*np.sin(df_CW['roll'])))

# adjusted_steering = ACC_G*(df_comma['steeringAngleDeg']/steering_ratio-wheelbase/df_comma['vEgo']*df_comma['YAW_RATE'])
# x1 = df_CCW['YAW_RATE']*df_CCW['vEgo']/57.3
# y1 = ACC_G*(df_CCW['steeringAngleDeg']/steering_ratio-wheelbase/df_CCW['vEgo']*df_CCW['YAW_RATE'])
# x2 = df_CW['YAW_RATE']*df_CW['vEgo']/57.3
# y2 = ACC_G*(df_CW['steeringAngleDeg']/steering_ratio-wheelbase/df_CW['vEgo']*df_CW['YAW_RATE'])

# x = df_comma['LAT_ACCEL']
# y = adjusted_steering

coef = np.polyfit(x1,y1,1)
print(coef)
fitted_line1 = np.polyval(coef,x1)
# Calculate the correlation matrix
corr_matrix1 = np.corrcoef(x1, y1)
# The R-value is the off-diagonal element
r_value1 = corr_matrix1[0, 1]
print(f"R-value: {r_value1}")

coef = np.polyfit(x2,y2,1)
print(coef)
fitted_line2 = np.polyval(coef,x2)
# Calculate the correlation matrix
corr_matrix2 = np.corrcoef(x2, y2)
# The R-value is the off-diagonal element
r_value2 = corr_matrix2[0, 1]
print(f"R-value: {r_value2}")

plt.figure(figsize = [6,5])
plt.scatter(df_comma['LAT_ACCEL']/ACC_G, adjusted_steering/ACC_G, s = 2)
plt.plot(x1/ACC_G,fitted_line1/ACC_G, color = 'red', label = 'Linear Fit')
plt.plot(x2/ACC_G,fitted_line2/ACC_G, color = 'red', label = 'Linear Fit')
plt.xlabel('lateral acceleration [g]')
plt.ylabel('adjusted steering angle [deg]')
plt.ylim(-2.5,2.5)
plt.xlim(-0.5,0.5)
plt.title('Comma')
plt.tight_layout()
# plt.show()

input_folder ="/home/hatci/openpilot/Telluride_test_data/"
# CCW: counterclockwise, left-turn, positive steering
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
all_combined = process_map(process_csv_filt, csv_files, max_workers=os.cpu_count())
all_combined[0] = all_combined[0][(201.5<all_combined[0]['time']) & (all_combined[0]['time']<1116.5) & (all_combined[0]['v_CAN']>1)]
all_combined[1] = all_combined[1][(201<all_combined[1]['time']) &(all_combined[1]['time']<941) & (all_combined[1]['v_CAN']>1)]
# all_combined = process_map(process_csv, csv_files, max_workers=os.cpu_count())
df = pd.concat(all_combined)
# df = df[np.abs(df['v_CAN'])>1]
# df = df[(np.abs(df['steeringRate'])<1) & (np.abs(df['Long_Accel_CAN'])<0.5) & (np.abs(df['yawAccel'])<0.5) & (np.abs(df['v_CAN'])>1)]  # & (np.abs(df['Lat_Accel_CAN'])>1)


plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],-1/df_CCW['currentCurvature'], color = 'red', label = 'from curvature')
plt.plot(df_CW.index-df_CCW.index[0],-1/df_CW['currentCurvature'], color = 'red')
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['vEgo']/df_CCW['YAW_RATE']*57.3, color = 'blue', label = 'from yawrate')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['vEgo']/df_CW['YAW_RATE']*57.3,  color = 'blue')
plt.plot(all_combined[0] ['time']-201.5,all_combined[0]['v_CAN']/all_combined[0]['Yaw_rate_IMU']*57.3, color = 'green', label = 'from IMU')
plt.plot(all_combined[1] ['time']+1331-200,all_combined[1]['v_CAN']/all_combined[1]['Yaw_rate_IMU']*57.3, color = 'green')
plt.legend()
plt.xlabel('time')
plt.ylabel('Yaw rate [m/s^2]')
plt.tight_layout()
plt.show()


'''
plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['vEgo'], color = 'black', label = 'Comma 3x')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['vEgo'], color = 'black')
plt.plot(all_combined[0] ['time']-201.5,all_combined[0] ['v_CAN'], color = 'green', label = 'IMU')
plt.plot(all_combined[1] ['time']+1331-200,all_combined[1] ['v_CAN'], color = 'green')
plt.legend()
plt.xlabel('time')
plt.ylabel('velocity [m/s]')
plt.tight_layout()
# plt.show()

plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['LAT_ACCEL'], color = 'black', label = 'Comma 3x')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['LAT_ACCEL'], color = 'black')
plt.plot(all_combined[0] ['time']-201.5,all_combined[0] ['Lat_Accel_CAN'], color = 'green', label = 'IMU')
plt.plot(all_combined[1] ['time']+1331-200,all_combined[1] ['Lat_Accel_CAN'], color = 'green')
plt.legend()
plt.xlabel('time')
plt.ylabel('Lat Accel [m/s^2]')
plt.tight_layout()

plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['YAW_RATE'], color = 'black', label = 'Comma 3x')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['YAW_RATE'], color = 'black')
plt.plot(all_combined[0] ['time']-201.5,all_combined[0] ['Yaw_rate_IMU'], color = 'green', label = 'IMU')
plt.plot(all_combined[1] ['time']+1331-200,all_combined[1] ['Yaw_rate_IMU'], color = 'green')
plt.legend()
plt.xlabel('time')
plt.ylabel('Yaw rate [m/s^2]')
plt.tight_layout()

plt.figure(figsize = [6,3])
plt.plot(df_CCW.index-df_CCW.index[0],df_CCW['steeringAngleDeg'], color = 'black', label = 'Comma 3x')
plt.plot(df_CW.index-df_CCW.index[0],df_CW['steeringAngleDeg'], color = 'black')
plt.plot(all_combined[0] ['time']-201.5,all_combined[0] ['SteerAg_CAN'], color = 'green', label = 'IMU')
plt.plot(all_combined[1] ['time']+1331-200,all_combined[1] ['SteerAg_CAN'], color = 'green')
plt.legend()
plt.xlabel('time')
plt.ylabel('Steering angle [deg]')
plt.tight_layout()


adjusted_steering = ACC_G*(df['SteerAg_CAN']/steering_ratio-wheelbase/df['v_CAN']*df['Yaw_rate_IMU'])
x = df['Lat_Accel_CAN']
y = adjusted_steering
x1 = all_combined[0]['Lat_Accel_CAN']
y1 = ACC_G*(all_combined[0]['SteerAg_CAN']/steering_ratio-wheelbase/all_combined[0]['v_CAN']*all_combined[0]['Yaw_rate_IMU'])
x2 = all_combined[1]['Lat_Accel_CAN']
y2 = ACC_G*(all_combined[1]['SteerAg_CAN']/steering_ratio-wheelbase/all_combined[1]['v_CAN']*all_combined[1]['Yaw_rate_IMU'])

coef1 = np.polyfit(x1,y1,1)
print(coef1)
fitted_line1 = np.polyval(coef1,x1)
# Calculate the correlation matrix
corr_matrix1 = np.corrcoef(x1, y1)
# The R-value is the off-diagonal element
r_value1 = corr_matrix1[0, 1]
print(f"R-value: {r_value1}")

coef2 = np.polyfit(x2,y2,1)
print(coef2)
fitted_line2 = np.polyval(coef2,x2)
# Calculate the correlation matrix
corr_matrix2 = np.corrcoef(x2, y2)
# The R-value is the off-diagonal element
r_value2 = corr_matrix2[0, 1]
print(f"R-value: {r_value2}")


plt.figure(figsize = [6,5])
plt.scatter(x/ACC_G, y/ACC_G, s = 2)
plt.plot(x1/ACC_G,fitted_line1/ACC_G, color = 'red', label = 'Linear Fit')
plt.plot(x2/ACC_G,fitted_line2/ACC_G, color = 'red', label = 'Linear Fit')
plt.xlabel('lateral acceleration [g]')
plt.title('IMU')
plt.ylabel('adjusted steering angle [deg]')
plt.ylim(-2.5,2.5)
plt.xlim(-0.5,0.5)
plt.tight_layout()
plt.show()

# # RLS fit
# df['adjustedSteer'] = adjusted_steering
# y = df['adjustedSteer']
# X = sm.add_constant(df['Lat_Accel_CAN'])
# X = X.rename(columns={'Lat_Accel_CAN': 'K_us'})

# # X = sm.add_constant(df['YAW_RATE']*df['vEgo']/57.3)   # Namyang
# mod = sm.RecursiveLS(y,X)
# res = mod.fit()
# print(res.summary())
# res.plot_recursive_coefficient([0,1],alpha = None, figsize = (6,4),legend_loc='lower right')
# plt.show()
'''