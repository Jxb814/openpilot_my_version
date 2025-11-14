import pandas as pd
from openpilot.tools.lib.logreader import LogReader
from opendbc.can.parser import CANParser
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map
from scipy import signal

def process_csv(csv_file):  # no filtering
    df = pd.read_csv(csv_file)
    # Merge horizontally
    df_combined = pd.concat([df.set_index('timestamp')], axis=1, join='outer')
    return df_combined

def process_csv_filt(csv_file): # withfiltering
    df = pd.read_csv(csv_file)
    df_not_null_CAN = df[df['LAT_ACCEL'].notnull()].copy()
    df_not_null_car = df[df['vEgo'].notnull()].copy()
    df_not_null_roll = df[df['lp.roll'].notnull()].copy()
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
    df_not_null_roll.loc[:,'lp.roll'] = signal.filtfilt(b, a, df_not_null_roll['lp.roll'].values)
    df_not_null_driving.loc[:,'desiredCurvature'] = signal.filtfilt(b, a, df_not_null_driving['desiredCurvature'].values)
    df_not_null_roll.loc[:,'lp.SF'] = df_not_null_roll['lp.SF']  # no need to filter the following values
    df_not_null_roll.loc[:,'lp.SR'] = df_not_null_roll['lp.SR']
    df_not_null_roll.loc[:,'lp.AngleOffset'] = df_not_null_roll['lp.AngleOffset']

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


# # no trailer
# # input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/"


# # with trailer
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/"

# dbc_name = "hyundai_kia_generic"
# target_signals = ["LAT_ACCEL", "LONG_ACCEL", "YAW_RATE"]
# target_message = "ESP12"
# # === Process all .zst files in the folder ===
# for input_file in glob.glob(os.path.join(input_folder, "*.zst")):
#     output_csv = os.path.splitext(input_file)[0] + "_tire_data.csv"
#     # === Load log and parser ===
#     lr = LogReader(input_file)
#     messages = [(target_message, 0)]
#     cp = CANParser(dbc_name, messages, bus=0)
#     # === Decode and collect signals in wide format ===
#     rows = []
#     for msg in lr:
#         row = {"timestamp": msg.logMonoTime / 1e9}  # convert ns → seconds
#         if msg.which() == "can":
#             frames = []
#             for i in range(len(msg.can)): # 100 Hz
#                 f = msg.can[i]
#                 frames.append((f.address, bytes(f.dat), f.src))

#             entry = (msg.logMonoTime, frames)
#             cp.update([entry])

#             if target_message in cp.vl:
#                 for sig_name in target_signals:
#                     val = cp.vl[target_message].get(sig_name, float('nan'))
#                     row[sig_name] = round(val, 5) if val is not None else float('nan')
#                 retrieved_values = [row.get(key) for key in target_signals]
#                 if any(retrieved_values):   # remove the all zero message!
#                     rows.append(row)

#         elif msg.which() == "carState":
#             row["vEgo"] = round(msg.carState.vEgo, 5)
#             row["yawRate"] = round(msg.carState.yawRate, 5)
#             row["steeringAngleDeg"] = round(msg.carState.steeringAngleDeg, 5)
#             rows.append(row)

#         elif msg.which() == 'liveParameters':
#             row["lp.roll"] = round(msg.liveParameters.roll, 5)
#             row["lp.SF"] = round(msg.liveParameters.stiffnessFactor, 5)
#             row["lp.SR"] = round(msg.liveParameters.steerRatio, 5)
#             row["lp.AngleOffset"] = round(msg.liveParameters.angleOffsetDeg, 5)
#             rows.append(row)

#         elif msg.which() == 'carControl':
#             row["currentCurvature"] = round(msg.carControl.currentCurvature, 5)
#             rows.append(row)

#         elif msg.which() == 'drivingModelData':
#             row["desiredCurvature"] = round(msg.drivingModelData.action.desiredCurvature, 5)
#             rows.append(row)

#         # elif msg.which() == 'carParams':   # not changing over time, also the time is duplicated
#         #     row["SF"] = msg.carParams.tireStiffnessFactor
#         #     row["Cf"] = msg.carParams.tireStiffnessFront
#         #     row["Cr"] = msg.carParams.tireStiffnessRear
#         #     rows.append(row)

#     # === Save to CSV, sorted by timestamp ===
#     df = pd.DataFrame(rows)
#     df = df.sort_values(by="timestamp")
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved decoded tire stiffness signals to {output_csv}")


input_folder_trailer = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/"
output_csv_all_trailer  = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/combined_tire.csv"
csv_files_trailer = glob.glob(os.path.join(input_folder_trailer, "*_tire_data.csv"))
all_combined_trailer = process_map(process_csv_filt, csv_files_trailer, max_workers=os.cpu_count())
df_all_trailer = pd.concat(all_combined_trailer).sort_index()
df_trailer = df_all_trailer.interpolate(method='linear')


input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/"
output_csv_all = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/combined_tire.csv"
csv_files = glob.glob(os.path.join(input_folder, "*_tire_data.csv"))
all_combined = process_map(process_csv_filt, csv_files, max_workers=os.cpu_count())
df_all = pd.concat(all_combined).sort_index()
df = df_all.interpolate(method='linear')

# df.to_csv(output_csv_all)
# df_trailer.to_csv(output_csv_all_trailer)
# print("✅ All CSVs combined and interpolated")

# fig, ax = plt.subplots(3,1,figsize = [6,8])
# ax[0].plot(df.index-df.index[0],df['lp.SF'], color = 'black', label = 'no trailer')
# ax[0].plot(df_trailer.index-df_trailer.index[0],df_trailer['lp.SF'], color = 'red', label = 'with trailer')
# ax[0].legend()
# ax[0].set_ylabel('Stiffness factor')
# ax[0].set_xlabel('time [s]')

# ax[1].plot(df.index-df.index[0],df['vEgo'], color = 'black')
# ax[1].plot(df_trailer.index-df_trailer.index[0],df_trailer['vEgo'], color = 'red')
# ax[1].set_ylabel('vEgo [m/s]')
# ax[1].set_xlabel('time [s]')

# ax[2].plot(df.index-df.index[0],df['steeringAngleDeg'], color = 'black', label = 'SA')
# ax[2].plot(df_trailer.index-df_trailer.index[0],df_trailer['steeringAngleDeg'], color = 'red', label = 'SA')
# ax[2].set_ylabel('Steering wheel angle [deg]')
# ax[2].set_xlabel('time [s]')

# fig.tight_layout()
# plt.show()

df['steeringRate'] = np.gradient(df['steeringAngleDeg'],df.index)
a = 1.16  # front distance to CG
b = 1.74  # rear distance to CG
L = 2.9   # wheelbase
Cf = 176779.3 # nominal tire stiffness front
Cr = 186301.4 # nominal tire stiffness rear
SR = 15.6*1.15  # openpilot steering ratio with coefficient, lp.steerRatio
sf = 0.63 # openpilot uses this to obtain Cf and Cr for cars other than Civic, already considered in Cr & Cr
m = 2135 # mass, kg
mf = b/L*m # front axle mass
mr = a/L*m # rear axle mass
G = 9.81  # gravity
Kus = (mf/Cf - mr/Cr) * G * 57.3
print("Calculated Kus (nominal)", Kus)

# fig, ax = plt.subplots(3,1,figsize = [6,8])
# ax[0].plot(df.index-df.index[0],(mf/Cf - mr/Cr) * G * 57.3/df['lp.SF'], color = 'black', label = 'no trailer')
# # ax[0].plot(df_trailer.index-df_trailer.index[0],df_trailer['lp.SF'], color = 'red', label = 'with trailer')
# ax[0].legend()
# ax[0].set_ylabel('K_us [deg/g]')
# ax[0].set_xlabel('time [s]')

# ax[1].plot(df.index-df.index[0],df['vEgo'], color = 'black')
# # ax[1].plot(df_trailer.index-df_trailer.index[0],df_trailer['vEgo'], color = 'red')
# ax[1].set_ylabel('vEgo [m/s]')
# ax[1].set_xlabel('time [s]')

# ax[2].plot(df.index-df.index[0],df['steeringAngleDeg'], color = 'black', label = 'SA')
# # ax[2].plot(df_trailer.index-df_trailer.index[0],df_trailer['steeringAngleDeg'], color = 'red', label = 'SA')
# ax[2].set_ylabel('Steering wheel angle [deg]')
# ax[2].set_xlabel('time [s]')

# fig.tight_layout()
# plt.show()

num_params = 2
theta = np.zeros((num_params,1))  # initial guess
theta[0,0]=1.5
P = 100*np.eye(num_params)
lam = 1.0

num_samples = len(df.index)
theta_history = np.full((num_params, num_samples),np.nan)
vmin = 20
max_steerrate = 1
max_longaccel = 0.1
for k in range(num_samples):
    if df['vEgo'].iloc[k] > vmin and np.abs(df['steeringRate'].iloc[k])<max_steerrate and np.abs(df['LONG_ACCEL'].iloc[k])<max_longaccel:
        # Current input and output
        yt = df['steeringAngleDeg'].iloc[k]/SR-L/df['vEgo'].iloc[k]*df['YAW_RATE'].iloc[k]
        ut = df['LAT_ACCEL'].iloc[k]/G
        # Form regressor vector
        H = np.array([[ut], [1.0]])

        # Prediction error
        e = yt - (H.T @ theta)[0, 0]

        # Gain vector
        K = P @ H / (lam + (H.T @ P @ H)[0, 0])

        # Update parameter estimates
        theta = theta + K * e

        # Update covariance
        P = (P - K @ H.T @ P) / lam

        # Store history
        theta_history[:, k] = theta.flatten()

fig, ax = plt.subplots(2,1,figsize = [6,8])
ax[0].plot(df.index-df.index[0],(mf/Cf - mr/Cr) * G * 57.3/df['lp.SF'],color = 'black', label = 'OP')
ax[0].scatter(df.index-df.index[0], theta_history[0, :],  s=3, color = 'blue', label = 'RLS')
ax[0].legend()
ax[0].set_ylabel('understeer gradient [deg/g]')
ax[0].set_ylim(0,5)
ax[0].set_xlabel('time [s]')

ax[1].plot(df.index-df.index[0],df['lp.AngleOffset'],color = 'black', label = 'OP')
ax[1].scatter(df.index-df.index[0], theta_history[1, :], s=3, color = 'blue', label='RLS')
ax[1].legend()
ax[1].set_ylabel('steering angle offset [deg]')
ax[1].set_xlabel('time [s]')
ax[1].set_ylim(-2,2)

fig.tight_layout()
plt.show()

# the estimation will not be correct because the mass is not updated with trailer weight
