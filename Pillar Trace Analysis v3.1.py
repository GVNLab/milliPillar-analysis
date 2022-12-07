# %% [markdown]
# # milliPillar Pillar Trace Analysis
# 
# ## Version 3.1
# 
# ## Last Updated: 10/24/2022
# 

# %% [markdown]
# ## Import Libraries

# %%
# Import necessary packages
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
import matplotlib.image 
import matplotlib.patches
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import scipy.signal
from pathlib import Path  
import warnings
from pandas import ExcelWriter
import ipykernel

warnings.filterwarnings("ignore", category=RuntimeWarning)    

# %%
# Select Data to Analyze 
main_folder = ''

# Set Frame Rate in FPS
frame_rate = 20

# Set Pixel to Micron Conversion Factor
microns_per_pixel = 6.5

# Set the Unloaded Distance Between the Pillar Heads (in microns)
unloaded_distance = 3000

# Set Elastic Coefficient in uN/um
coefficient = 2.1 

# Can change this to select from a pop up window

# %%
# Load Regimen Excel File
regimen_file = ''
regimen = pd.read_excel(regimen_file)

# %%
# Makes sure all CSV files are in the proper configuration 

# Create a List of .csv Files
csv_list = glob.glob(main_folder + "*.csv")
csv_names = []
for x in csv_list: 
    csv_names.append(os.path.splitext(os.path.basename(x))[0])
    
csv_left = glob.glob(main_folder + "*_left.csv")
csv_left_names = []
for x in csv_left: 
    csv_left_names.append(os.path.splitext(os.path.basename(x))[0])
    
csv_right = glob.glob(main_folder + "*_right.csv")
csv_right_names = []
for x in csv_right: 
    csv_right_names.append(os.path.splitext(os.path.basename(x))[0])

# Create a list of .csv files in the correct format 
with_left = []
for x in csv_left_names: 
    with_left.append(os.path.basename(x).replace("_left", ""))
with_right = []
for x in csv_right_names: 
    with_right.append(os.path.basename(x).replace("_right", ""))

correct_csv = list(set.intersection(set(with_left), set(with_right)))

# Corrects remaining .csv files and adds them to the list 
to_process = list((set(csv_names) - set(csv_right_names)) - set(csv_left_names))

for x in to_process: 
    data = data = pd.read_csv(os.path.join(main_folder, x + '.csv'))

    left_x = data.loc[:, 'Left X']
    left_y = data.loc[:, 'Left Y']
    right_x = data.loc[:, 'Right X']
    right_y = data.loc[:, 'Right Y'] 

    left = {'X' : left_x, 'Y': left_y}
    left_df = pd.DataFrame(data=left)
    right = {'X' : right_x, 'Y' : right_y}
    right_df = pd.DataFrame(data=right) 

    left_name = os.path.join(main_folder, x + "_left.csv")
    right_name = os.path.join(main_folder, x + "_right.csv")

    left_df.to_csv(left_name)
    right_df.to_csv(right_name)

    correct_csv.append(x)

# Makes list unique and sort
correct_csv = list(set(correct_csv))
correct_csv.sort()


# %%
# Load Tissue Widths and Calculate XC Areas
# Note: Width Input File Should be in Pixels 
width_file = os.path.join(main_folder, 'Widths', "TissueWidths.csv")
widths = pd.read_csv(width_file)
width_microns = widths['Width'] * microns_per_pixel
widths['Width (um)'] = width_microns 
widths['Width (mm)'] = width_microns / 1000 
areas = (np.square(((widths['Width (mm)'])/2))) * np.pi
widths['Area (mm2)'] = areas 


# %%
def load_trace(current_tissue):

    # Display Tissue Name
    print("Processing Tissue {}".format(current_tissue))

    # Load Tissue Trace
    left_file = os.path.join(main_folder, current_tissue + "_left.csv")
    right_file = os.path.join(main_folder, current_tissue + "_right.csv")
    left_data = pd.read_csv(left_file)
    right_data = pd.read_csv(right_file) 

    # Create New Dataframe 
    left_x = left_data.loc[:, 'X']
    left_y = left_data.loc[:, 'Y']
    right_x = right_data.loc[:, 'X']
    right_y = right_data.loc[:, 'Y'] 
    data = {'Left X':left_x, 'Left Y':left_y, 'Right X':right_x, 'Right Y':right_y}
    data = pd.DataFrame(data=data)


    # Add Time Column to DF
    time = np.arange(0,(data.shape[0])/frame_rate,(1/frame_rate))
    data['Time'] = time

    #Add distance to DF
    x_diff =  right_x - left_x
    y_diff = right_y - left_y 
    x_diff_sq = np.square(x_diff)
    y_diff_sq = np.square(y_diff)
    sum_sq = x_diff_sq + y_diff_sq
    distance = np.sqrt(sum_sq)
    data['Distance'] = distance * microns_per_pixel
    
    # Add Deflection Column to DF
    # Note: deflection refers to the total pillar deflection
    deflection = unloaded_distance - (distance * microns_per_pixel)
    data['Deflection'] = deflection

    # Plot Deflection
    fig, ax1 = plt.subplots(1, figsize=(20, 10))
    fig.suptitle(current_tissue)
    ax1.plot(data['Time'], data['Deflection'])
    ax1.set_ylabel("Deflection (Microns)")
    ax1.set_xlabel("Time (s)")
    
    # Select Spontaneous Beating Region
    sp_start = regimen.loc[regimen['Region'] == "Spontaneous", 'First Frame'][0]
    sp_stop = regimen.loc[regimen['Region'] == "Spontaneous", 'Last Frame'][0]
    spontaneous = data.iloc[sp_start:sp_stop]
    x = spontaneous.loc[:,'Deflection']

    # Determine Baseline Correction Based on Minima for Each Second in Spontaneous Region
    seconds = np.arange(sp_start, sp_stop, frame_rate)
    num = len(seconds)
    mins_deflection = np.zeros(num)
    idxs = np.zeros(num)
    maxs_distance = np.zeros(num)
    for i, x in enumerate(seconds):
        min_val = spontaneous.loc[x:(x+frame_rate)].loc[:,'Deflection'].min()
        min_index = spontaneous.loc[x:(x+frame_rate)].loc[:,'Deflection'].idxmin()
        max_distance = spontaneous.loc[x:(x+frame_rate)].loc[:,'Distance'].max()
        mins_deflection[i] = min_val
        idxs[i] = min_index
        maxs_distance[i] = max_distance

    baseline_deflection_avg = np.mean(mins_deflection)
    baseline_distance_avg = np.mean(maxs_distance)

    # Calculate Active Deflection Based on Baseline
    active_deflection = data.loc[:,'Deflection'] - baseline_deflection_avg
    data['Active Deflection'] = active_deflection

    # Reformat Tissue Name
    current_tissue = current_tissue.replace('.', '')

    # Calculate Force and Stress
    data['Active Force (uN)'] = active_deflection * coefficient 
    data['Force (uN)'] = deflection * coefficient
    area = widths[widths['Tissue'] == current_tissue]['Area (mm2)']
    area = area.to_numpy()[0]
    width = widths[widths['Tissue'] == current_tissue]['Width (um)']
    width = width.to_numpy()[0]
    data['Stress (mN/mm2)'] = ((data['Force (uN)'])/1000) / area
    data['Active Stress (mN/mm2)'] = (data['Active Force (uN)']/1000) / area
    

    # Define Passive Tension and Length
    passive_tension = baseline_deflection_avg * coefficient
    passive_stress = passive_tension / area
    passive_length = baseline_distance_avg

    # Calculate Velocity and Add to Data
    velocity = (pd.Series(data=deflection).diff())/(1/frame_rate)
    data['Velocity'] = velocity

    # Differentiate Contraction and Relaxation Velocity 
    data['Contraction Velocity'] = velocity
    data.loc[(data['Contraction Velocity']<0),'Contraction Velocity']=0

    data['Relaxation Velocity'] = velocity 
    data.loc[(data['Relaxation Velocity']>0),'Relaxation Velocity']=0
    data['Relaxation Velocity'] = abs(data['Relaxation Velocity'])

    # Replace all NaNs with 0 
    data = data.fillna(0)

    # Calculate Max Values 
    max_contraction_velocity = max(data['Contraction Velocity'])
    max_relaxation_velocity = max(data['Relaxation Velocity'])
    max_force = max(data['Force (uN)'])
    max_active_force = max(data['Active Force (uN)'])
    max_stress = max(data['Stress (mN/mm2)'])
    max_active_stress = max(data['Active Stress (mN/mm2)'])

    # Create Dict for Results 

    results = { 'Tissue':[current_tissue], 
                'Passive Tension (uN)':[passive_tension], 
                'Passive Stress (uN/mm2)': [passive_stress], 
                'Passive Length (um)':[passive_length], 
                'Max Contraction Velocity (um/s)':[max_contraction_velocity], 
                'Max Relaxation Velocity (um/s)':[max_relaxation_velocity], 
                'Max Force (uN)':[max_force], 
                'Max Active Force (uN)':[max_active_force], 
                'Max Stress (mN/mm2)':[max_stress], 
                'Max Active Stress (mN/mm2)':[max_active_stress], 
                'Width':[width], 
                'XC Area':[area]}

    return data, results, fig
    

# %%
def region_contractility(data, tissue, start, stop, set_freq, region_name):
    
    # Select and Plot Trace Region
    region = data.iloc[start:stop]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
    fig.suptitle('Tissue {} {} Region'.format(tissue, region_name))
    ax1.plot(region['Time'], region['Active Force (uN)'])
    ax1.set_ylabel("Active Force (uN)")
    ax1.set_xlabel("Time (s)")

    # Find Range of Signal in Region 
    max_region = max(region.loc[:,'Active Force (uN)'])
    min_region = min(region.loc[:,'Active Force (uN)'])
    range_region = max_region - min_region

    # Find Max Force and Stress
    max_force = max(region['Force (uN)']) 
    max_active_force = max(region['Active Force (uN)'])
    max_stress = max(region['Stress (mN/mm2)'])
    max_active_stress = max(region['Active Stress (mN/mm2)'])

    # Change set_freq for regions without stimulation 
    if set_freq == 0: 
        set_freq = 2
    else: 
        set_freq = set_freq

    # Find Peaks in Region 
    x = region['Active Force (uN)']
    peaks, properties = scipy.signal.find_peaks(x, distance= ((1/set_freq) * frame_rate)*0.9, height=0, prominence = (range_region/4))

    # Create DataFrame to Store Contraction Peak Info
    peak_info = pd.DataFrame(properties)
    peak_info.insert(0, "Location", peaks)

    # Exclude Peaks That Overlap Ends of Video
    min_cutoff = int(frame_rate)
    max_cutoff = len(x) - int(frame_rate)
    peak_info = peak_info.drop(peak_info[(peak_info['Location'] < min_cutoff) | (peak_info['Location'] >max_cutoff)].index.to_list(), axis=0)
    peaks = np.array(peak_info['Location'])
    num_peaks = len(peaks)
    
    fw90m_left = scipy.signal.peak_widths(x, peaks, rel_height=0.9)[2] * (1/frame_rate)
    fw90m_right = scipy.signal.peak_widths(x, peaks, rel_height=0.9)[3] * (1/frame_rate)

    left_bases = np.around(scipy.signal.peak_widths(x, peaks, rel_height=0.9)[2])
    right_bases = np.around(scipy.signal.peak_widths(x, peaks, rel_height=0.9)[3])
    
    # Plot Region Trace with Overlaid Peaks 
    ax1.plot((peaks + start)/frame_rate, x[peaks + start], "X")
    ax1.vlines(x=(peaks + start)/frame_rate, ymin=x[peaks + start] - np.array(peak_info['prominences']), ymax = x[peaks + start])
    ax1.hlines(y=x[peaks + start] - (0.9 * np.array(peak_info['prominences'])), xmin=(fw90m_left + (start/frame_rate)), xmax=(fw90m_right + (start/frame_rate)))


    # Determine the Force and Stress of Each Peak in the Region
    peak_info['Active Force'] = (x[peaks + start]).to_list()
    x_force = region['Force (uN)']
    peak_info['Force'] = x_force[peaks + start].to_list()
    x_stress = region['Stress (mN/mm2)']
    peak_info['Stress'] = x_stress[peaks + start].to_list()
    x_active_stress = region['Active Stress (mN/mm2)']
    peak_info['Active Stress'] = x_active_stress[peaks + start].to_list()

    # Repeat Peak Finding for Velocity

    # Plot Contraction Velocity
    ax2.plot(region['Time'], region['Velocity'])
    ax2.set_ylabel("Velocity (Microns/S)")
    ax2.set_xlabel("Time (s)")

    # Find Range of Velocity in Region 
    max_contraction_velocity = max(region.loc[:,'Contraction Velocity'])
    min_contraction_velocity = min(region.loc[:,'Contraction Velocity'])
    range_region = max_contraction_velocity - min_contraction_velocity

    # Select Velocity Based on Max Velocity During Contract 90
    con_vel_peaks = [] 
    for i, p in enumerate(peaks):
        vel = np.array(region['Contraction Velocity'][int(left_bases[i]):(p+1)])
        vel_peak = np.argmax(vel) + int(left_bases[i]) 
        con_vel_peaks.append(vel_peak)

    con_vel_peaks = np.array(con_vel_peaks)
    
    # Plot Region Trace with Overlaid Peaks 
    
    ax2.plot((con_vel_peaks + start)/frame_rate, region['Contraction Velocity'][start + con_vel_peaks], "X")

    # Add Contraction Velocity Peaks to Peak Info DF
    peak_info['Contraction Velocity'] = region['Contraction Velocity'][start + con_vel_peaks].to_list()

    #Relaxation Velocity

    # Find Range of Relaxation Velocity in Region 
    max_relaxation_velocity = max(region.loc[:,'Relaxation Velocity'])
    min_relaxation_velocity = min(region.loc[:,'Relaxation Velocity'])
    range_region = max_relaxation_velocity - min_relaxation_velocity

    # Select Velocity Based on Max Velocity During Contract 90
    relax_vel_peaks = [] 
    for i, p in enumerate(peaks):
        vel = np.array(region['Relaxation Velocity'][p:(1 + int(right_bases[i]))])
        vel_peak = np.argmax(vel) + p
        relax_vel_peaks.append(vel_peak)

    relax_vel_peaks = np.array(relax_vel_peaks)
    
    # Plot Region Trace with Overlaid Peaks 
    ax2.plot((relax_vel_peaks + start)/frame_rate, -1 *(region['Relaxation Velocity'][start + relax_vel_peaks]), "X")

    # Add Relaxation Velocity Peaks to Peak Info DF
    peak_info['Relaxation Velocity'] = region['Relaxation Velocity'][start + relax_vel_peaks].to_list()

    # Determine Region Beat Frequency
    front = peaks[1:num_peaks]
    back = peaks[0:(num_peaks-1)]
    periods = front - back 
    periods = periods / frame_rate
    delta_periods = periods[1:len(periods)] - periods[0:(len(periods) - 1)]

    # Remove Outlier from RR Intervals to Correct Calculated Frequency
    periods = pd.Series(periods)
    stat = cb.boxplot_stats(periods)
    periods = periods.replace(stat[0]['fliers'], np.nan)
    freqs = np.reciprocal(periods)
    freq = np.mean(freqs)

    # Characterize Beat Variability
    rr_interval = np.mean(periods)
    sdrr = np.std(periods)
    rmssd = np.sqrt(np.mean(delta_periods))
    
    # Determine Max Frequency in Region 
    if len(freqs) > 0:
        max_freq = freqs.max()
    else: 
        max_freq = 0

    # Remove Outliers from Peak Metrics 
    metrics = ['Force', 'Active Force', 'Stress', 'Active Stress', 'Contraction Velocity', 'Relaxation Velocity']
    no_out = pd.DataFrame()
    for i in metrics: 
        stat = cb.boxplot_stats(peak_info[peak_info[i].notna()][i])
        print('Tissue {} {} Region {} Outliers: {}'.format(tissue, region_name, i, stat[0]['fliers']))
        new_variable = peak_info[i].replace(stat[0]['fliers'], np.nan)
        no_out[i] = new_variable

    print('')
    print('**************************************')
    print('')

    # Calculate Averages After Outlier Removal
    mean_force = np.mean(no_out['Force'])
    mean_stress = np.mean(no_out['Stress'])
    mean_active_force = np.mean(no_out['Active Force'])
    mean_active_stress = np.mean(no_out['Active Stress'])
    mean_contraction_velocity = np.mean(no_out['Contraction Velocity'])
    mean_relaxation_velocity = np.mean(no_out['Relaxation Velocity'])

    # Create Dictionary of Results

    results = { 'Tissue':[tissue], 
                'Frequency': [freq], 
                'Max Frequency': [max_freq], 
                'Mean Force': [mean_force], 
                'Max Force': [max_force],
                'Mean Stress': [mean_stress], 
                'Max Stress': [max_stress], 
                'Mean Active Force': [mean_active_force], 
                'Max Active Force': [max_active_force], 
                'Mean Active Stress': [mean_active_stress], 
                'Max Active Stress': [max_active_stress], 
                'Mean Contraction Velocity':[mean_contraction_velocity],
                'Max Contraction Velocity':[max_contraction_velocity], 
                'Mean Relaxation Velocity' : [mean_relaxation_velocity], 
                'Max Relaxation Velocity':[max_relaxation_velocity],
                'RR Interval':[rr_interval], 
                'SDRR':[sdrr], 
                'RMSSD':[rmssd] }
    
    # Returns the Results Dict
    return results, fig


# %%
results_dict = {}
region_names = [] 
count = 0     

with matplotlib.backends.backend_pdf.PdfPages(main_folder + 'Compiled Traces.pdf') as pdf: 
    for x in correct_csv:    
        tissue = x 
        data, results, fig = load_trace(tissue)

        # Save Figure to PDF
        pdf.savefig(fig)

        if count==0:
            results_overall = pd.DataFrame(results);
        else: 
            results_to_add = pd.DataFrame(results)
            results_overall = pd.concat([results_overall, results_to_add]);
        
        del results

        for i, row in regimen.iterrows():
            start = int(row['First Frame'])
            stop = int(row['Last Frame'])
            set_freq = row['Freq (Hz)']
            region_name = (str(row['Region']) + " " + str(set_freq) +  " Hz")

            results, fig = region_contractility(data, tissue, start, stop, set_freq, region_name)

            # Save Figure to PDF
            pdf.savefig(fig)
            
            region_results = pd.DataFrame(results)

            if count == 0: 
                results_dict[region_name] = region_results
                region_names.append(region_name)
            else: 
                results_dict[region_name] = pd.concat([results_dict[region_name], region_results])

        # Change count flag     
        count = 1



# %%
# Save results to Excel sheet
xlsx_file = os.path.join(main_folder, "Analysis.xlsx")
writer = ExcelWriter(xlsx_file)

# Save 'Overall' results to excel 
results_overall.to_excel(writer, sheet_name = 'Overall')

# Cycle through results for each region 
print(region_names)
for x in region_names:
    results_dict[x].to_excel(writer, sheet_name = x)
    
writer.save()



