import matplotlib.pyplot as plt

def plot_exposure_level(exposures, catalog):
    total_num_plots = len(exposures)

    columns = 3
    rows = total_num_plots // columns

    if total_num_plots % columns != 0:
        rows += 1
    
    position = range(1, total_num_plots + 1)

    fig = plt.figure(1)
    
    stars_in_sus_reg = []
    for exposure, plot_position in zip(exposures, range(total_num_plots)):
      for angle in exposure.sweeps:
        if True in exposure.sweeps[angle]['targets_in']:
            
      # add every single subplot to the figure with a for loop
      ax = fig.add_subplot(rows, columns, position[plot_position])
      ax.scatter(catalog['ra'], catalog['dec'])

    plt.show()

for obs_id in program.exposure_frames:
    for exposure in program.exposure_frames[obs_id]:
        for angle in program.exposure_frames[obs_id][exposure].sweeps:
            if False in program.exposure_frames[1][1].sweeps[angle]['targets_in']:
                 for i, index in enumerate(program.exposure_frames[1][1].sweeps[angle]['targets_in']):
                     if index is True:
                         print(program.catalog.iloc[program.exposure_frames[1][1].sweeps[angle]['targets_loc'][i]])
                     else:
                         print('There are no stars in sus reg')