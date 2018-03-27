# Standard modules
import os
import itertools
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') #used to be mpl.use('macosx')
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import progressbar
from scipy.io.idl import readsav
from sunpy.util.metadata import MetaDict
import sys

# Custom modules
from jpm_logger import JpmLogger
from jpm_number_printing import latex_float
# from get_goes_flare_events import get_goes_flare_events  # TODO: Uncomment once sunpy method implemented
from determine_preflare_irradiance import determine_preflare_irradiance
from light_curve_peak_match_subtract import light_curve_peak_match_subtract
from automatic_fit_light_curve import automatic_fit_light_curve
from determine_dimming_depth import determine_dimming_depth
from determine_dimming_slope import determine_dimming_slope
from determine_dimming_duration import determine_dimming_duration

__author__ = 'James Paul Mason (and Shawn Polson and Tyler Albee)'
__contact__ = 'shpo9723@colorado.edu'


def generate_cme_signature(start_timestamp='2010-08-07 17:12:11',
                          end_timestamp='2010-08-07 21:18:11',
                          output_path='/Users/shawnpolson/Documents/School/Spring 2018/Data Mining/StealthCMEs/PyCharm/JEDI Catalog/',
                          verbose=True):
    """Wrapper code for generating the dimming depth, duration, and slope for one CME event.

    Inputs:
        None.

    Optional Inputs:
        start_timestamp [str]:                                  A timestamp for the beginning of an event.
        end_timestamp [str]:                                    A timestamp for the end of an event.
        flare_index_range [range]                               The range of GOES flare indices to process. Default is range(0, 5052).
        output_path [str]:                                      Set to a path for saving the JEDI catalog table and processing
                                                                summary plots. Default is '/Users/shawnpolson/Documents/School/Spring 2018/Data Mining/StealthCMEs/PyCharm/JEDI Catalog/'.
        verbose [bool]:                                         Set to log the processing messages to disk and console. Default is False.

    Outputs:
        No direct return, but writes a (csv? sql table? hdf5?) to disk with the dimming paramerization results.
        Subroutines also optionally save processing plots to disk in output_path.

    Optional Outputs:
        None

    Example:
        generate_cme_signature(verbose=True, start_timestamp='2010-08-07 17:12:11', end_timestamp='2010-08-07 23:18:11')

    """
    # Prepare the logger for verbose
    if verbose:
        logger = JpmLogger(filename='generate_shawns_jedi_catalog', path=output_path, console=True)
        logger.info("Starting CME signature processing pipeline.")
    else:
        logger = None

    # Get EVE level 2 extracted emission lines data
    # Load up the actual irradiance data into a pandas DataFrame
    # Declare that column 0 is the index then convert it to datetime
    eve_lines = pd.read_csv('/Users/shawnpolson/Documents/School/Spring 2018/Data Mining/StealthCMEs/savesets/eve_selected_lines.csv', index_col=0)
    eve_lines.index = pd.to_datetime(eve_lines.index)

    if verbose:
        logger.info('Loaded EVE data')

    # Define the columns of the JEDI catalog
    jedi_row = pd.DataFrame([OrderedDict([
                             ('Event #', np.nan),
                             ('Start Time', np.nan),
                             ('End Time', np.nan),
                             ('GOES Flare Class', np.nan),
                             ('Pre-Flare Start Time', np.nan),
                             ('Pre-Flare End Time', np.nan),
                             ('Flare Interrupt', np.nan)])])
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Pre-Flare Irradiance [W/m2]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Min [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Max [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Mean [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Uncertainty [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth Uncertainty [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Fitting Gamma'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Fitting Score'))

    ion_tuples = list(itertools.permutations(eve_lines.columns.values, 2))
    ion_permutations = pd.Index([' by '.join(ion_tuples[i]) for i in range(len(ion_tuples))])

    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Min [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Max [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Mean [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Uncertainty [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth Uncertainty [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Correction Time Shift [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Correction Scale Factor'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Fitting Gamma'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Fitting Score'))

    csv_filename = output_path + 'jedi_{0}.csv'.format(Time.now().iso)
    jedi_row.to_csv(csv_filename, header=True, index=False, mode='w')

    if verbose:
        logger.info('Created JEDI row definition.')

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]

    # Prepare a hold-over pre-flare irradiance value,
    # which will normally have one element for each of the 39 emission lines
    preflare_irradiance = np.nan

    # TODO: Now that we have our 6 selected lines, for the time range of our one example CME (2010/08/07 17:12-21:18), smooth them out, and run them through James's routines to produce the "signature" of the CME.
    # Note: See this link if James's "eve_lines[start:end]" syntax is desired: https://stackoverflow.com/questions/16175874/python-pandas-dataframe-slicing-by-date-conditions  (Note we get KeyError if requested times in this range don't exist exactly)
    # Get only rows in our dimming window
    startTime = pd.to_datetime(start_timestamp) # default value is '2010-08-07 17:12:11'
    endTime = pd.to_datetime(end_timestamp)     # default value is '2010-08-07 21:18:11'
    eve_lines_event = eve_lines.loc[(eve_lines.index >= startTime) & (eve_lines.index <= endTime)] # this syntax is more forgiving than "eve_lines[startTime:endTime]"
    #print(eve_lines_event.head)

    if verbose:
        logger.info('Sliced rows in dimming window time range: ' + start_timestamp + ' -> ' + end_timestamp)
        logger.info("Event {0} EVE data clipped to dimming window.".format(1))

    # Fill the event information into the JEDI row
    jedi_row['Event #'] = 1
    jedi_row['Start Time'] = start_timestamp
    jedi_row['End Time'] = end_timestamp
    if verbose:
        logger.info("Event {0} details stored to JEDI row.".format(1))

    # Convert irradiance units to percent
    # (in place, don't care about absolute units from this point forward)
    # Note: "preflare_irradiance" is pandas series with columns for each line and just one irradiance (float) per column
    preflare_irradiance = eve_lines_event.iloc[0]
    eve_lines_event_percentages = (eve_lines_event - preflare_irradiance) / preflare_irradiance * 100.0

    if verbose:
        logger.info("Event {0} irradiance converted from absolute to percent units.".format(1))

    # Fit the light curves to reduce influence of noise on the parameterizations to come later
    uncertainty = np.ones(len(eve_lines_event_percentages)) * 0.002545  # got this line from James's code

    progress_bar_fitting = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Light curve fitting: ')] + widgets,
                                                   max_value=len(eve_lines_event_percentages.columns)).start()
    for i, column in enumerate(eve_lines_event_percentages):
        if eve_lines_event_percentages[column].isnull().all().all():
            if verbose:
                logger.info(
                    'Event {0} {1} fitting skipped because all irradiances are NaN.'.format(1, column))
        else:
            eve_line_event_percentages = pd.DataFrame(eve_lines_event_percentages[column])
            eve_line_event_percentages.columns = ['irradiance']
            eve_line_event_percentages['uncertainty'] = uncertainty

            fitting_path = output_path + 'Fitting/'
            if not os.path.exists(fitting_path):
                os.makedirs(fitting_path)

            plt.close('all')
            light_curve_fit, best_fit_gamma, best_fit_score = automatic_fit_light_curve(eve_line_event_percentages,
                                                                                        plots_save_path='{0} Event {1} {2} '.format(
                                                                                            fitting_path, 1,
                                                                                            column),
                                                                                        verbose=verbose, logger=logger)
            eve_lines_event_percentages[column] = light_curve_fit
            jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
            jedi_row[column + ' Fitting Score'] = best_fit_score

            if verbose:
                logger.info('Event {0} {1} light curves fitted.'.format(1, column))
            progress_bar_fitting.update(i)

    progress_bar_fitting.finish()

    if verbose:
        logger.info('Light curves fitted')
        #print(eve_lines_event_percentages.head)



if __name__ == '__main__':
    generate_cme_signature(verbose=True, start_timestamp='2010-08-07 17:12:11', end_timestamp='2010-08-07 23:18:11')
