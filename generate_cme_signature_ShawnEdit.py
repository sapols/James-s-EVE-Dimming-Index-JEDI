# Standard modules
import os
import itertools
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.use('macosx')
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


def generate_cme_signature(threshold_time_prior_flare_minutes=240.0,
                          dimming_window_relative_to_flare_minutes_left=0.0,
                          dimming_window_relative_to_flare_minutes_right=240.0,
                          threshold_minimum_dimming_window_minutes=120.0,
                          flare_index_range=range(0, 5052),
                          output_path='/Users/shawnpolson/Documents/School/Spring 2018/Data Mining/StealthCMEs/PyCharm/JEDI Catalog/',
                          verbose=True):
    """Wrapper code for creating James's Extreme Ultraviolet Variability Experiment (EVE) Dimming Index (JEDI) catalog.

    Inputs:
        None.

    Optional Inputs:
        threshold_time_prior_flare_minutes [float]:             How long before a particular event does the last one need to have
                                                                occurred to be considered independent. If the previous one was too
                                                                recent, will use that event's pre-flare irradiance.
                                                                Default is 240 (4 hours).
        dimming_window_relative_to_flare_minutes_left [float]:  Defines the left side of the time window to search for dimming
                                                                relative to the GOES/XRS flare peak. Negative numbers mean
                                                                minutes prior to the flare peak. Default is 0.0.
        dimming_window_relative_to_flare_minutes_right [float]: Defines the right side of the time window to search for dimming
                                                                relative to the GOES/XRS flare peak. If another flare
                                                                occurs before this, that time will define the end of the
                                                                window instead. Default is 240 (4 hours).
        threshold_minimum_dimming_window_minutes [float]:       The smallest allowed time window in which to search for dimming.
                                                                Default is 120.
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
        generate_jedi_catalog(output_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
                              verbose=True)
    """
    # Prepare the logger for verbose
    if verbose:
        logger = JpmLogger(filename='generate_shawns_jedi_catalog', path=output_path, console=True)
        logger.info("Starting JEDI processing pipeline.")
    else:
        logger = None

    # Get EVE level 2 extracted emission lines data
    # Load up the actual irradiance data into a pandas DataFrame
    eve_lines = pd.read_csv('/Users/shawnpolson/Documents/School/Spring 2018/Data Mining/StealthCMEs/savesets/eve_selected_lines.csv')
    print(eve_lines)

    if verbose:
        logger.info('Loaded EVE data')

    # Define the columns of the JEDI catalog
    jedi_row = pd.DataFrame([OrderedDict([
                             ('Event #', np.nan),
                             ('GOES Flare Start Time', np.nan),
                             ('GOES Flare Peak Time', np.nan),
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
    # Note this breaks when flare_index_range doesn't span at least 2 things
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Flare Event Loop: ')] + widgets,
                                           min_value=flare_index_range[0], max_value=flare_index_range[-1]).start()

    # Prepare a hold-over pre-flare irradiance value,
    # which will normally have one element for each of the 39 emission lines
    preflare_irradiance = np.nan

    # TODO: Now that we have our 6 selected lines, for the time range of our one example CME (2010/08/07 17:12-21:18), smooth them out, and run them through James's routines to produce the "signature" of the CME.

    # Start loop through all flares
    for curve_time in flare_index_range:

        progress_bar.update(curve_time)

    progress_bar.finish()


if __name__ == '__main__':
    generate_cme_signature(verbose=True, flare_index_range=range(1, 3))
