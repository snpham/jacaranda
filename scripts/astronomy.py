import numpy as np
from astropy.io import fits


def calculate_mean(vals):
  """slow version to compute mean
  """
  return sum(vals)/len(vals)


def calc_stats(data):
  data = np.loadtxt(data, delimiter=',')
  mean = np.mean(data)
  median = np.median(data)
  return (mean, median)


def mean_csvs(datafiles):
  """get the element-wise mean of a list of csv files
  """
  n = len(datafiles)
  if n > 0:
    data = np.loadtxt(datafiles[0], delimiter=',')
    for i in range(1,n):
      data += np.loadtxt(datafiles[i], delimiter=',')

    mean = data/n
    return mean


def mean_fits(fits_files):
  """get the element-wise means of a list of fits files
  """
  n = len(fits_files)
  if n > 0:
    with fits.open(fits_files[0]) as hdulist:
      data = np.copy(hdulist[0].data)
      for i in range(1,n):
        data += fits.open(fits_files[i])[0].data

    data_mean = data/n
    return data_mean


def get_argmax_fits(fits_file):
  """get index of highest value in a fits file
  """
  with fits.open(fits_file) as hdulist:
    data = hdulist[0].data
    argmax = np.argwhere(data==data.max())[0]
  return tuple(argmax)


def mean_median(data):
  """get the mean and median of a list of values
  """
  mean = np.mean(data)
  data_sorted = np.sort(data)
  mid = int(len(data_sorted)/2)
  if len(data_sorted)%2 == 0:
    median = (data_sorted[mid]+data_sorted[mid-1])/2.0
  else:
    median = data_sorted[mid]
  median = round(median, 3)
  return (round(float(median), 2), mean)


def median_fits(fitsfiles):
  """get median of list of fits files using numpy stacking
  """
  # read in all the FITS files and store in list
  fits_list = []
  for fitsfile in fitsfiles: 
    with fits.open(fitsfile) as hdulist:
      fits_list.append(hdulist[0].data)

  # stack fits arrays in 3D arrays
  fits_stack = np.dstack(fits_list)
  median = np.median(fits_stack, axis=2)

  # calculate the memory consumed by the data (kB)
  memory = fits_stack.nbytes / 1024

  return median


def welfords_method(fitsfiles):
  '''Calculates the running mean and stdev for a list of FITS files 
  using Welford's method. 
  https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
  https://www.jstor.org/stable/1266577
  '''

  for n, fitsfile in enumerate(fitsfiles, 1):
    with fits.open(fitsfile) as hdulist:
      data = hdulist[0].data
      # initialize mean and std 
      if n == 1:
        mean = np.zeros_like(data)
        s = np.zeros_like(data)

      # running (x - x_bar)
      delta = data - mean
      # running x_bar
      mean += delta/n
      # running std
      s += delta*(data - mean)

  # compute total std dev
  s /= (n - 1)
  np.sqrt(s, s)

  if n < 2:
    return mean, None
  else:
    return mean, s


def median_bins(data, nbins):
  """use bin approximation method to compute median; this function 
  organizes the data samples into bins
  """
  mean = np.mean(data)
  std = np.std(data)
  left_bin = 0
  bins = np.zeros(nbins)
  bin_width = 2 * std / nbins
  
  for val in data:
    # place data points smaller than min val to ignore bin
    if val < mean-std:
      left_bin += 1
    # stores data points within range into bins
    elif val < mean+std:
      bin = int((val - (mean-std))/bin_width)
      bins[bin] += 1 
    # Ignore values above mean + std

  return mean, std, left_bin, bins


def median_approx(values, nbins):
  """use bin approximation method to compute median; this function
  calls median_bins, then approximates the median by counting 
  magnitudes of each bin in sequence.
  """
  mean, std, left_bin, bins = median_bins(values, nbins)

  # position of the middle element
  mid_count = (len(values)+1)/2

  # start with the ignored bin (val < mean-std)
  count = left_bin
  
  # add bin count to total count
  for b, bin_count in enumerate(bins):
    count += bin_count
    if count >= mid_count:
      break
  
  # get the median approximation
  width = 2 * std/nbins
  median = mean - std + width*(b+0.5)
  return median
  

def median_bins_fits(fitsfiles, nbins):
      
  # calculate the mean and standard deviation using the running method
  mean, std = welfords_method(fitsfiles)
    
  # get dimensions of the FITS file
  dim = mean.shape
  
  # initialise 3D bins
  left_bin = np.zeros(dim)
  bins = np.zeros((dim[0], dim[1], nbins))
  bin_width = 2 * std / nbins 

  # loop over all FITS files
  for fitsfile in fitsfiles:
    with fits.open(fitsfile) as hdulist:
      data = hdulist[0].data

      # loop over every point in the 2D array
      for i in range(dim[0]):
        for j in range(dim[1]):
          value = data[i, j]
          lower_bound = mean[i, j] - std[i, j]
          upper_bound = mean[i, j] + std[i, j]

          # adding to the ignored bin (val < x_bar-std)
          if value < lower_bound:
            left_bin[i, j] += 1
          # adding to the appropriate bin
          elif value >= lower_bound and value < upper_bound:
            bin = int( (value - lower_bound) / bin_width[i, j] )
            bins[i, j, bin] += 1
          # ignoring values higher than upper bound limit

  return mean, std, left_bin, bins


def median_approx_fits(filenames, nbins):
  mean, std, left_bin, bins = median_bins_fits(filenames, nbins)
    
  # get dimensions of the FITS file
  dim = mean.shape
    
  # position of the middle element over all files
  mid = (len(filenames) + 1) / 2
	
  # calculate the approximated median for each array element
  bin_width = 2 * std / nbins
  median = np.zeros(dim)   
  for i in range(dim[0]):
    for j in range(dim[1]):    

      count = left_bin[i, j]
      for b, bincount in enumerate(bins[i, j]):
        count += bincount
        if count >= mid:
          # stop when the cumulative count exceeds the midpoint
          break
      median[i, j] = mean[i, j] - std[i, j] + bin_width[i, j]*(b + 0.5)
      
  return median


def hms2dec(hours, minutes, seconds):
  """convert right ascension HMS to decimal degrees; 0 <= hours < 24;
  RA = angle from the vernal equinox to the point, going east along
  the celestial equator
  """
  return 15*(hours + minutes/60 + seconds/3600)


def dms2dec(degrees, arcminutes, arcseconds):
  """convert declination from DMS to decimal degrees; -90 <= degrees < 90;
  the anglefrom the celestial equator to the point, (+) going north
  """
  return degrees + np.sign(degrees)*(arcminutes/60 + arcseconds/3600)


def greatcirc_dist(coords1, coords2):
  """computes the great circle distance between 2 points on a sphere
  using RA and Dec; applies the haversine formula
  :param coords1: [RA, Dec] for first object
  :param coords2: [RA, Dec] for second object
  :return: angular distance (radians)
  """
  ra1 = coords1[0]
  dec1 = coords1[1]
  ra2 = coords2[0]
  dec2 = coords2[1]

  b = np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2
  a = np.sin(np.abs(dec1 - dec2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))

  return d


if __name__ == '__main__':
  pass

  fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
  m = np.mean(fluxes)
  m2 = sum(fluxes)/len(fluxes)
  print(m, m2)

  h_degs = greatcirc_dist([np.radians(21.07), np.radians(0.1)], 
                          [np.radians(21.15), np.radians(8.2)])
  assert np.allclose(np.degrees(h_degs), 8.1003923)
