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
    hdulist = fits.open(fits_files[0])
    data = np.copy(hdulist[0].data)
    for i in range(1,n):
      data += fits.open(fits_files[i])[0].data

    data_mean = data/n
    return data_mean
      

def get_argmax_fits(fits_file):
  """get index of highest value in a fits file
  """
  hdulist = fits.open(fits_file)
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
    hdulist = fits.open(fitsfile)
    fits_list.append(hdulist[0].data)
    hdulist.close()

  # stack fits arrays in 3D arrays
  fits_stack = np.dstack(fits_list)
  median = np.median(fits_stack, axis=2)

  # calculate the memory consumed by the data (kB)
  memory = fits_stack.nbytes / 1024

  return median



if __name__ == '__main__':
  pass


  fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
  m = np.mean(fluxes)
  m2 = sum(fluxes)/len(fluxes)
  print(m, m2)