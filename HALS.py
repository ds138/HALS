# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Wed Jul  1 16:14:38 2020

@author: Daniel
"""

class hals_lsq:
    def __init__(self, time, y, freqs):
        """
        Generalized harmonic least squares amplitude and phase estimation method.
            
        Parameters
        ----------
        time : N x 1 numpy array
            Sample times.
        y : N x 1 numpy array
            Sample values.
        freqs : array_like
            Known frequencies of the sinusoids
        
        Returns
        -------
        alpha_est : array_like
            estimated amplitudes of the sinusoids.
    
        phi_est : array_like
            estimated phases of the sinusoids.
        error_variance : array_like
            variance of the error. MSE of the reconstructed signal compared to y.
        theta :
            parameters such that ||y - Phi*theta|| is minimized, where Phi is the
            matrix defined by freqs and x that when multiplied by theta is a sum of sinusoids.
            
        Notes
        -----
        An optimisation approach which aims to minimise the sum of the squared residuals when fitting
        sinusoids of known frequencies to a time series of discrete measurements 
        
        References
        ----------
        ..[1]   Stoica P, Moses R L, others (2005) Spectral analysis of signals. 
                Upper Saddle River, New Jersey 07458:Pearson Prentice Hall, ISBN 0-13-113956
        """    
    
        N = y.shape[0]
        f = np.array(freqs)*2*np.pi
        num_freqs = len(f)
        Phi = np.empty((N, 2*num_freqs))
        Sins = np.zeros((N,num_freqs))
        Coss = np.zeros((N,num_freqs))
        for i in range(N):
            for j in range(num_freqs):
                Sins[i,j] = np.sin(f[j]*time[i])
                Coss[i,j] = np.cos(f[j]*time[i])
    
        Phi[:,::2] = Sins
        Phi[:,1::2] = Coss
        Phi = np.hstack((Phi,np.ones((N,1))))
        # when data is short, 'singular value' is important!
        # 1 is perfect, larger than 10^5 or 10^6 there's a problem
          
        theta = np.linalg.lstsq(Phi, y,rcond=None)[0]
        self.theta = theta
                    
        # the best fitted result
        self.y_est = Phi@theta
        
        # the coefficients
        theta_nomean = theta[:-1]
        a_hat = theta_nomean[::2]
        b_hat = theta_nomean[1::2]
        #
        # amplitude estimates
        self.amp_est = np.sqrt(a_hat**2 + b_hat**2)       
        # phase estimates
        self.phase_est = np.arctan2(b_hat,a_hat)
        # signal error variance
        self.error_var = np.linalg.norm(y - self.y_est)**2/y.shape[0] 
        
class lin_window_ovrlp:
    def __init__(self, time,y,length=3,stopper=3,n_ovrlp=2):
        """
        Windowed linear detrend function with optional window overlap
        
        Parameters
        ----------
        time : N x 1 numpy array
            Sample times.
        y : N x 1 numpy array
            Sample values.
        length : int
            Window size in days
        stopper : int 
            minimum number of samples within each window needed for detrending
        n_ovrlp : int
            number of window overlaps relative to the defined window length
            
        Returns
            -------
            y.detrend : array_like
                estimated amplitudes of the sinusoids.
        Notes
        -----
        A windowed linear detrend function with optional window overlap for pre-processing of non-uniformly sampled data.
        The reg_times array is extended by value of "length" in both directions to improve averaging and window overlap at boundaries. High overlap values in combination with high
        The "stopper" values will cause reducion in window numbers at time array boundaries.   
        """
        x = np.array(time).flatten()
        y = np.array(y).flatten()
        y_detr      = np.zeros(shape=(y.shape[0]))
        counter     = np.zeros(shape=(y.shape[0]))
        A = np.vstack([x, np.ones(len(x))]).T
        #num = 0 # counter to check how many windows are sampled   
        interval    = length/(n_ovrlp+1) # step_size interval with overlap 
        # create regular sampled array along t with step-size = interval.         
        reg_times   = np.arange(x[0]-(x[1]-x[0])-length,x[-1]+length, interval)
        # extract indices for each interval
        idx         = [np.where((x > tt-(length/2)) & (x <= tt+(length/2)))[0] for tt in reg_times]  
        # exclude samples without values (np.nan) from linear detrend
        idx         = [i[~np.isnan(y[i])] for i in idx]
        # only detrend intervals that meet the stopper criteria
        idx         = [x for x in idx if len(x) >= stopper]
        for i in idx:        
            # find linear regression line for interval
            coe = np.linalg.lstsq(A[i],y[i],rcond=None)[0]
            # and subtract off data to detrend
            detrend = y[i] - (coe[0]*x[i] + coe[1])
            # add detrended values to detrend array
            np.add.at(y_detr,i,detrend)
            # count number of detrends per sample (depends on overlap)
            np.add.at(counter,i,1)
    
        # window gaps, marked by missing detrend are set to np.nan
        counter[counter==0] = np.nan
        # create final detrend array
        y_detrend = y_detr/counter       
        if len(y_detrend[np.isnan(y_detrend)]) > 0:
            # replace nan-values assuming a mean of zero
            y_detrend[np.isnan(y_detrend)] = 0.0
    
        self.y_detrend = y.detrend
        