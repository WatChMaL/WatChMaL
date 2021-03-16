
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import random

"""
This module provides a class to do digitization of events.

class Digitizer:
  - hold all of the settings for the digitization
  - has a main method

def digitize_event( self, truepmts, truetimes )

which takes np.arrays of true PMTs and truetimes (with multiple entries per PMT).

The digitization follows these steps:
1) for each truetime generate the number of pe's following the PDF described by the function f_of_q
2) build a wavetrain of pulses following model of arXiv:1801.08690 using the function buildwavetrain
   - each pulse shape given by the function f_waveform, and 
   - the noise by the function f_noise
3) process the wavetrains to find the times and charges using the function WaveformsToTQ
   - an initial search for peaks is done to find the baseline, time and peak-height of each pulse using the
     function get_wf_peak_guesses
   - the output of that function is used to fit each peak to a gaussian in the function get_times_charges_from_wf
     which returns the 10% of the gaussian time and area of gaussian as charge for each pulse

After calling digitize_event, you can access the results of intermediate steps in the following class members:
        self.pmt_time_dict      pmt : list of truetimes of photon arrivals
        self.pmt_pe_dict        pmt : list of number of pe for each photon arrival
        self.pmt_wf_dict        pmt : ( times, charges) wavetrain for each pmt
        self.digi_qt_dict       pmt : (digitized time, charge)
        self.firstguesses       pmt : (baseline, peaktime, peakheight)
        self.peakfits           pmt : (params, covariance)
        
Blair Jamieson, Sidney Leggett (Jan. 2021)
"""


class Digitizer:
    def __init__( self ):
        """
        parameters for the PDF of charge for single photon
        parameters from arXiv:1801.08690
        """
        self.foq_q_0   = 1.0
        self.foq_q_p   = 0.3
        self.foq_sigma = 0.3
        self.foq_omega = 0.2
        self.foq_tau   = 0.5
        
        """
        parameters for the waveform pulse shape of single pe pulse
        parameters also from arXiv:1801.08690
        U0        SPE peak amplitude of pulse
        sigma0    SPE sigma (log normal)
        tau0      SPE time constant (log normal)
        U1        overshoot amplitude 1
        sigma1    overshoot sigma 1
        t1        overshoot time 1
        U2        overshoot amplitude 2
        tau2      overshoot time constant 2
        """
        self.wf_U0      = 20
        self.wf_sigma0  = 0.3 #0.15
        self.wf_tau0    = 30
        self.wf_U1      = -1.2
        self.wf_sigma1  = 55
        self.wf_t1      =-4
        self.wf_U2      =-2.8
        self.wf_tau2    =80

        """
        parameters for the noise to add to the waveforms
        nmu       dc offset of noise
        nsigma    gaussian amplitude of noise
        ncomp     number of components of oscillatory noise to add
        comprange range of frequencies in MHz to randomly sample oscillatory noise frequencies from
        compamp   range of amplitudes to randomly sample the oscillatory noise amplitudes from
        """
        self.noise_nmu       = 1.5
        self.noise_nsigma    = 1.0
        self.noise_ncomp     = 5
        self.noise_comprange = [1, 480]
        self.noise_compamp   = [0.1, 0.3]
        
        """
        Parameters for building the digitized wavetrain for photons arriving at times with pes indicated.  Convert pe to ADC.
        dt         timebin size of digitizer (0.5 ns)
        trange     digitizer time range
        """
        self.bwf_dt          = 8.0
        self.bwf_trange      = [-2000.,10000.]
        self.bwf_dcoffset    = 2045.0
        self.bwf_negpolarity = True
        
        # apply a global quantum efficiency? Set below less than 1.0 to set a global qe
        self.globalqe = 1.0
        
        # threshold in ADC on waveform (below baseline) to declare a new hit on a PMT
        self.hit_threshold = 5
        self.adc_to_pe     = 800.0 #divide by this to go from ADC to pe
        
        
    def build_wavetrain( self, times, pes, dt=8.0, trange=[-2000.,10000.], dcoffset=2045.0, negpolarity=True ):
        """
        Return digitized wavetrain for photons arriving at times with pes indicated.  Convert pe to ADC.
        Inputs:
                times      np.array of times that photons arrive at PMT
                pes        np.array of pe amplitudes for given photons
                pe_to_adc  ADC count peak per pe
                dt         timebin size of digitizer (0.5 ns)
                trange     digitizer time range
        Return:
                (t,adc)    tuple of np.arrays of time and adc values
        """
        t    = np.arange( trange[0], trange[1], dt )
        adc  = f_noise( t ,
                        self.noise_nmu,
                        self.noise_nsigma,    
                        self.noise_ncomp,
                        self.noise_comprange,
                        self.noise_compamp 
                      )
        ttmp = np.arange( 0.01, 400., dt )
    
        for (time,pe) in zip (times, pes):
            adctmp = f_waveform( ttmp, 
                                    pe * self.wf_U0, 
                                    self.wf_sigma0, 
                                    self.wf_tau0, 
                                    self.wf_U1, 
                                    self.wf_sigma1, 
                                    self.wf_t1, 
                                    self.wf_U2, 
                                    self.wf_tau2 )
                                   
            idxoffset = int( (time - trange[0])/dt )
            for (tcur, adccur) in zip( ttmp, adctmp ):
                idx = idxoffset + int( tcur/dt )
                if idx>=0 and idx<len(t):
                    adc[idx] += adccur
        if negpolarity:
            adc *= -1
        adc += dcoffset
        adc = np.rint( adc )
    
        return (t, adc)
    
    
    
    def digitize_event( self, truepmts, truetimes ):
        """
        Get list of truepmts and trutimes, build digitized data ( digipmt, digicharge, digitime )
        
        Note that dictionaries are built that could be accessed after calling digitize_event:
        self.pmt_time_dict      pmt : list of truetimes of photon arrivals
        self.pmt_pe_dict        pmt : list of number of pe for each photon arrival
        self.pmt_wf_dict        pmt : ( times, charges) wavetrain for each pmt
        self.digi_qt_dict       pmt : (digitized time, charge)
        
        """
        pmtlist  = []
        timelist = []
        pelist   = []
        
        start_time = time.time()
        for pmt in truepmts:
            pmtlist.append( pmt )
            timelist.append( truetimes[ truepmts==pmt ] )
            pelist.append( self.get_pes( len(truetimes[ truepmts==pmt ]), self.globalqe ) )
            
        self.pmt_time_dict = dict( zip(pmtlist,timelist) )
        self.pmt_pe_dict = dict( zip(pmtlist, pelist) )
        time1 = time.time()
        
        
        wflist =[]
        for pmt in truepmts:
            wflist.append( self.build_wavetrain( self.pmt_time_dict[pmt], 
                                                  self.pmt_pe_dict[pmt], 
                                                  self.bwf_dt, 
                                                  self.bwf_trange, 
                                                  self.bwf_dcoffset, 
                                                  self.bwf_negpolarity
                                                 ) )
        self.pmt_wf_dict = dict( zip(pmtlist,wflist) )
            
        time2 = time.time()
        
        #self.pmt_time_dict = { pmt: truetimes[ truepmts==pmt ] for pmt in truepmts }
        #self.pmt_pe_dict = { pmt: self.get_pes( len(truetimes[ truepmts==pmt ]), self.globalqe )  for pmt in truepmts }
        #self.pmt_wf_dict = { pmt: self.build_wavetrain( self.pmt_time_dict[pmt], 
        #                                          self.pmt_pe_dict[pmt], 
        #                                          self.bwf_dt, 
        #                                          self.bwf_trange, 
        #                                          self.bwf_dcoffset, 
        #                                          self.bwf_negpolarity
        #                                         ) for pmt in truepmts }
        
        self.digi_qt_dict = self.WaveformsToTQ( self.pmt_wf_dict, self.hit_threshold )
    
    
        time3 = time.time()
        # Pick one hit per PMT
        # currently it picks the one closest to t=0
        digipmt = []
        digitime = []
        digicharge = []
        for pmt in self.digi_qt_dict:
            digipmt.append( pmt )
            idxmin = 0
            tmin = -90000
            for i,t in enumerate( self.digi_qt_dict[pmt][0] ):
                if abs(t) < abs(tmin):
                    tmin=t
                    idxmin=i
            digitime.append( self.digi_qt_dict[pmt][0][idxmin] )
            digicharge.append( self.digi_qt_dict[pmt][1][idxmin] )
        
        time4 = time.time()
        
        elapsed_time = cur_time-start_time
        print_time( time1 - start_time,   "Digitizer::digitize_event pe and time dict building =" )
        print_time( time2 - time1,        "Digitizer::digitize_event waveform dictionary building =" )
        print_time( time3 - time2,        "Digitizer::digitize_event t,q digitization building =" )
        print_time( time4 - time3,        "Digitizer::digitize_event pick one t,q per PMT =" )
        return ( digipmt, digitime, digicharge )
    

    
    
    def get_pes( self, ntimes=1, globalqe = 1.0 ):
        """
            Function to randomly sample from f_of_q to generate a list of 
            number of photoelectrons detected. Apply overal quantum efficiency
            that doesn't care about angle of incidence or wavelength of photon.
    
            Inputs:
                ntimes   number of pes to calculate
                globalqe fraction of pe to keep non-zero
            Outputs:
                np.array of length ntimes of photoelectron pulse heights
        """
        pes = np.empty(ntimes)
        for i in range(ntimes):
            setpe = False
            while setpe==False:
                q = np.random.uniform(0.3,3.0)
                h = np.random.uniform(0.0,1.12)
                if ( f_of_q(q,
                            self.foq_q_0,  
                            self.foq_q_p, 
                            self.foq_sigma, 
                            self.foq_omega, 
                            self.foq_tau
                           ) > h ):
                    qethrow = np.random.uniform(0.0,1.0)
                    if qethrow  < globalqe:
                        pes[i] = q
                    else:
                        pes[i] = 0.0
                    setpe=True
        return pes
    
    
    def WaveformsToTQ( self, pmtwd, threshold=5 ):
        """
        Take dictionary of waveforms and build list of digitized (pmt, time, charge) values.
    
        Build hit based on a time window around value going below threshold.  Fit the waveform to a Gaussian.
    
        Find time and adc using a fit to a gaussian of samples around each threshold crossing.
    
        Input:
        pmtwd      Dictionary of { pmt : (times, adcs) }
                   times is an np.array of times, and
                   adcs is an np.array of corresponding adc values
        threshold  threshold below baseline to declare a new pulse
    
        Output:
        digipmt   {pmt : (digitimes, digiadcs) }  dictionary of pmt Tuple of np.arrays of corresponding time, and adc
        digitimes  has np.array for each digipmt entry of length number of peaks found 
        digiadcs   has np.array for each digipmt entry of lenght number of peaks found
        
        Intermediate steps are stored in the class for later checks.
        self.firstguesses = {pmt : (baseline, peaktime, peakheight)}
        self.peakfits     = {pmt : (params, covariance)} ---> actually this is added to in get_times_charges_from_wf
        """
        pmtlist    = []
        digitqlist = []
        guesslist  = []
        self.peakfits     = {}
        for pmt in pmtwd:
            (bl, tpks, apks) = get_wf_peak_guesses( pmtwd[pmt][0], pmtwd[pmt][1] ) 
            guesslist.append(  (bl,tpks,apks) )
            self.curpmt = pmt
            if len( tpks ) > 0:
                (tp, qp)         = self.get_times_charges_from_wf( pmtwd[pmt][0], pmtwd[pmt][1], threshold, bl, tpks, apks )
                if len(tp)!=0: # ignore pmts that didn't fire
                    pmtlist.append( pmt )
                    digitqlist.append( (tp, qp) )
            else:
                print("WaveformsToTQ PMT ",pmt," wf_peak_guesses found no peaks")
        
        digipmt = dict( zip( pmtlist, digitqlist ) )
        self.firstguesses = dict( zip( list(pmtwd.keys()), guesslist ) ) 
       
        #digipmt =  {}
        #self.firstguesses = {}
        #self.peakfits     = {}
        #for pmt in pmtwd:
        #    (bl, tpks, apks) = get_wf_peak_guesses( pmtwd[pmt][0], pmtwd[pmt][1] )
        #    self.firstguesses[ pmt ] = (bl,tpks,apks)
        #    self.curpmt      = pmt
        #    
        #    if len( tpks ) > 0:
        #        (tp, qp)         = self.get_times_charges_from_wf( pmtwd[pmt][0], pmtwd[pmt][1], threshold, bl, tpks, apks )
        #        if len(tp)!=0: # ignore pmts that didn't fire
        #            digipmt[ pmt ] = (tp, qp)
        #    else:
        #        print("WaveformsToTQ PMT ",pmt," wf_peak_guesses found no peaks")
        return digipmt
    
        
    
    def get_times_charges_from_wf( self, twf, adcwf, thresh, baseline, tbinpeaks, adcpeaks, adcwferr=1.0 ):
        """
        Fit each peak in tbinspeaks,adcpeaks and return np.arrays of times and charges of peaks
        Inputs:
        twf       np.array of digitized waveform times
        adcwf     np.array of digitized waveform adc values
        thresh    threshold at which to identify peak
        baseline  is the first guess at the baseline
        tbinpeaks np.array of time bins that peaks are at
        adcpeaks  np.array of adc heights of the peaks
        adcwferr  Uncertainty in adc values 
      
        Output:
        times     np.array of peak times (where the peak passes threshold? -- just the peak for now)
        charges   np.array of total charges (integral of the gaussian fit)
        
        Intermediate values saved for later checks:
        self.peakfits
        """
    
        tfrac = 1.0 # fraction of peak at which to declare trigger!
        times = []
        charges = []
        makeplots = False
    
        dtlocal = twf[1]-twf[0]
        window_width = int( 80/dtlocal )
        #print("window_width=",window_width)
        adcwferrs = adcwferr*np.ones( window_width*2 )
        #print("adcwferrs=",adcwferrs)
    
        if len(twf)<window_width*2:
            print("Warning waveform is too short :",len(twf)," < ",2*window_width )
            return (np.array(times),np.array(charges))
    
        for i in range(len(tbinpeaks)):
            tbin = tbinpeaks[i]
            apk  = adcpeaks[i]
            if i>0:
                print("PMT ",self.curpmt," peak_fit=",i)
        
            tmin = tbin-int(window_width/2)
            if tmin<0:
                tmin = 0
            tmax = tbin+window_width
           
            if tmax >= len(twf):
                tmax=len(twf)-1
            if tmax-tmin < window_width:
                print("Warning waveform segment is too short :",tmax-tmin," < ",window_width )
                return (np.array(times),np.array(charges))
            
            """
            print("curve_fit tbin=",tbin," tmin=",tmin," tmax=",tmax," len(twf)=",len(twf))
            print("curve_fit twf[tmin]=",twf[tmin]," twf[tbin]=", twf[tbin]," twf[tmax]=",twf[tmax] )
            print("baseline=",baseline, " tbin=",tbin, " twf[tbin]=", twf[tbin], " apk=",apk)
            print("twf=",twf[tmin:tmax])
            print("adcwf=",adcwf[tmin:tmax])
            print("sigma=",adcwferrs[0:tmax-tmin])
            print("p0=",[baseline, apk, twf[ tbin ], 10.0])
            """
            
            try:
                params, covs = curve_fit( pulseshape, 
                          twf[ tmin: tmax], 
                          adcwf[ tmin: tmax],
                          sigma = adcwferrs[0:tmax-tmin],
                          p0 = [baseline, apk, twf[ tbin ], 5.0] )
                #print("curve_fit success params=",params)
                #print("curpmt=",self.curpmt)
                if self.curpmt in self.peakfits:
                    self.peakfits[ self.curpmt ].append( (params,covs) )
                else:
                    self.peakfits[ self.curpmt ] =  [ (params, covs ) ]
                #print("here")
            except:
                print("curve_fit failed tbin=",tbin," tmin=",tmin," tmax=",tmax," len(twf)=",len(twf))
                continue
            
            if makeplots:
                plot_fitted_waveform( "fit result", twf, adcwf, tbin, window_width, params, adcwferr )
                
        
            times.append( params[2] - 2*params[3]*np.sqrt(-np.log(tfrac))  )
            charges.append( np.sqrt(2*np.pi) * params[1] * params[3]  / self.adc_to_pe )

            return ( np.array(times), np.array(charges) )

    
    def plot_waveform_fit( self, pmt ):
        """
        Plot waveform fits for pmt.
        """
        if pmt in self.pmt_wf_dict:
            pmtwd = self.pmt_wf_dict
            twf = pmtwd[pmt][0]
            adcwf = pmtwd[pmt][1]
            dtlocal = twf[1]-twf[0]
            window_width = int( 80/dtlocal )
            (bl, tpks, apks) = self.firstguesses[ pmt ]
            if len(tpks)>0:
                for i,tbin in enumerate(tpks):
                    if i < len( self.peakfits[pmt] ):
                        params, cov = self.peakfits[ pmt ][i]
                        plot_fitted_waveform( "PMT %d peak %d"%(pmt,i), twf, adcwf, tbin, window_width, params, 1.0 )
                    else:
                        print("plot_waveform_fit: no fit for PMT ",pmt," waveform i=",i," tbin=",tbin)
            else:
                print("plot_waveform_fit: no first guess for PMT",pmt)
        else:
            print("plot_waveform_fit: no fit for PMT %d"%pmt)

    
"""
The following functions are called from the Digitizer class, but are not part of it.
def f_of_q
def f_waveform
def f_noise
def get_wf_peak_guesses
def pulseshape
"""
        
        
        
        
def f_of_q( q, q_0 = 1.0, q_p=0.3, sigma=0.3, omega=0.2, tau=0.5):
    """
    Get probability density of charge for a single photon.
    Replace this function with model for our PMT!
    """
    fofq = (1-omega)/(sigma*np.sqrt(2*math.pi)) * np.exp( -0.5*( (q-q_0)/sigma )**2 )
    fofq += omega / tau * np.exp( -q / tau )
    return fofq

def f_waveform( t, U0=20, sigma0=0.15, tau0=30, U1=-1.2, sigma1=55, t1=-4, U2=-2.8, tau2=80):
    """
    Build waveform template for a single photon pulse as np.array of amplitudes. Based on arXiv:1801.08690 (Eq. 2.5)
    Inputs:
    t         np.array of time samples
    U0        SPE peak amplitude of pulse
    sigma0    SPE sigma (log normal)
    tau0      SPE time constant (log normal)
    U1        overshoot amplitude 1
    sigma1    overshoot sigma 1
    t1        overshoot time 1
    U2        overshoot amplitude 2
    tau2      overshoot time constant 2
    Returns:
    np.array of amplitudes for standard time pulse
    """
    Upeak = np.where( t > 0,  U0 * np.exp( -0.5 * ( np.log(t/tau0)/sigma0 )**2 ), 0. )
    Uos1 = U1 * np.exp( -0.5 * ( ( t-t1)/ sigma1 )**2 )
    Uos2 = U2 / ( np.exp( (50.0-t)/10.0 ) + 1.0 ) * np.exp( -t/tau2 )
    U = Upeak + Uos1 + Uos2
    for i in range( len(t) ):
        if t[i]>tau0:
            break
        if U[i]< 0.:
            U[i] = 0.
    return U

def f_noise( t, nmu=1.5, nsigma=0.5, ncomp=5, comprange=[1,480], compamp=[0.05,0.15]):
    """
    Builds waveform template for noise in pe as np.array of amplitudes. Based on arXiv:1801.08690
    Inputs:
    t         np.array of time samples to calculate noise over
    nmu       dc offset of noise
    nsigma    gaussian amplitude of noise
    ncomp     number of components of oscillatory noise to add
    comprange range of frequencies in MHz to randomly sample oscillatory noise frequencies from
    compapm   range of amplitudes to randomly sample the oscillatory noise amplitudes from
    Returns:
    np.array of amplitudes of noise
    """
    n = nsigma * np.random.randn( len(t) ) + nmu
    compfreqs = np.random.rand( ncomp ) * (comprange[1]-comprange[0]) + comprange[0]
    compamps  = np.random.rand( ncomp ) * (compamp[1]-compamp[0]) + compamp[0]
    compphi   = np.random.rand( ncomp ) * 2 * np.pi
    for (freq, amp, phi) in zip(compfreqs, compamps, compphi):
        n += amp * np.sin( freq*2*np.pi*1e6 * t + phi )
    return n



def build_dicts( pmt_time_dict, pmt_pe_dict, pmt_parent_dict, onlynoise=False):
    """
    Build dictionary that has noise hits filtered out (either keep only noise hits
    or only signal hits).
    Inputs:
    tpmts      np.array of PMT Ids
    ttimes     np.array of true timess
    tparents   np.array of photon parents (-1 noise)
    onlynoise  set to True to get only noise hits, False to remove noise hits
    Outputs:
    time_dict     dictionary of PMT Id to photon arrival times
    min_time_dict dictionary of PMT Id to photon earliest arrival time
    pe_dict       dictionary of PMT Id to number of pe's
    photons_dict  dictionary of PMT Id to number of photons
    pecount_dict  dictionary of PMT Id to pe count
    """
    time_dict    = {}
    mintime_dict = {}
    pe_dict      = {}
    photons_dict = {}
    pecount_dict = {}
    for pmt in pmt_time_dict:
        times = []
        pes   = []
        for i,parent in enumerate( pmt_parent_dict[pmt] ):
            if parent != -1 and onlynoise == False:
                times.append( pmt_time_dict[pmt][i] )
                pes.append( pmt_pe_dict[pmt][i] )
            if parent == -1 and onlynoise == True:
                times.append( pmt_time_dict[pmt][i] )
                pes.append( pmt_pe_dict[pmt][i] )
        if len(times)>0:
            time_dict[pmt] = np.array( times )
            mintime_dict[pmt] = np.min( times )
            pe_dict[pmt] = np.array( pes )
            photons_dict[pmt] = len( times )
            pecount_dict[pmt] = np.sum( pes )
    return ( time_dict, mintime_dict, pe_dict, photons_dict, pecount_dict )

"""
Example filtering of events for noise and signal photons
noise_tdict, noise_tmindict, noise_pedict, noise_pdict, noise_ndict = build_dicts( pmt_time_dict, pmt_pe_dict, pmt_parent_dict, True)
sig_tdict, sig_tmindict, sig_pedict, sig_pdict, sig_ndict = build_dicts( pmt_time_dict, pmt_pe_dict, pmt_parent_dict, False)


noisepmts    = np.array( list(noise_tmindict.keys()) )
noisetimes   = np.array( list(noise_tmindict.values()) )
noisecharges = np.array( list(noise_pdict.values()) )
noisepes     = np.array( list(noise_ndict.values() ) )
EventDisplay( noisepmts, noisetimes, "Times of noise hits",[0.,2000.])

sigpmts    = np.array( list(sig_tmindict.keys()) )
sigtimes   = np.array( list(sig_tmindict.values()) )
sigcharges = np.array( list(sig_pdict.values()) )
sigpes     = np.array( list(sig_ndict.values() ) )
EventDisplay( sigpmts, sigtimes, "Times of signal hits",[0.,100.])
"""


def get_wf_peak_guesses( t, adc, thresh=5 ):
    """
    Find list of times where there is a peak in the waveform (minimum?)
    Inputs:
    t       np.array of times at which adc values are located
    adc     np.array of adc values at the corresponding times
    thresh  distance in adc below baseline to identify a peak
    Return:
    baseline  adc value that peaks decend from
    tpeak     np.array of time-bin where the peaks are
    adcpeak   np.array of adc values of distance from baseline to peak
    """
    
    # first find baseline?
    # average over all adc values that are less than thresh different than next/prev one
    baseline = 0.0
    blcount  = 0
    dt = t[1]-t[0]
    nbinswidth = int( 400/dt )
    for i in range(1,len(adc)-1):
        if ( abs( adc[i] - adc[i-1] ) < thresh and 
             abs( adc[i] - adc[i+1] ) < thresh ):
            baseline += adc[i]
            blcount += 1
    if blcount>0:
        baseline /= blcount
    
    tpeak   = []
    adcpeak = []
    i       = 0
    while i < len(adc):
        if ( baseline - adc[i] > thresh ):
            # found a peak... now find where it is maxium
            maxbin = i
            maxpk  = baseline-adc[i]
            for j in range(i+1, min(i+nbinswidth,len(adc)-1)):
                if baseline-adc[j] > maxpk:
                    maxbin = j
                    maxpk = baseline-adc[j]
            tpeak.append( maxbin )
            adcpeak.append( maxpk )
            i = i+nbinswidth
        i+=1
    return ( baseline, np.array(tpeak), np.array(adcpeak) )
            
def pulseshape(tval, dcoffset, adcpeak, tmean, tsigma ):
    """
    Model for pulse shapes observed in digitizer
    Parameters:
    dcoffset  pedestal value from which pulse goes down/up 
    adcpeak   height of the pulse
    tmean     time of the pulse
    tsigma     width of the pulse
    """
    #print("pulseshape: dcoffset=",dcoffset," adcpeak=",adcpeak," tmean=",tmean, " tsigma=",tsigma)
    return dcoffset - adcpeak * np.exp( -0.5*( (tval-tmean)/tsigma )**2 ) 


def plot_fitted_waveform( plot_title, twf, adcwf, tbin, window_width, params, adcwferr ):
    """
    Takes waveform (twf,adcwf) and plots around time bin (tbin) with width in number of
    bins given by window_width.  Uses params as the parameters of the waveform taken from
    the function pulseshape (a gaussian with offset).  The errorbar on the adc values
    is given by adcwferr
    """
    adcwfexp = pulseshape( twf[ tbin-int(window_width/4): tbin+window_width], *params)
    resid = adcwf[ tbin-int(window_width/4): tbin+window_width] - adcwfexp
    chisq = np.sum((resid/adcwferr)**2)
    df    = window_width*2 - 4
    #print("chisq =",chisq,"df =",df)
    plt.figure(figsize=(10,10))
    plt.xlim( twf[tbin-int(window_width/2)], twf[tbin+window_width])
    plt.errorbar( twf, adcwf, adcwferr*np.ones( len(adcwf)), fmt='.' )
    plt.xlabel('time (ns)',fontsize=24)
    plt.ylabel('ADC', fontsize=24)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.title( plot_title+" chi2/ndf = %d/%d"%(chisq,df) )
    tttmin = twf[ tbin-int(window_width/2) ]
    tttmax = twf[ tbin+window_width ]
    tttfine = np.arange( tttmin, tttmax, 0.1 )    
    plt.plot( tttfine, pulseshape( tttfine, *params ) )
    
