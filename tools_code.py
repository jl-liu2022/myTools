import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
import matplotlib.collections as collections
import emcee
import corner

from scipy.interpolate import interp1d

def my_filter(data, window, n):
    if window == 1:
        return data
    else:
        return signal.savgol_filter(data, window, n)

def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def fcontinuum(x, wave_fit, flux_fit):
    return (flux_fit[0] - flux_fit[-1]) / (wave_fit[0] - wave_fit[-1]) * (x - wave_fit[0]) + flux_fit[0]

def f_rela(x):
    return 1/(x*x+1)*3e5*(x*x-1)
def f_common(x):
    return (x-1)*3e5
def f_c_to_r(x):
    return f_rela(1+x/3e5)

def Calculate_velocity(wave_p, flux_p, rest_wavelength, flux_err_p=None, phase=0., guess_wave=None, zone=600, method='guassian', window=5, save=None, MC=30, subtract=False, guess=None, bounds=None, window_toAA=True,methods=None,
    kind = 'cubic', label=None, mask_region=None, velocity_formula = 'relativistic', guess_b=None, guess_r=None, calibrate=1, calibrate_err=0, fit_err=0, window_max=None, fit_smooth=True, rest_frame=True, MCMC=False,
    phase_=0.):
    '''
    wave: spectrum wavelength, array
    flux: spectrum flux, array
    rest_wavelength: rest wavelength of the line(s), float or array
    flux_err: spectrum flux error, array
    phase: phase of the spectrum, float
    guess_wave: the rough wavelength of the feature, float
    zone: determine the examined region (= [guess_wave - zone, guess_wave + zone]), float
    method: fitting method, 'gaussian' means using an gaussian component to fit, 'direct' means direct measure, or you can input your own function
    window: the initial window used for smoothing, int
    save: the name of the output file that saves the results, string
    MC: the range of variation of the endpoints of the features, used for the uncertainty calculation with Monte Carlo, int
    subtract: flux normalized way, True for (F - F_continum), False for (F - F_continum) / F_continum, bool
    guess: guesses for parameters of method, no use for 'direct' method
    bounds: boundaries for parameters of method, no use for 'direct' method
    window_toAA: whether transform the unit of smoothing window to \AA, bool
    methods: used for plot the different components of the methods, array of function
    kind: method for interp1d when using the 'direct' method, string
    label: extra note for the velocity component, e.g., 'HV' for HV component, string
    mask_region: regions to be masked, array of number pair, e.g. [[6300,6400],[6500,6600]] means the regions 6300~6400 \AA and 6500~6600 \AA are masked
    velocity_formula: formula of the transformation from wavelength to velocity, 'relativistic' or 'common'
    guess_b: guess of the wavelength of the half max point of the feature in the blue side, float
    guess_r: guess of the wavelength of the half max point of the feature in the red side, float
    calibrate: flux calibration factor, float
    calibrate_err: error of flux calibration factor, float
    fit_err: add the covariance error given by the curve_fit method to the total error, 1 or 0
    window_max: the max window for smooothing, used for the uncertainty calculation with Monte Carlo, int
    fit_smooth: whether fit the smoothed spectrum, bool
    restframe: whether correct to rest frame
    MCMC: whether estimate with MCMC
    '''
    np.random.seed(123456789)
    if velocity_formula == 'relativistic':
        lambda_to_velocity = f_rela
    else:
        lambda_to_velocity = f_common
    feature_endpoints = [0, -1]
    if not guess_wave:
        if type(rest_wavelength) == type(['haha']):
            guess_wave = np.mean(rest_wavelength)
        else:
            guess_wave = rest_wavelength
    if window_max is None:
        window_max = window + 30
    if window_toAA == True:
        window = int(window*(len(wave_p)-1)/(wave_p[-1] - wave_p[0]))  # AA to true window
        window = window + (window%2 + 1)%2
        window_max = int(window_max*(len(wave_p)-1)/(wave_p[-1] - wave_p[0]))  # AA to true window_max
        window_max = window_max + (window_max%2 + 1)%2
    plot_range = [guess_wave - zone, guess_wave + zone]
    mark_ = (wave_p > plot_range[0]) * (wave_p < plot_range[1])
    scale0 = flux_p[mark_].max()
    flux_p = flux_p / scale0
    fluxerr_p = fluxerr_p / scale0
    flux_psmooth = my_filter(flux_p, window, 1)
    mask_ = np.array([True for i in range(len(wave_p))])
    if mask_region is not None:
        for i in range(len(mask_region)):
            mark_ = mark_ * (~((wave_p > mask_region[i][0])*(wave_p < mask_region[i][1])))
    '''
    if window == 1:
        flux_smooth = flux_
    else:
        flux_smooth = signal.savgol_filter(flux_, window, 1)
    '''
    smooth_error_all = np.abs(flux_p - flux_psmooth)
    wave_ = wave_p[mark_]
    flux_ = flux_p[mark_]
    fluxerr_ = fluxerr_p[mark_]
    flux_smooth = flux_psmooth[mark_]

    fig, ax = plt.subplots()
    ax.set_xlabel('Wavelength [$\\rm \\AA$]')
    ax.set_ylabel('Scaled Flux')
    line1, = ax.plot(wave_, flux_, marker='o', linestyle = '', picker=True, pickradius=2, ms=2)
    ax.plot(wave_, flux_smooth, c='gray')
    line_blue = ax.axvline(wave_[feature_endpoints[0]],c='b')
    line_red = ax.axvline(wave_[feature_endpoints[1]],c='r')

    endpoints_pattern = ['b']
    
    def choose_endpoints(event):
        if event.key == 'b':
            endpoints_pattern[0] = 'b'
            ax.set_title('blue')
        elif event.key == 'r':
            endpoints_pattern[0] = 'r'
            ax.set_title('red')
        fig.canvas.draw_idle()

    def onpick(event):
        thisline = event.artist
        ind = event.ind
        xdata = thisline.get_xdata()
        if endpoints_pattern[0] == 'b':
            line_blue.set_xdata([xdata[ind[0]], xdata[ind[0]]])
            feature_endpoints[0] = ind[0]
        elif endpoints_pattern[0] == 'r':
            line_red.set_xdata([xdata[ind[0]], xdata[ind[0]]])
            feature_endpoints[1] = ind[0]
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', choose_endpoints)

    plt.show()

    wave_fit = wave_[feature_endpoints[0] : (feature_endpoints[1]+1)]
    #flux_fit = flux_[feature_endpoints[0] : feature_endpoints[1]]
    if fit_smooth:
        flux_fit = flux_smooth[feature_endpoints[0] : (feature_endpoints[1]+1)]
    else:
        flux_fit = flux_[feature_endpoints[0] : (feature_endpoints[1]+1)]
    fluxerr_fit = fluxerr_[feature_endpoints[0] : (feature_endpoints[1]+1)]


    
    continuum = fcontinuum(wave_fit,  wave_fit, flux_smooth[feature_endpoints[0] : (feature_endpoints[1]+1)])
    if np.sum(flux_fit) > np.sum(continuum):
        line_type = 'emission'
    else:
        line_type = 'absorption'
    if subtract:
        flux_nor = flux_fit - continuum
        fluxerr_nor = fluxerr_fit - 0
    else:
        flux_nor = (flux_fit - continuum)/continuum
        fluxerr_nor = fluxerr_fit/continuum
    scale = np.abs(flux_nor).max()
    flux_nor = 1/scale*1000*flux_nor
    fluxerr_nor = 1/scale*1000*fluxerr_nor
    fig, ax = plt.subplots()
    if line_type == 'absorption':
        ax.set_ylabel('Normalized Flux Density $f_{\\lambda}$', fontsize=15)
    else:
        ax.set_ylabel('Scaled Flux Density $f_{\\lambda}$', fontsize=15)
    ax.set_xlabel('Rest Wavelength [$\\rm \\AA$]', fontsize=15)
    #if len(wave_fit) < 5:
    resolution = (wave_fit[-1] - wave_fit[0])/(len(wave_fit)-1)
    #else:
    #   resolution = (wave_fit[5] - wave_fit[0])/5
    if MC is None:
        random_edge = int(30 / resolution)
    else:
        random_edge = int(MC / resolution)
    print(random_edge)
    feature_endpoints_ = feature_endpoints.copy()
    print(feature_endpoints_)
    
    if method == 'gaussian':
        #def gaussians(x, params):

        #continuum = (flux_nor[0] - flux_nor[-1]) / (wave_fit[0] - wave_fit[-1]) * (wave_fit - wave_fit[0]) + flux_nor[0]
        if guess is None:
            if line_type == 'emission':
                guess = [1000, guess_wave, 10]
            else:
                guess = [-1000, guess_wave, 10]

        if bounds is None:
            bounds = ([-float('inf'),guess_wave-500,0],[float('inf'),guess_wave+500,float('inf')])
        params,params_covariance=optimize.curve_fit(gaussian, wave_fit, flux_nor, p0=guess, maxfev=500000, bounds=bounds)
        result_x = np.linspace(wave_fit[0], wave_fit[-1], 100)
        fit_result = gaussian(result_x, params[0], params[1], params[2])
        
        #print(params_covariance)
        

        
        params0 = params.copy()

        
        ax.plot(wave_fit, flux_nor, c='b', label='data', linestyle='', markersize=markersize,marker='o')
        ax.plot(result_x, fit_result, c='r', label='fit')
        ax.axvline(x=params[1],linestyle='--')
        ax.legend()
        plt.show()
        #params[0] = params[0]*scale*1e-3*scale0
        #params_covariance[0,0] = params_covariance[0,0]*scale**2*1e-6*scale0**2
        #print(params)
        #print(wave_fit[0], wave_fit[-1])
        #print(continuum[0]*scale0, continuum[-1]*scale0)
        go = int(input('go?: '))


        #velocity = (params[1] - rest_wavelength)/rest_wavelength * 300000 #km/s
        velocity = lambda_to_velocity(params[1]/rest_wavelength)

        if not subtract:
            pEW = np.abs(np.sqrt(2*np.pi)*params[0]*params[2])*scale*1e-3
            #params_covariance[0,0] = params_covariance[0,0]*scale**2*1e-6
        else:
            #pEW, pEW_integ_err = itg.quad(lambda x: gaussian(x, params[0], params[1], params[2]) / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
            #flux
            pEW = np.abs(np.sqrt(2*np.pi)*params[0]*params[2])*scale*1e-3*scale0
            #params_covariance[0,0] = params_covariance[0,0]*scale**2*1e-6*scale0**2

        #FWHM = np.sqrt(2*np.log(2))*params[2]/rest_wavelength * 300000 * 2
        FWHM = lambda_to_velocity(2.355*params[2]/rest_wavelength+1)
        print(velocity)

        

        velocity_list = []
        FWHM_list = []
        pEW_list = []
        params_list = []
        
        if MC is not None:
            for i in range(1000):
                print(i, end='\r')
                feature_endpoints_[0] = feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
                feature_endpoints_[1] = feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge
                wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
                flux_fit = flux_smooth[feature_endpoints_[0] : feature_endpoints_[1]]
                continuum = fcontinuum(wave_fit,  wave_fit, flux_fit)
                if subtract:
                    flux_nor = flux_fit - continuum
                else:
                    flux_nor = (flux_fit - continuum)/continuum
                scale = np.abs(flux_nor).max()
                flux_nor = 1/scale*1000*flux_nor
                #plt.plot(wave_fit, flux_nor, marker='o', linestyle='')
                #plt.plot(wave_fit, gaussian(wave_fit, *params0))
                #plt.show()
                params_,params_covariance_=optimize.curve_fit(gaussian, wave_fit, flux_nor, p0=params0, maxfev=500000, bounds=bounds)
                #params_[0] = params_[0]*scale*1e-3*scale0
                params_list.append(params_)
                #velocity_ = (params_[1] - rest_wavelength)/rest_wavelength * 300000 #km/s
                velocity_ = lambda_to_velocity(params_[1]/rest_wavelength)
                if not subtract:
                    pEW_ = np.abs(2.507*params_[0]*params_[2])*scale*1e-3 # np.sqrt(2*np.pi)=2.507
                else:
                    '''
                    def fpEW_(x):
                        return (gaussian(x, params_[0], params[1], params[2])) / fcontinuum(x, wave_fit, flux_fit)
                    pEW_, pEW_integ_err_ = itg.quad(fpEW_, wave_fit[0], wave_fit[-1])
                    '''
                    '''
                    pEW_, pEW_integ_err_ = itg.quad(lambda x: gaussian(x, params[0], params[1], params[2]) / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
                    '''
                    # flux
                    pEW_ = np.abs(2.507*params_[0]*params_[2])*scale*1e-3*scale0 # np.sqrt(2*np.pi)=2.507
                #FWHM_ = 2.355*params_[2]/rest_wavelength * 300000 # np.sqrt(2*np.log(2))*2=2.355
                FWHM_ = lambda_to_velocity(2.355*params_[2]/rest_wavelength+1)
                velocity_list.append(velocity_) #km/s
                FWHM_list.append(FWHM_)
                pEW_list.append(pEW_)
            velocity_MCerr = np.std(velocity_list, ddof=1)
            FWHM_MCerr = np.std(FWHM_list, ddof=1)
            pEW_MCerr = np.std(pEW_list, ddof=1)
            params_list = np.array(params_list)
            print('%s+-%s'%((np.mean(params_list[:,1]) - rest_wavelength)/rest_wavelength + 1, 
                np.std(params_list[:,1], ddof=1)/rest_wavelength))
        else:
            velocity_MCerr = 0.
            FWHM_MCerr = 0.
            pEW_MCerr = 0.
        #print(params_covariance[1,1]**0.5/rest_wavelength * 300000)
        params_uncertainty = [params_covariance[i][i]**0.5 for i in range(len(params))]
        velocity_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[1]+params[1])/rest_wavelength) - 
            lambda_to_velocity((-params_uncertainty[1]+params[1])/rest_wavelength))/2
        FWHM_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[2]+params[2])/rest_wavelength+1) - 
            lambda_to_velocity((-params_uncertainty[2]+params[2])/rest_wavelength+1))/2
        pEW_FIT_uncertainty = pEW*np.sqrt(params_uncertainty[0]**2/params[0]**2 + params_uncertainty[2]**2/params[2]**2 + calibrate_err**2)

        velocity_uncertainty = np.sqrt(velocity_MCerr**2 + velocity_FIT_uncertainty**2*fit_err)
        FWHM_uncertainty = np.sqrt(FWHM_MCerr**2 + FWHM_FIT_uncertainty**2*fit_err)
        pEW_uncertainty = np.sqrt(pEW_MCerr**2 + pEW_FIT_uncertainty**2*fit_err)
    elif method == 'direct':
        if line_type == 'absorption':
            flux_fit_direct = flux_nor
            #flux_fit_direct = signal.savgol_filter(flux_nor, window, 1)
        else:
            flux_fit_direct = -1*flux_nor
            #flux_fit_direct = -1*signal.savgol_filter(flux_nor, window, 1)
        if fit_smooth == True:
            f_flux_fit = interp1d(wave_fit, flux_fit_direct, kind=kind)
        else:
            def f_flux_fit(x):
                f_interp = fit1dcurve.Interpolator(type='hyperspline',x=wave_fit,y=flux_fit_direct,dy=flux_fit_direct/flux_fit_direct)
                return f_interp(x)[0]
        #plt.plot(np.linspace(wave_fit[0], wave_fit[-1],1000), f_flux_fit(np.linspace(wave_fit[0], wave_fit[-1],1000)))
        #plt.show()
        #exit()
        if guess is None:
            guess = guess_wave
        res_v = optimize.minimize(f_flux_fit, guess, bounds=[(wave_fit[0], wave_fit[-1])]).x[0]
        min_ = f_flux_fit(res_v)
        half_ = min_*0.5
        wave_b = np.linspace(wave_fit[0],res_v,50)
        wave_r = np.linspace(res_v,wave_fit[-1],50)
        flux_b = f_flux_fit(wave_b)
        flux_r = f_flux_fit(wave_r)
        bounds_b_min = wave_b[flux_b<0.25*min_][0]
        bounds_b_max = wave_b[flux_b>0.75*min_][-1]
        bounds_b = [(bounds_b_min,bounds_b_max)]
        if guess_b is None:
            guess_b = (bounds_b_min+bounds_b_max)/2
        else:
            bounds_b = None
        bounds_r_max = wave_r[flux_r<0.25*min_][-1]
        bounds_r_min = wave_r[flux_r>0.75*min_][0]
        bounds_r = [(bounds_r_min,bounds_r_max)]
        if guess_r is None:
            guess_r = (bounds_r_min+bounds_r_max)/2
        else:
            bounds_r = None
        print(bounds_b, bounds_r)

        res_w_b = optimize.minimize(lambda x: (1/half_*(f_flux_fit(x)-half_)*10)**2, guess_b, bounds=bounds_b).x[0]
        res_w_r = optimize.minimize(lambda x: (1/half_*(f_flux_fit(x)-half_)*10)**2, guess_r, bounds=bounds_r).x[0]
        res_w = abs(res_w_r - res_w_b)
        if not subtract:
            pEW, pEW_integ_err = itg.quad(lambda x: -1*f_flux_fit(x), wave_fit[0], wave_fit[-1])
        else:
            #pEW, pEW_integ_err = itg.quad(lambda x: f_flux_fit(x)*scale*1e-3 / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
            # flux
            pEW, pEW_integ_err = itg.quad(lambda x: f_flux_fit(x), wave_fit[0], wave_fit[-1])
        pEW = pEW * scale*1e-3
        

            
        #print(f_flux_fit(np.array([res_w_b, res_v, res_w_r])))
        #print((f_flux_fit(res_w_b)-half_)**2, (f_flux_fit(res_w_r)-half_)**2)
        flux_br = np.concatenate([np.linspace(wave_fit[0],wave_fit[-1],100), wave_fit]).sort()
        ax.plot(np.linspace(wave_fit[0],wave_fit[-1],100), f_flux_fit(np.linspace(wave_fit[0],wave_fit[-1],100)), c='b')
        ax.plot(wave_fit, flux_fit_direct, c='purple')
        ax.axvline(res_v, c = 'black')
        ax.axvline(res_w_b, c = 'b')
        ax.axvline(res_w_r, c = 'r')
        plt.show()

        #velocity = (res_v - rest_wavelength)/rest_wavelength * 300000 #km/s
        #FWHM = res_w/rest_wavelength * 300000
        velocity = lambda_to_velocity(res_v/rest_wavelength)
        FWHM = lambda_to_velocity(res_w_r/rest_wavelength) - lambda_to_velocity(res_w_b/rest_wavelength)

        go = int(input('go?: '))
        

        velocity_list = []
        FWHM_list = []
        pEW_list = []

        #print(feature_endpoints[0], feature_endpoints[1])
        for i in range(1000):
            print(i, end='\r')
            window_ = np.random.randint(window, window_max+1)
            #window_ = int(window_/resolution)
            window_ = window_ + (window_%2 + 1)%2
            flux_psmooth = my_filter(flux_p, window_, 1)
            flux_fit_ = flux_psmooth[mark_]
            if MC:
                feature_endpoints_[0] = feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
                feature_endpoints_[1] = feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge    
            wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
            flux_fit_ = flux_fit_[feature_endpoints_[0] : feature_endpoints_[1]]
                #smooth_error = smooth_error_all[feature_endpoints_[0] : feature_endpoints_[1]]
            #else:
            #   smooth_error = smooth_error_all[feature_endpoints[0] : feature_endpoints[1]]
            #flux_fit_ = flux_fit + np.random.normal(scale=smooth_error)*smooth_error
            continuum = fcontinuum(wave_fit,  wave_fit, flux_fit_)
            if subtract:
                flux_nor = flux_fit_ - continuum
            else:
                flux_nor = (flux_fit_ - continuum)/continuum
            scale = np.abs(flux_nor).max()
            flux_nor = 1/scale*1000*flux_nor
            if line_type == 'emission':
                flux_fit_direct = -flux_nor
            else:
                flux_fit_direct = flux_nor
                
            f_flux_fit_ = interp1d(wave_fit, flux_fit_direct, kind=kind)
            try:
                res_v_ = optimize.minimize(f_flux_fit_, res_v, bounds=[(wave_fit[0]+1,wave_fit[-1]-1)]).x[0] #bounds=([wave_fit[0]+1,wave_fit[-1]-1])
            except Exception as e:
                print(e)
                plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b')
                plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o')
                plt.plot(wave_fit, flux_fit_direct, c='purple')
                plt.show()
            '''
            print(res_v_)
            
            plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b', label='inter')
            plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o', label='data')
            plt.plot(wave_fit, flux_fit_direct, c='purple', linestyle='', markersize=markersize, marker='o', label='smooth')
            plt.axvline(res_v_, c = 'black')
            plt.legend()
            plt.show()
            go = int(input('go?:'))
            '''
            half_ = f_flux_fit_(res_v_)*0.5
            bounds_b_ = [(wave_fit[0]+1, res_v_)]
            bounds_r_ = [(res_v_, wave_fit[-1]-1)]
            res_w_b_ = optimize.minimize(lambda x: (1/half_*(f_flux_fit_(x)-half_)*10)**2, res_w_b, bounds=bounds_b_).x[0]
            res_w_r_ = optimize.minimize(lambda x: (1/half_*(f_flux_fit_(x)-half_)*10)**2, res_w_r, bounds=bounds_r_).x[0]
            res_w_ = abs(res_w_r_ - res_w_b_)

            if not subtract:
                pEW_, pEW_integ_err_ = itg.quad(lambda x: -1*f_flux_fit_(x), wave_fit[0], wave_fit[-1])
            else:
                #pEW_, pEW_integ_err_ = itg.quad(lambda x: f_flux_fit_(x) / fcontinuum(x, wave_fit, flux_fit_), wave_fit[0], wave_fit[-1])
                #flux
                pEW_, pEW_integ_err_ = itg.quad(lambda x: f_flux_fit_(x), wave_fit[0], wave_fit[-1])
            pEW_ = pEW_ * scale*1e-3

            #rint(window_, feature_endpoints_[0], feature_endpoints_[1], pEW_)
            #plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b', label='inter')
            #plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o', label='data')
            #plt.plot(wave_fit, flux_fit_direct, c='purple', linestyle='', markersize=markersize, marker='o', label='smooth')
            #plt.legend()
            #plt.show()
            #go = int(input('go?: '))
            
            #velocity_ = (res_v_ - rest_wavelength)/rest_wavelength * 300000 #km/s
            #FWHM_ = res_w_/rest_wavelength * 300000
            velocity_ = lambda_to_velocity(res_v_/rest_wavelength)
            FWHM_ = lambda_to_velocity(res_w_r_/rest_wavelength) - lambda_to_velocity(res_w_b_/rest_wavelength)

            velocity_list.append(velocity_) #km/s
            FWHM_list.append(FWHM_)
            pEW_list.append(pEW_)

        velocity = np.mean(velocity_list)
        velocity_uncertainty = np.std(velocity_list, ddof=1)
        FWHM = np.mean(FWHM_list)
        FWHM_uncertainty = np.std(FWHM_list, ddof=1)
        pEW = np.mean(pEW_list)
        pEW_uncertainty = np.std(pEW_list, ddof=1)

    elif type(method) == type(gaussian):
        print(bounds)
        print(guess)
        if MCMC == True:
            params,params_covariance=optimize.curve_fit(method, wave_fit, flux_nor, sigma=fluxerr_nor, p0=guess, maxfev=500000, bounds=bounds)
            print(params)
            print(np.sum(((method(wave_fit, *params) - flux_nor)/fluxerr_nor)**2) / (len(wave_fit) - len(params)))
        else:
            params,params_covariance=optimize.curve_fit(method, wave_fit, flux_nor, p0=guess, maxfev=500000, bounds=bounds)
            print(params)
        #print(params_covariance)       
        #params = [638.05303814,  674.6194568,  1878.97827889]
        #print(2.507*params[3]*params[5]*scale*1e-3)

        #result_x = np.linspace(wave_fit[0], wave_fit[-1], 100)
        #fig, ax = plt.subplots() 
        #ax.plot(wave_fit, flux_nor, c='b', label='smoothed data', linestyle='', markersize=markersize, marker='o')

        ###2022pul
        wave_plot = wave_[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
        flux_plot = flux_smooth[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
        fluxerr_plot = fluxerr_[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
        ax.errorbar(wave_plot, flux_plot, yerr=fluxerr_plot, c='black', linestyle='', alpha=0.25)
        ax.plot(wave_plot, flux_plot, c='black', label='data')
        ax.plot(wave_fit, method(wave_fit, *params)*scale*1e-3 + continuum, c='r', label='best fit')
        ax.plot(wave_fit, continuum, c='b', label='pseudocontinuum')
        ax.legend(fontsize=15)
        plt.show()
        exit()


        '''
        ###double gaussian
        ax.plot(wave_fit, flux_nor, c='black', label='data')
        ax.plot(wave_fit, gaussian(wave_fit, *params[:3]), c='purple', label='gaussian 1', linestyle='--')
        ax.plot(wave_fit, gaussian(wave_fit, *params[3:6]), c='green', label='gaussian 2', linestyle='--')
        #ax.plot(wave_fit, gaussian(wave_fit, *params[6:]), c='lime', label='gaussian 3', linestyle='--')
        ax.plot(wave_fit, method(wave_fit, *params), c='r', label='best fit')
        ax.legend(fontsize=15)
        plt.show()
        #exit()
        '''


        if MCMC == True:
        ### MCMC
            len_para = len(params)
            np.random.seed(123456789)
            def log_likelihood(theta, wave_fit, flux_nor, fluxerr_nor):
                return -0.5*np.sum(((flux_nor - method(wave_fit, *theta))/fluxerr_nor)**2)
            def log_prior(theta):
                for item_i, item in enumerate(theta):
                    if item < bounds[0][item_i] or item > bounds[1][item_i]:
                        return -np.inf
                return 0.0
            def log_probability(theta, wave_fit, flux_nor, fluxerr_nor):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(theta, wave_fit, flux_nor, fluxerr_nor)
            rand_start = np.zeros([32,len_para])
            for start_i in range(1,32):
                while(1):
                    for para_i in range(len_para):
                        rand_start[start_i][para_i] = np.random.randn()*params[para_i]*0.05
                    if not np.isinf(log_prior(params + rand_start[start_i])):
                        break
            start_MC = params + rand_start
            nwalkers, ndim = start_MC.shape
            steps = 20000
            AutocorrError = emcee.autocorr.AutocorrError
            while(1):
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability, args=(wave_fit, flux_nor, fluxerr_nor)
                )
                sampler.run_mcmc(start_MC, steps, progress=True)
                try:
                    tau = sampler.get_autocorr_time()
                except AutocorrError:
                    steps *= 2
                else:
                    break
            samples = sampler.get_chain()
            fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
            labels = ['A', 'v', 'width']
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            plt.show()
            flat_samples = sampler.get_chain(discard=int(2.5*np.max(tau)), thin=int(np.max(tau)/2), flat=True)
            log_likelihood_value = []
            for i in range(flat_samples.shape[0]):
                log_likelihood_value.append(log_likelihood(flat_samples[i], wave_fit, flux_nor, fluxerr_nor))
            max_log_likelihood = np.max(log_likelihood_value)
            for i in range(flat_samples.shape[0]):
                if log_likelihood_value[i] == max_log_likelihood:
                    pos_max = i
            theta = flat_samples[pos_max]
            print(theta)
            print(np.sum(((method(wave_fit, *theta) - flux_nor)/fluxerr_nor)**2) / (len(wave_fit) - len(theta)))
            fig = corner.corner(flat_samples, labels=labels, truths=theta)
            plt.show()
            for i in range(len_para):
                print(labels[i])
                print(np.percentile(flat_samples[:,i], [16, 50, 84]))
            ###

        go = int(input('go?: '))

        params_list = []
        for i in range(len(params)):
            params_list.append([])
        
        if MC is not None:
            for i in range(1000):
                print(i, end='\r')
                feature_endpoints_[0] = feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
                feature_endpoints_[1] = feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge
                
                wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
                flux_fit = flux_smooth[feature_endpoints_[0] : feature_endpoints_[1]]
                continuum = fcontinuum(wave_fit,  wave_fit, flux_fit)
                if subtract:
                    flux_nor = flux_fit - continuum
                else:
                    flux_nor = (flux_fit - continuum)/continuum
                flux_nor = 1/scale*1000*flux_nor
                params_,params_covariance_=optimize.curve_fit(method, wave_fit, flux_nor, p0=params, maxfev=500000, bounds=bounds)
                #print(feature_endpoints_, scale)
                #print(2.507*params_[3]*params_[5]*scale*1e-3)
                #go = int(input('go?: '))
                #print(params_)
                #fig, ax = plt.subplots() 
                #ax.plot(wave_fit, flux_nor, c='b', label='smoothed data', linestyle='', markersize=markersize,marker='o')
                #ax.plot(result_x, gaussian(result_x, params[0], params[1], params[2]), c='purple', label='gaussian1')
                #ax.plot(result_x, gaussian(result_x, params[3], params[4], params[5]), c='green', label='gaussian2')
                #ax.plot(result_x, method(result_x, *params), c='r', label='all')
                #ax.legend()
                #plt.show()
                #go = input('go?: ')
                for i in range(len(params_list)):
                    params_list[i].append(params_[i])
            params_MCerr_list = [np.std(params_list[i], ddof=1) for i in range(len(params_list))]
        else:
            #params_MCerr_list = [0 for i in range(len(params_list))]
            params_list = None
        #params_uncertainty = [np.sqrt(params_MCerr_list[i]**2 + params_covariance[i,i]*fit_err) for i in range(len(params_list))]
        if subtract:
            scale = scale*scale0
        return params, params_list, scale*1e-3*calibrate, wave_fit, flux_nor, params_covariance
    else:
        raise Excpetion('method %s is not included'%method)
    if subtract:
        pEW = pEW*scale0*calibrate
        pEW_uncertainty = np.sqrt((pEW_uncertainty*scale0*calibrate)**2 + (pEW*calibrate_err)**2)
    # Save
    if save is not None:
        with open(save, 'a') as f:
            if label is not None:
                f.writelines('%s %.1f %s %s %s %s %s %s\n' %(str(rest_wavelength)+label+'_'+str(index), phase_, velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty))
            else:
                f.writelines('%s %.1f %s %s %s %s %s %s\n' %(str(rest_wavelength)+'_'+str(index), phase_, velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty))
    feature_endpoints = [0, -1]
    return [velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty]