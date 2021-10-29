import scipy
import scipy.integrate
import scipy.optimize
import numpy as np
import pylab
import matplotlib.pyplot as plt
import copy
import os
import sys
import scipy.stats as stats


class Param:
    '''Parameter for model fitting.

    Grabbed Code from http://www.scipy.org/Cookbook/FittingData
    '''
    def __init__(self, value, name='unknown'):
        self.value = value
        self.uncertainty = None
        self.covindex = None  # the corresponding index in the cov matrix
        self.initial_value = value  # this will be unchanged after the fit
        self.name = name

    def set(self, value):
        self.value = value

    def __call__(self):
        # When paramter is called, return itself as the value
        return self.value


def fminfit(function, parameters, y, yerr, x=None, algorithm=None):
    '''
    Note: this code is not well fleshed out compared to fit() in terms of the
    object-orienty nature or returning estimates of uncertainties. Right now
    all it will solve/return is the best fit values.

    Use fmin or a variant thereof to optimize/find the minimum of a the
    chi-squared statistic
    ((y-function(x))**2/(yerr**2)).sum()

    where yerr is the known variance of the observation, y is the observed data
    and function(x) is the theoretical data.[1] This definition is only useful
    when one has estimates for the error on the measurements, but it leads to a
    situation where a chi-squared distribution can be used to test goodness of
    fit, provided that the errors can be assumed to have a normal distribution.

    if algorithm == bfgs then use the bfgs fmin fit,
        if algorithm == None use the default simplex

    from fmin guide:
        This algorithm has a long history of successful use in applications.
        But it will usually be slower than an algorithm that uses first or
        second derivative information. In practice it can have poor performance
        in high-dimensional problems and is not robust to minimizing
        complicated functions. Additionally, there currently is no complete
        theory describing when the algorithm will successfully converge to the
        minimum, or how fast it will if it does.
    '''
    def errfunc(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return ((y-function(x))**2/(yerr**2)).sum()  # chi sq

    paraminit = [param() for param in parameters]
    print("Initial Parameters: ", paraminit)
    if not algorithm:
        p = scipy.optimize.fmin(errfunc, paraminit, maxfun=None,
                                maxiter=None, full_output=1)
    elif algorithm == 'bfgs':
        p = scipy.optimize.fmin_bfgs(errfunc, paraminit)
    solved_values = p[0]
    func_value = p[1]  # final function value, chi-sq in this case
    niter = p[2]  # number of iterations
    nfunc = p[3]  # number of function calls

    print("Solved Values: ", solved_values)
    # don't really need to return this as the values are going to be
    # stored in the Param objects
    return solved_values


def fit(function, parameters, y, yerr=None, x=None, return_covar=False, 
        method='lm', bounds=None, loss='linear'):
    '''Fit performs a simple least-squares fit on a function.  To use:
    
    method can be one of ['lm','trf','dogbox'], as accepted by curve_fit
    'lm' calls leastsq, the others call least_squares
    
    TODO: Add ability for bounded problems (trf & dogbox only) and loss functions
    
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of x0 or be a scalar, in the latter case
        a bound will be the same for all variables. Use np.inf with an
        appropriate sign to disable bounds on all or some variables.

    Give initial paramaters:
        mu = Param(7)
        sigma = Param(3)
        height = Param(5)

    Define your function:
        def f(x): return height() * exp(-((x-mu())/sigma())**2)

    Make simulated data:
        xvals = np.arange(100)
        gaussian = lambda x: 3*exp(-(30-x)**2/20.)
        ydata = gaussian(xvals)
        ydata = scipy.randn(100)*.1+ydata #adding noise
        yerr = np.zeros(100)+.1 #array of uncertainties

    Fit the function (provided 'data' is an array with the data to fit):
        fit(f, [mu, sigma, height], ydata, yerr, xvals)

    Plot the fitted model over the data if desired
        simxvals = np.arange(10000)/100. # 10000 points from 0-100
        plot(simxvals, f(simxvals))
        scatter(xvals, ydata)

    Your input parameters will now have the fitted values assigned to them.
    '''
    # This errfunc assumes a uniform error for each datapoint!
    def errfunc(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return (y - function(x)) / yerr

    # errfunc = lambda x, y, err: (y-function(x))/err

    pr = """Fitting {0} free parameters on {1} datapoints...
         """.format(len(parameters), len(y))
    print(pr)

    # If no x axis given, set x array as integral steps starting with
    # zero for the length of the y array.  For instance, if
    # y = array([1.32, 2.15, 3.01, 3.92]),
    # then x will be x = array([0, 1, 2, 3])
    if x is None:
        x = np.arange(y.shape[0])  # not sure if this works
        
    if yerr is None:
        print("Warning: No y-uncertainties specified; assuming yerr=1")
        yerr = 1

    paraminit = [param() for param in parameters]
    print('Initial Parameters:', paraminit)
    # print "Parameter list:", paraminit
    # print "error function:", errfunc
    if method == 'lm':
        if bounds is not None:
            print("WARNING: Bounds are not accepted for lm method")
        if loss is not 'linear':
            print("WARNING: non-linear loss is not accepted for lm method")
            
        fitout = scipy.optimize.leastsq(errfunc, paraminit, full_output=1)
        # paramfinal = [param() for param in parameters]
        paramfinal = fitout[0]
        covarmatrix = fitout[1]
        info = fitout[2]
        mesg = fitout[3]
        errint = fitout[4]
    
    # Could in principle call least_squares regardless since it wraps around
    # leastsq as well if method == 'lm', but the 'full_output' kwarg is only
    # used in the leastsq method.  
    # Need to calculate the covarmatrix if least_squares is used; see curve_fit.
    # However the covarmatrix returned by leastsq is different than that 
    # returned by curve_fit; the latter is multiplied by chi2/dof 
    elif method == 'trf' or method == 'dogbox': 
        if bounds is None:
            bounds=(-np.inf, np.inf)
        fitout = scipy.optimize.least_squares(errfunc, 
                                              paraminit, 
                                              method=method, 
                                              bounds=bounds,
                                              loss=loss)
        paramfinal = fitout['x']
        covarmatrix = None #FIXME 
        errint = fitout['status']
        mesg = fitout['message']
        info = {'nfev':fitout['nfev']}
        # print fitout
    else:
        raise ValueError('Invalid fit method: {}'.format(method))

    # errint of 1-4 means a solution was found
    if errint not in np.arange(1, 5):
        raise Exception(mesg)
    print(info['nfev'], ' function calls required to find solution')
    # print errint
    # print mesg
    print('Final Parameters:', paramfinal)
    print('Covariance Matrix', covarmatrix)
    print('')
    # If paramfinal is not an array, make it one to avoid scripting errors
    if not isinstance(paramfinal, np.ndarray):
        paramfinal = np.array([paramfinal])

    # Calculate the chi-squared value, and the reduced chi-squared
    # http://mail.scipy.org/pipermail/scipy-user/2005-June/004632.html
    chi2 = sum(np.power(errfunc(paramfinal), 2))
    degrees_of_freedom = y.shape[0] - len(paramfinal)
    chi2r = chi2/degrees_of_freedom

    print("chi^2 / dof = %.2f / %i" % (chi2, degrees_of_freedom))
    print("reduced chi^2 = %.3f  \n" % (chi2r))

    retdict = {'parameters': parameters, 'covarmatrix': covarmatrix,
               'chi2': chi2, 'dof': degrees_of_freedom
               }

    count = 0
    fitstrlist = []
    values_dict = {}
    uncertainties_dict = {}
    for param in retdict['parameters']:
        param.covindex = count  # corresponding index in the covmatrix
        try:
            uncertainty = np.sqrt(retdict['covarmatrix'].diagonal()[count])
        except:
            print("ERROR: Cannot calculate uncertainty.")
            uncertainty = np.nan
        # update the uncertainty in the param object
        param.uncertainty = uncertainty
        fitstr = '{0}: {1:.3f} +/- {2:.3f}'.format(param.name, param.value, uncertainty)
        print(fitstr)
        fitstrlist.append(fitstr)
        
        values_dict.update({param.name:param.value})
        uncertainties_dict.update({param.name:param.uncertainty})
        
        count += 1
    retdict.update({'strings': fitstrlist,
                    'values': values_dict,
                    'uncertainties': uncertainties_dict})

    if return_covar:
        return covarmatrix
    else:
        return retdict  # return the dictonary of outputs


def plot_marg_from_fitdict(fitdict, paramnames):
    '''Given a fit dictionary from fit(), and a tuple of parameter names (from
    param.name), get the covariance matrix and plot the marginalization e.g.
    paramnames = ('Av_1','beta_1')
    '''
    allvalues = np.zeros(len(fitdict['parameters']))
    indices = [-1, -1]
    values = [0, 0]
    names = ['a1', 'a2']
    covmat = fitdict['covarmatrix']
    count = 0
    for param in fitdict['parameters']:
        allvalues[count] = param.value
        count += 1
        if paramnames[0] == param.name:
            indices[0] = param.covindex
            names[0] = param.name
            values[0] = param.value
        if paramnames[1] == param.name:
            indices[1] = param.covindex
            names[1] = param.name
            values[1] = param.value

    # print indices
    # return allvalues
    ret = plot_marginalization(covmat=covmat,
                               indices=indices,
                               names=names,
                               values=values)
    return ret


def plot_marginalization(covmat=None, indices=None,
                         names=None, values=None,
                         plot_delta_values=False,
                         invert_yticks=True,
                         storepath='./'):
    '''Suppose we dont care much about some parameters and want to explore
    the uncertainties involved in just two. We can marginalize over the
    other parameters by first extracting the relevant values from the
    covariance matrix above to form a new 2x2 covariance matrix:
    '''
    if covmat is None:  # default just for illustration
        covmat = np.matrix([
            [5.29626719, 0.57454987, -0.73125854],
            [0.57454987, 1.16079146, -0.28095744],
            [-0.73125854, -0.28095744, 0.23075755]])

    if indices is None:  # default for illustration
        ind1 = 1
        ind2 = 2
    else:
        ind1 = indices[0]
        ind2 = indices[1]
    if names is None:  # default for names
        names = ['a1', 'a2']

    unc_1 = np.sqrt(covmat[ind1, ind1])
    unc_2 = np.sqrt(covmat[ind2, ind2])

    # slice the covariance matrix
    cov_i = np.matrix([[covmat[ind1, ind1], covmat[ind1, ind2]],
                       [covmat[ind2, ind1], covmat[ind2, ind2]]])

    # And we invert this to get a new curvature matrix:
    curv_i = cov_i.getI()

    # Now from this curvature matrix we can write (c.f. Eq.
    # 9.3 of Aficionados):
    # \[
    # \Delta \chi^2_{\mathbf{a_i}} = \delta \mathbf{a_i^T} \cdot
    #               [\alpha_\chi]_\mathbf{i} \cdot \delta \mathbf{a_i},
    # \]
    # where $\delta \mathbf{a_i^T} = [\delta a_1 \; \delta a_2]$.
    #
    # To visualize these $\chi^2$ values, we create a 256 by 256 grid of
    #  $\delta a_1, \delta a_2$ values and calculate the $\chi^2$ for each.

    scale = max((unc_1, unc_2))*3.5

    dx = np.linspace(-1*scale, scale, 256)
    dy = np.linspace(-1*scale, scale, 256)
    delta_chi_sq = np.zeros((256, 256))

    # calculate grid of delta_chi_squared:
    x_ind = 0
    for dx_i in dx:
        y_ind = 0
        for dy_j in dy:
            delta_a = np.matrix([[dx_i], [dy_j]])
            delta_chi_sq[x_ind, y_ind] = delta_a.getT()*curv_i*delta_a
            y_ind += 1
        x_ind += 1

    # Now with this data grid, we can plot the contours corresponding to
    # $\Delta \chi^2_{\mathbf{a_i}} = 1.0$ and
    # $\Delta \chi^2_{\mathbf{a_i}} = 2.3.$
    # The latter value is where 68.3\% of measured values should lie inside for
    # two degrees of freedom.
    # $\Delta \chi^2_{\mathbf{a_i}} = 6.17.$ is where 95.4% of measured values
    # should lie for 2 degrees of freedom (corresponding to 4sigma for 1dof)
    # 9.21 is 99% confidence
    levels = np.array([1.0, 2.3, 9, 11.8])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    CS = ax.contour(dx, dy, delta_chi_sq, levels)
    ax.clabel(CS, inline=1, fontsize=10)
    # label these axes only if desired; could add confusion
    if plot_delta_values:
        yname = '$\delta %s$' % names[0]
        xname = '$\delta %s$' % names[1]
        ax.set_ylabel(yname)
        ax.set_xlabel(xname)
    else:  # get rid of the tickmarks
        ax.set_xticks([])
        ax.set_yticks([])

    # marginalize over two parameters and plot the corresponding lines;
    # these values are equivalent to the uncertainties from the diagonals
    # of the original covariance matrix above.  It can be seen that they bound
    # the error ellipse corresponding to$\Delta \chi^2 = 1.0,$ as they should
    # (hence the term 'marginalization' - this contour, projected into the
    # margins, gives the uncertainty for a single parameter of interest).

    ax.axvline(x=0, linestyle='dotted', color='grey')
    ax.axhline(y=0, linestyle='dotted', color='grey')

    # 1 sigma
    ax.axvline(x=unc_2, linestyle='dashed')
    ax.axvline(x=-1*unc_2, linestyle='dashed')

    ax.axhline(y=unc_1, linestyle='dashed')
    ax.axhline(y=-1*unc_1, linestyle='dashed')
    # 3 sigma
    ax.axvline(x=3*unc_2, linestyle='dashed', color='orange')
    ax.axvline(x=-3*unc_2, linestyle='dashed', color='orange')

    ax.axhline(y=3*unc_1, linestyle='dashed', color='orange')
    ax.axhline(y=-3*unc_1, linestyle='dashed', color='orange')

    if not plot_delta_values:  # get rid of the tickmarks
        ax.set_xticks([])
        ax.set_yticks([])

    if values is not None:
        xlim = ax.get_xlim() + values[1]
        ylim = ax.get_ylim() + values[0]
        ax2 = ax.twiny()
        ax3 = ax.twinx()
        ax2.set_xlim(xlim)
        ax3.set_ylim(ylim)
        # HACK
        if not "Av_1" and "beta_1" in names:
            yname = '$%s$' % names[0]
            xname = '$%s$' % names[1]
        else:     # HACK changing Av_1 to \Delta A_V
            yname = '$\Delta A_V$'
            xname = '$\Delta \\beta$'
        # END HACK
        ax3.set_ylabel(yname)
        ax2.set_xlabel(xname)

    # ### HACK ### Av needs to be inverted
    if invert_yticks:
        yticks = ax3.get_yticks()
        ax3.set_yticklabels(yticks * -1)
    # ### END HACK ###
    if not plot_delta_values:  # get rid of the tickmarks
        ax.set_xticks([])
        ax.set_yticks([])

    path = storepath + 'marginalization.png'
    fig.savefig(path)
    path = storepath + 'marginalization.pdf'
    fig.savefig(path)

    return (dx, dy, delta_chi_sq, levels)


def test_fit():
    '''
    Test the leastsq algorithm on a toy model

    It seems that the standard leastsq fit is more sensitive to this choice
    of initial parameters (for mu in particular) than the fmin fit below.
    When I had mu = 20, it sometimes converged to noise with a terrible chisq,
    '''
    import matplotlib.pyplot as plt

    # Make simulated data:
    xvals = np.arange(100)
    zeros = np.zeros(100)
    zipxvals = list(zip(xvals, zeros))

    gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
    # true values: mu = 30, height = 3, sigma = sqrt(20) = 4.472
    ydata = gaussian(xvals)
    ydata = scipy.randn(100)*.05+ydata  # adding noise
    yerr = np.zeros(100)+.05  # array of uncertainties

    # Give initial paramaters:

    mu = Param(34, name='mu')
    sigma = Param(4, name='sigma')
    height = Param(5, name='height')

    # Define your function:
    def f(x):
        ''' here we are using proof of concept of a multi-input function
        e.g. y = f(x, t)
        'x' is a list of tuples, zipped with the zip function (see below)
        we unzip them and then evaluate.
        x needs to be a single parameter because of the way the errfunc
        in fit() is defined
        e.g., x = [(0, 0.0), (1, 0.0), (2, 0.0),
                   (3, 0.0), (4, 0.0), (5, 0.0)]
        '''
        xval, zero = list(zip(*x))  # unzip the x feature vector
        return height() * np.exp(-((xval-mu())/sigma())**2) + zero

    # Fit the function (provided 'data' is an array with the data to fit):
    retdict = fit(f, [mu, sigma, height], ydata, yerr, zipxvals)

    # Plot the fitted model over the data if desired
    simxvals = np.arange(10000)/100.  # 10000 points from 0-100
    simzeros = np.zeros(len(simxvals))
    zipsimxvals = list(zip(simxvals, simzeros))

    fig2 = plt.figure()
    ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(simxvals, f(zipsimxvals))

    ax.scatter(xvals, ydata)
    fig2.show()
    return retdict


def test_linear(fix_intercept=False, fixed_noise=False, 
                method='lm', loss='linear', include_outlier=False):
    '''
    Test the leastsq algorithm on a toy linear model with heteroscedastic 
    errors. 
    '''
    import matplotlib.pyplot as plt

    model1color = 'green'
    model2color = 'red'
    truthcolor = 'black'

    # Make simulated data:
    xvals = np.arange(20)*4
    # gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
    # true values: mu = 30, height = 3, sigma = sqrt(20) = 4.472
    truedict = {'slope': 0.50, 'intercept': 0.0}

    linear = lambda x: truedict['slope'] * x + truedict['intercept']
    ydata_true = linear(xvals)
    est_y_err = 5000*(xvals+10.0)**-1.8  # making some heteroskedastic noise
    # if you want to keep things constant to test methods
    if fixed_noise:
        noise = np.array([-0.83541075, -2.2786471 , -0.36255932,  0.78580705,  0.32141668,
                          -1.51922958, -0.67322265,  0.2738563 ,  0.20824763, -0.81179434,
                           0.85400535, -0.02339395,  0.31723236,  1.5935366 , -1.3481765 ,
                           0.39672636,  0.2270063 , -1.02623091, -0.78666783,  0.2448412 ])
    else:
        noise = scipy.randn(20)
            
    ydata = noise*est_y_err+ydata_true  # adding noise to the data
    yerr = np.zeros(20)+est_y_err  # array of uncertainties

    # include a crazy outlier if desired
    if include_outlier:
        outlier_index = -5
        outlier_val = 100.0
        ydata[outlier_index] = outlier_val

    # set up your dictonary of values to fit
    fitdict = {}
    names = ['slope', 'intercept']
    for name in names:
        fitdict.update({name: {'init': 1, 'fixed': False}})

    if fix_intercept:
        # fit the intercept at 0
        fitdict['intercept']['init'] = 0
        fitdict['intercept']['fixed'] = True

    # BUILD UP PARAMETERS
    fullparamlist = []
    fitparamlist = []
    fixparamlist = []

    # set parameters
    for key, val in fitdict.items():
        param = Param(val['init'], name=key)
        fullparamlist.append(param)
        if val['fixed'] is False:
            fitparamlist.append(param)
        elif val['fixed'] is True:
            fixparamlist.append(param)
        else:
            raise ValueError('Invalid value for Fixed. Must be True/False.')

    # creating parameters to fit
    # beta1 = Param(3, name = 'slope')
    # beta2 = Param(10, name = 'intercept')
    # paramlist = [beta1, beta2]
    #
    # Define your function:
    def myfunc(x, paramlist):
        for param in paramlist:
            if param.name == 'slope':
                slope = param.value
            elif param.name == 'intercept':
                intercept = param.value
            else:
                raise Exception('Invalid parameter')
        return slope*x + intercept

    def f(x):
        return myfunc(x, fullparamlist)

    # Plot the fitted model over the data if desired
    simxvals = np.array([-5, 50, 85])  # 10000 points from 0-100
    simzeros = np.zeros(len(simxvals))

    fig2 = plt.figure()
    ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8])

    # plot the underlying truth
    ax.plot(simxvals, linear(simxvals), lw=7, alpha=0.4, color=truthcolor)

    # Fit the function (provided 'data' is an array with the data to fit):
    retdict = fit(f, fitparamlist, ydata, yerr, xvals, method=method, loss=loss)
    ax.plot(simxvals, f(simxvals), lw=2, color=model1color)

    # Now assuming tiny/no error equal to the mean of the real errors 
    # yerr_wrong = np.zeros(len(yerr))+yerr.mean()
    yerr_wrong = None
    # Fit the function (provided 'data' is an array with the data to fit):
    retdictnoerror = fit(f, fitparamlist, ydata, yerr_wrong, xvals, method=method, loss=loss)
    ax.plot(simxvals, f(simxvals), lw=2, color=model2color)

    # Plot the data with error bars
    ax.errorbar(xvals, ydata, yerr=yerr, fmt='o', color=truthcolor, lw=2, capthick=2)
    ax.errorbar(xvals, ydata, yerr=yerr_wrong, fmt='.', color=truthcolor, ecolor=model2color, capsize=0)
    
    if include_outlier:
        # plot the outlier again in a different color
        ax.plot([xvals[outlier_index]],[outlier_val], color='pink',marker='o')
        fig2.text(0.65, 0.80, 'outlier', color='pink')
    ax.set_xlim((-5, 85))
    ax.set_ylim((-120, 120))

    # bottom left annotations
    xloc = 0.2
    textoffset = 0.16
    textincrement = 0.04
    textsize = 12
    color = model1color
    modeldict = retdict
    title = 'Including measurement error'
    # model1 annotations
    string = 'chi2 / dof = %.2f / %i' % (modeldict['chi2'], modeldict['dof'])
    fig2.text(xloc, textoffset, string)
    textoffset += textincrement
    for string in modeldict['strings']:
        fig2.text(xloc, textoffset, string, color=color)
        textoffset += textincrement
    fig2.text(xloc, textoffset, title, size=textsize)

    # bottom right annotations
    xloc = 0.55
    textoffset = 0.16
    textincrement = 0.04
    textsize = 12
    color = model2color
    modeldict = retdictnoerror
    title = 'Assuming const measurement error'
    # model2 annotations
    string = 'chi2 / dof = %.2f / %i' % (modeldict['chi2'], modeldict['dof'])
    fig2.text(xloc, textoffset, string)
    textoffset += textincrement
    for string in modeldict['strings']:
        fig2.text(xloc, textoffset, string, color=color)
        textoffset += textincrement
    fig2.text(xloc, textoffset, title, size=textsize)

    # True annotations
    textoffset = 0.7
    for key, val in truedict.items():
        string = '{}: {}'.format(key,val)
        fig2.text(0.2, textoffset, string, color=truthcolor,
                  alpha=0.4, size=14)
        textoffset += 0.042
    fig2.text(0.2, textoffset, 'True Inputs:', size=14)

    string = 'Fixed params:'
    if fixparamlist:
        fig2.text(0.7, 0.8, string, size=14)
        textoffset = 0.76
    for myparam in fixparamlist:
        string = '{}: {}'.format(myparam.name,myparam.value)
        fig2.text(0.7, textoffset, string, alpha=0.4, size=14)
        textoffset -= 0.04
    
    ax.set_title("Method '{}' with loss function '{}'".format(method,loss))
    fig2.show()
    return retdict


def sample_from_multivariate_normal_test(retdict, plot_every_model=True):
    '''Proof of concept of sampling from a multivariate gaussian to generate
    a distribution of model fits.
    docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html
    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions. Such a distribution is specified by its mean and covariance
    matrix. These parameters are analogous to the mean (average or "center")
    and variance (standard deviation, or "width," squared) of the
    one-dimensional normal distribution.
    '''

    # definition of the underlying functional model
    def f(x, mu, sigma, height):
        return height * np.exp(-((x-mu)/sigma)**2)

    # extracting the mean values and covariance matrix from the fit retdict
    mean = [param.value for param in retdict['parameters']]
    mean = tuple(mean)
    cov = retdict['covarmatrix']

    nxvals = 1000
    nsimulations = 500

    # nxvals samples from normal distribution
    samples = np.random.multivariate_normal(mean, cov, (nsimulations))

    simxvals = np.linspace(0, 100, nxvals)  # 1000 points from 0-100

    fig2 = plt.figure()
    ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8])

    # Array of simulated y values
    simymat = np.zeros((nsimulations, nxvals))

    count = 0
    for sample in samples:
        simyvals = f(simxvals, sample[0], sample[1], sample[2])
        simymat[count] = simyvals
        # plot every simulated model with low opacity. This takes a long time
        # but gives a nice illustration of the underlying possible models
        if plot_every_model:
            ax.plot(simxvals, simyvals, lw=2, color='black', alpha=0.01)
        count += 1

    # Two sigma is 95.4%
    simylower02point3 = stats.scoreatpercentile(simymat, 2.3, axis=0)
    simyupper97point7 = stats.scoreatpercentile(simymat, 97.7, axis=0)

    if not plot_every_model:
        ax.fill_between(simxvals, simylower02point3,
                        simyupper97point7, color='#CCCCCC')
        simyupper84 = stats.scoreatpercentile(simymat, 84.1, axis=0)
        simylower16 = stats.scoreatpercentile(simymat, 15.9, axis=0)
        ax.fill_between(simxvals, simylower16, simyupper84, color='#888888')
        ax.plot(simxvals, f(simxvals, mean[0], mean[1], mean[2]),
                color='black')

    else:
        ax.plot(simxvals, simyupper97point7, color='red', lw=2)
        ax.plot(simxvals, simylower02point3, color='blue', lw=2)

    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(14, 46)
    plt.show()

    return simymat


def test_fit_fmin():
    import matplotlib.pyplot as plt

    # Make simulated data:
    xvals = np.arange(100)
    zeros = np.zeros(100)
    zipxvals = list(zip(xvals, zeros))

    gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
    # True values: mu = 30, height = 3, sigma = sqrt(20) = 4.472
    ydata = gaussian(xvals)
    ydata = scipy.randn(100)*.05+ydata  # adding noise
    yerr = np.zeros(100)+.05  # array of uncertainties

    # Give initial paramaters:
    mu = Param(34, name='mu')
    sigma = Param(4, name='sigma')
    height = Param(5, name='height')

    # Define your function:
    def f(x):
        ''' here we are using proof of concept of a multi-input function
        e.g. y = f(x, t)
        'x' is a list of tuples, zipped with the zip function (see below)
        we unzip them and then evaluate.
        x needs to be a single parameter because of the way the errfunc
        in fit() is defined
        e.g., x = [(0, 0.0), (1, 0.0), (2, 0.0),
                   (3, 0.0), (4, 0.0), (5, 0.0)]
        '''
        xval, zero = list(zip(*x))  # unzip the x feature vector
        return height() * np.exp(-((xval-mu())/sigma())**2) + zero

    # Fit the function (provided 'data' is an array with the data to fit):
    fminfit(f, [mu, sigma, height], ydata, yerr, zipxvals)

    # Plot the fitted model over the data if desired
    simxvals = np.arange(10000)/100.  # 10000 points from 0-100
    simzeros = np.zeros(len(simxvals))
    zipsimxvals = list(zip(simxvals, simzeros))

    fig2 = plt.figure()
    ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(simxvals, f(zipsimxvals))

    ax.scatter(xvals, ydata)
    fig2.show()


def test_multi():
    # Borrowed from SEDfit code
    y0 = np.array([12, 13, 14, 15, 16, 17]*10)
    x0 = np.arange(300).reshape(60, 5)
    names = ['b1', 'b2', 'b3', 'b4', 'b5']
    fitdict = {}
    for name in names:
        fitdict.update({name: {'init': 1, 'fixed': False}})
    # BUILD UP PARAMETERS
    fullparamlist = []
    fitparamlist = []
    fixparamlist = []

    # Set parameters
    for key, val in fitdict.items():
        param = Param(val['init'], name=key)
        fullparamlist.append(param)
        if val['fixed'] is False:
            fitparamlist.append(param)
        elif val['fixed'] is True:
            fixparamlist.append(param)
        else:
            raise ValueError('Invalid value for Fixed. Must be True/False.')

    def myf(x):
        return np.dot(x, np.array([beta.value for beta in fitparamlist]).T)

    retdict = fit(myf, fitparamlist, y0, np.array([0.1]*60), x0)
    return retdict

def robust_least_squares_test(plot=True):
    '''http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    
    scipy version 0.17.0 introduced the least_squares method, which unlike
    the leastsq method, allows for bounds on the variables and also has an
    option for including a loss function rho(s) for more robust estimates

    '''
    from scipy.optimize import least_squares, leastsq, curve_fit

    def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):
        y = a + b * np.exp(t * c)

        rnd = np.random.RandomState(random_state)
        error = noise * rnd.randn(t.size)
        outliers = rnd.randint(0, t.size, n_outliers)
        error[outliers] *= 10

        return y + error

    # True values
    a = 0.5
    b = 2.0
    c = -1
    t_min = 0
    t_max = 10
    n_points = 15

    t_train = np.linspace(t_min, t_max, n_points)
    y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
    
    def fun(t, a, b, c):
        return a + b * np.exp(c * t)

    def cost_fun(x, t, y):
        # here it's just regular least sq. For weighted, divide by err.
        return fun(t, x[0], x[1], x[2]) - y

    x0 = np.array([1.0, 1.0, 0.0])
    
    res_lsq = least_squares(cost_fun, x0, args=(t_train, y_train))
    
    res_soft_l1 = least_squares(cost_fun, x0, loss='soft_l1', f_scale=0.1,
                                args=(t_train, y_train))
    res_log = least_squares(cost_fun, x0, loss='cauchy', f_scale=0.1,
                            args=(t_train, y_train))

    # grab the outputs from the legacy leastsq for comparison 
    fitout = leastsq(cost_fun, x0, full_output=1, args=(t_train, y_train))
    res_leastsq = {'x':fitout[0], 'cov':fitout[1], 'msg':fitout[3], 'status':fitout[4]}
    res_leastsq.update(fitout[2])
    
    # 'lm' uses the legacy leastsq method and is the default for unconstrained 
    # problems using curve_fit as of scipy 0.17.0
    curve_out =  curve_fit(fun, t_train, y_train, p0=x0, method='lm', full_output=1)
    
    curve_fit_lm = {'x':curve_out[0], 'pcov':curve_out[1], 'msg':curve_out[3], 'status':curve_out[4]}
    curve_fit_lm.update(curve_out[2])

    # trf is trust-region reflective method, which can deal with constrained problems.  Called through least_squares method. 
    curve_out = curve_fit(fun, t_train, y_train, p0=x0, method='trf')
    curve_fit_trf = {'x':curve_out[0], 'pcov':curve_out[1]}

    if plot:
        t_test = np.linspace(t_min, t_max, n_points * 10)
        y_true = gen_data(t_test, a, b, c)
        y_lsq = gen_data(t_test, *res_lsq.x)
        y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
        y_log = gen_data(t_test, *res_log.x)
        
        plt.plot(t_train, y_train, 'o')
        plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
        plt.plot(t_test, y_lsq, label='linear loss')
        plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
        plt.plot(t_test, y_log, label='cauchy loss')
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    
    r = {'least_squares':res_lsq,
         'soft_l1':res_soft_l1,
         'cauchy':res_log,
         'leastsq':res_leastsq,
         'curve_fit_lm':curve_fit_lm,
         'curve_fit_trf':curve_fit_trf,
     }
    
    fit_types = list(r.keys())
    for fit_type in fit_types:
        str_out_params = ['{0:.2f}'.format(val) for val in r[fit_type]['x']]
        print('{}: {}'.format(fit_type.ljust(15), ' '.join(str_out_params)))
    
    # TODO:
    # the least_squares, leastsq, and curve_fit methods should all be equal
    # note there's a difference in the cov_x returned by leastsq and the pcov 
    # returned by curve_fit.  
    # http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es/14857441#14857441
    # basically pcov = cov_x*chi^2/dof  
    # TODO: am i calculating the parameter uncertainties correctly in qFit, given this information? 
    
    # but really, bootstrap methods should be used to calculate the uncertainties anyway 
    
    return r 

def plot_robust_fit_example_1():
    '''Fitting a gaussian sampled sine with polynomial of order 3
    
    Shows difference between OLS, Theil-Sen, RANSAC regressions
    
    '''
    from matplotlib import pyplot as plt
    import numpy as np

    from sklearn import linear_model, metrics
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    np.random.seed(42)

    X = np.random.normal(size=400)
    y = np.sin(X)
    # Make sure that it X is 2D
    X = X[:, np.newaxis]

    X_test = np.random.normal(size=200)
    y_test = np.sin(X_test)
    X_test = X_test[:, np.newaxis]

    y_errors = y.copy()
    y_errors[::3] = 3

    X_errors = X.copy()
    X_errors[::3] = 3

    y_errors_large = y.copy()
    y_errors_large[::3] = 10

    X_errors_large = X.copy()
    X_errors_large[::3] = 10

    estimators = [('OLS', linear_model.LinearRegression()),
                  ('Theil-Sen', linear_model.TheilSenRegressor(random_state=42)),
                  ('RANSAC', linear_model.RANSACRegressor(random_state=42)), ]

    x_plot = np.linspace(X.min(), X.max())

    for title, this_X, this_y in [
            ('Modeling errors only', X, y),
            ('Corrupt X, small deviants', X_errors, y),
            ('Corrupt y, small deviants', X, y_errors),
            ('Corrupt X, large deviants', X_errors_large, y),
            ('Corrupt y, large deviants', X, y_errors_large)]:
        plt.figure(figsize=(5, 4))
        plt.plot(this_X[:, 0], this_y, 'k+')

        for name, estimator in estimators:
            model = make_pipeline(PolynomialFeatures(3), estimator)
            model.fit(this_X, this_y)
            mse = metrics.mean_squared_error(model.predict(X_test), y_test)
            y_plot = model.predict(x_plot[:, np.newaxis])
            plt.plot(x_plot, y_plot,
                     label='%s: error = %.3f' % (name, mse))

        plt.legend(loc='best', frameon=False,
                   title='Error: mean absolute deviation\n to non corrupt data')
        plt.xlim(-4, 10.2)
        plt.ylim(-2, 10.2)
        plt.title(title)
    plt.show()
    

def plot_robust_fit_example_2():
    '''Fitting a uniformly sampled sine with polynomial of order 5'''
    from matplotlib import pyplot as plt
    import numpy as np

    from sklearn import linear_model, metrics
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    np.random.seed(42)

    X = np.random.uniform(size=400)*10 - 2
    y = np.sin(X)
    # Make sure that it X is 2D
    X = X[:, np.newaxis]

    X_test = np.random.normal(size=200)
    y_test = np.sin(X_test)
    X_test = X_test[:, np.newaxis]

    y_errors = y.copy()
    y_errors[::3] = 3

    X_errors = X.copy()
    X_errors[::3] = 3

    y_errors_large = y.copy()
    y_errors_large[::3] = 10

    X_errors_large = X.copy()
    X_errors_large[::3] = 10

    estimators = [('OLS', linear_model.LinearRegression()),
                  ('Theil-Sen', linear_model.TheilSenRegressor(random_state=42)),
                  ('RANSAC', linear_model.RANSACRegressor(random_state=42)), ]

    x_plot = np.linspace(X.min(), X.max())

    for title, this_X, this_y in [
            ('Modeling errors only', X, y),
            ('Corrupt X, small deviants', X_errors, y),
            ('Corrupt y, small deviants', X, y_errors),
            ('Corrupt X, large deviants', X_errors_large, y),
            ('Corrupt y, large deviants', X, y_errors_large)]:
        plt.figure(figsize=(5, 4))
        plt.plot(this_X[:, 0], this_y, 'k+')

        for name, estimator in estimators:
            model = make_pipeline(PolynomialFeatures(5), estimator)
            model.fit(this_X, this_y)
            mse = metrics.mean_squared_error(model.predict(X_test), y_test)
            y_plot = model.predict(x_plot[:, np.newaxis])
            plt.plot(x_plot, y_plot,
                     label='%s: error = %.3f' % (name, mse))

        plt.legend(loc='best', frameon=False,
                   title='Error: mean absolute deviation\n to non corrupt data')
        plt.xlim(-4, 10.2)
        plt.ylim(-2, 10.2)
        plt.title(title)
    plt.show()
    