#!/usr/bin/env python
"""
Plotomatic.py
Author: Adam N Morgan

Collection of modules to generate exploratory plots from pandas dataframes.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import seaborn as sns


class MegaDF(object):
    '''
    Wrapper around pandas dataframe, storing additional aggregate stats so they
    don't need to be recalculated.
    '''
    def __init__(self, df):
        self.df = df
        self.aggdict = {}
        self.maddict = {}  # Not sure I wanna keep this
        # May not keep PopulateAggDict() in the __init__ as we may not want to
        # aggregate every time.
        self.PopulateAggDict()

    def PopulateAggDict(self, class_col=None):
        '''
        Generage the aggdict() attribute, which contains aggregate statistitcs
        for the dataframe that will not have to be recalculated.
        
        If class_col specified, split dataframe into different columns 
        based on the column name.  TODO: This is not a great way to split things up.. 
        '''
        # TODO: Store different stats depending if float, int, str, etc
        if class_col != None:
            for class_key in self.df[class_col].unique():
                for column in self.df.columns:
                    self.df[column+'_'+str(class_key)] = self.df[column][self.df[class_col] == class_key]

        for col in self.df.columns:
            curr_series = self.df[col]
            self.aggdict.update({col: {}})
            # Create root node
            # Loop over the `describe()` dataframe and store those values
            name = 'root'
            for ind, val in curr_series.describe().iteritems():
                if name not in self.aggdict[col]:
                    self.aggdict[col].update({name: {}})
                self.aggdict[col][name].update({ind: val})
            # Calculate median absolute deviation
            if '50%' in self.aggdict[col][name]:
                med = self.aggdict[col][name]['50%']
                mad = (self.df[col] - med).abs().median()
                self.aggdict[col][name]['mad'] = mad
                if mad > 0:
                    # Create mad5 node
                    name = 'mad5'
                    mad5_series = curr_series[(
                        (curr_series < med + 5*mad) &
                        (curr_series > med - 5*mad))]
                    for ind, val in mad5_series.describe().iteritems():
                        if name not in self.aggdict[col]:
                            self.aggdict[col].update({name: {}})
                        self.aggdict[col][name].update({ind: val})
                    if '50%' in self.aggdict[col][name]:
                        med = self.aggdict[col][name]['50%']
                        # Subtract from mad5
                        mad = (mad5_series - med).abs().median()
                        self.aggdict[col][name]['mad'] = mad

                    self.maddict.update({col: mad5_series})

    def pdhist(self, col_name_list, colors=None, labels=None, bins=None,
               nbins=50, fancylegend=False, plotmad5=False, suptitle=None,
               verbose=False, plot_row_span=3):
        '''
        Given a list of pandas series, plot aligned histograms along with a
        legend giving descriptive statistics of the populations.

        TODO: change series_list into series_name_or_list,
            accepting a list of names (or a single name)
        TODO: loop over multiple series
        '''

        if fancylegend:
            nitems = len(col_name_list)
            if plotmad5:
                nitems *= 2  # Double number if plotting mad-removed
            nextra = (nitems + 2) / 3  # Keep max of 3 rows
            gss = gs.GridSpec(3, 3+nextra)  # nrows, ncols
            # Make wider as adding info
            fig = plt.figure(figsize=(8+2*nextra, 6))
            # Main plot axis
            ax = fig.add_subplot(gss[:plot_row_span, :3])  # row, col
            axdict = {}

        else:
            fig = plt.figure()
            fig.set_size_inches(8, 6)

            ax = fig.add_subplot(111)

        if colors is None:  # Assign defaults
            colors = [
                '#c0392b',
                '#2980b9',
                '#27ae60',
                '#9b59b6'] * ((len(col_name_list)+2)/3)

        force_categorical = False

        count = 0
        axind = 0
        for col in col_name_list:
            if verbose:
                print 'Plotting {}'.format(col)

            try:
                color = colors[count]
            except:
                color = None  # FIXME colors not handled correctly
            try:
                label = labels[count]
            except:
                label = None

            series = self.df[col]
            # Check dtype of column
            if series.dtype is np.dtype('O'):
                try_categorical = True  # may be strings
                if verbose:
                    print ("  Warning: datatype for {} is 'object'."
                           " May not be able to plot.".format(col))
            else:
                try_categorical = False

            x_limit = None

            mmn = 0
            mmx = 100
            buff = 1e-10

            if bins is None:
                if series.name in self.aggdict:
                    ad = self.aggdict[series.name]

                    if (plotmad5 is True) and ('mad5' in ad):
                        mmn = ad['mad5']['min']
                        mmx = ad['mad5']['max']
                        buff = float(mmx-mmn)/nbins
                    elif 'root' in ad:
                        if 'min' in ad['root']:
                            mmn = ad['root']['min']
                            mmx = ad['root']['max']
                            buff = 1e-10
                        else:
                            if verbose:
                                print ('  Warning: min/max not in root aggdict'
                                       ' for {}. Possible'
                                       ' categorical?'.format(series.name))
                            force_categorical = True
                else:
                    if verbose:
                        print ('  Warning: {} not in aggdict. Setting bins'
                               ' to default values.'.format(series.name))

                bins = np.linspace(mmn - 20*buff, mmx + 20*buff, nbins+40)
                x_limit = (mmn-10*buff, mmx+10*buff)

            if not force_categorical:
                res = ax.hist(series.values, bins=bins, alpha=0.2,
                              edgecolor='none', normed=False,
                              histtype='stepfilled', color=color)
                if not plotmad5:
                    res = ax.hist(series.values, bins=bins, alpha=0.92,
                                  linewidth=3, normed=False, histtype='step',
                                  label=label, color=color)
                else:  # Attempt to plotmad5
                    if hasattr(self, 'maddict'):
                        if col in self.maddict:
                            res = ax.hist(self.maddict[col].values, bins=bins,
                                          alpha=1.0, linewidth=2, normed=False,
                                          histtype='step', color=color)
                        else:
                            if verbose:
                                print ('  Warning: {} not in maddict. Cannot plot'
                                       ' mad hist.'.format(col))
                            plotmad5 = False
                    else:
                        if verbose:
                            print ('  Warning: maddict does not exist for this'
                                   ' MegaDF. Cannot plot mad hist.')
                        plotmad5 = False

                if fancylegend:
                    types = ['root']
                    if plotmad5:
                        types += ['mad5']
                    for typename in types:
                        tripcount = axind / 3
                        # frameon=False turns off just the background and edges
                        # set_axis_off gets rid of ticks, ticklabels, edges
                        tmpax = fig.add_subplot(
                            gss[axind % 3, tripcount-nextra], frameon=False)
                        tmpax.set_axis_off()
                        axstr = 'ax'+str(axind)

                        # TODO: TEST THIS
                        if series.name in self.aggdict:
                            ad = self.aggdict[series.name]
                            nn = ad[typename]['count']
                            fmtdict = {}
                            for key in ['mean','min','max','50%','std','mad']:
                                val = ad[typename][key]
                                # Making long numbers into sci notation
                                outstr = '{:.2f}'.format(val)
                                if len(outstr) > 8:
                                    outstr = '{:.2e}'.format(val)
                                    outstr = outstr.replace('e','\\times 10^{{')
                                    outstr += '}}'
                                    outstr = outstr.replace('+','')
                                fmtdict.update({key: outstr})
                            textstr = (
                                '$n={0:.0f}$\n'
                                '${1}\leq x \leq {2}$\n'
                                '$\mu={3} \pm {4}$\n'
                                '$\mathrm{{Md}}={5} \pm {6}$'
                                ''.format(nn, 
                                          fmtdict['min'],
                                          fmtdict['max'],
                                          fmtdict['mean'],
                                          fmtdict['std'],
                                          fmtdict['50%'],
                                          fmtdict['mad']))

                            # These are matplotlib.patch.Patch properties

                            if typename != 'root':
                                al = 0.4
                            else:
                                al = 0.2
                            props = dict(boxstyle='round',
                                         ec='none',
                                         fc=color,
                                         alpha=al)

                            # Place a text box in upper left in axes coords
                            tmpax.text(
                                0.5, 0.8, textstr,
                                transform=tmpax.transAxes,
                                fontsize=14,
                                verticalalignment='top',
                                horizontalalignment='center',
                                linespacing=1.5,
                                bbox=props)

                            title = str(label)
                            if typename == 'mad5':
                                title += ' (mad5)'

                            tmpax.text(
                                0.5, 1.0, str(title),
                                transform=tmpax.transAxes,
                                fontsize=13,
                                verticalalignment='top',
                                horizontalalignment='center',
                                color=color)

                        else:
                            if verbose:
                                print '  {0} Not in aggdict'.format(series.name)

                        axdict.update({axstr: tmpax})
                        axind += 1

                ax.set_ylabel('Count')
                ax.set_xlabel('Value')

                if x_limit:
                    ax.set_xlim(x_limit)

                count += 1
                bins = res[1]

            else:  # If force_categorical, try plotting a bar chart
                msg = "  Categorical plotting not yet fully implemented"
                print msg
                # ax.text(0.5, 0.5, msg, transform=ax.transAxes, fontsize=13,
                #     verticalalignment='top', horizontalalignment='center',
                #     color='red')

                # Only plot if there are fewer than 20 values
                if len(series.value_counts()) < 20:
                    ax = series.value_counts().plot(kind='bar', lw=0,
                                                    ec='none', fc=color,
                                                    alpha=0.2)
                    ax = series.value_counts().plot(kind='bar', lw=3,
                                                    ec=color, fc='none',
                                                    alpha=0.92)
                else:
                    print ("  Too many types to plot bar, "
                           "skipping {}".format(series.name))
                count += 1
                

        fig.tight_layout()
        if suptitle:
            fig.subplots_adjust(top=0.90)
            fig.suptitle(suptitle, fontsize=16)

        self.fig = fig

    def pdhistloop(self, col_name_list, saveto='./plots/', dpi=144):
        saveto += '/'
        if not os.path.exists(saveto):
            raise ValueError('Path does not exist: {}'.format(saveto))

        for colstr in col_name_list:
            try:
                res = self.pdhist(
                    [colstr], bins=None, nbins=100, colors=None,
                    fancylegend=True, labels=['Stats'], plotmad5=True,
                    suptitle=colstr)
                self.fig.savefig("{0}{1}.png".format(saveto, colstr), dpi=dpi)
            except:
                print "Cannot plot " + colstr


def qcut_and_replace(df, colname, outcolname, nbins=8, precision=2):
    """
    Bin a dataframe column into equal parts and add a column with
    labels stripped of parenthesis to the dataframe
    """

    import pandas as pd

    bg = pd.qcut(df[colname], nbins, precision=precision)
    bg = bg.astype('object')
    bg = (((bg.str.replace('(', '')
            ).str.replace(')', '')
           ).str.replace('[', '')
          ).str.replace(']', '')
    df[outcolname] = bg
    print ("Added '{}' to dataframe via qcut on column '{}'"
           " with {} bins".format(outcolname, colname, nbins))


def PlotViolin(indf, colname, groupname,
               title=None, ylabel=None, xlabel=None,
               titlefontsize=None, ylabelfontsize=None, xlabelfontsize=None,
               rotate_xlabels=True, remove_outliers=True, bw=0.05,
               color=None):
    '''
    Given a dataframe, name of column to plot, and name of column to
    groupby, generate a violinplot of the distributions.

    TODO: Make part of MegaDF?
    '''
    df = pd.DataFrame(dict(score=indf[colname], group=indf[groupname]))

    # Adjust the width to be wider if plotting many violins
    nviolins = len(indf[groupname].unique())
    maxwidth = (nviolins / 1.8)
    minheight = (nviolins * 0.10)

    figsize = (min(45, max(9, maxwidth)), max(4, minheight))
    sns.set_context("notebook", font_scale=1.8,
                    rc={
                        "figure.figsize": figsize
                        })

    fig = plt.figure()
    # fig.tight_layout()
    ax1 = fig.add_subplot(111)

    # VIOLINPLOT DOESNT LIKE DATETIMEINDECES??
    df = df.reset_index()

    if remove_outliers:
        med = np.median(df.score)
        mad = np.median(np.abs(df.score - med))

        df2 = df[(df.score < med + 5*mad) & (df.score > med - 5*mad)]
    else:
        df2 = df

    if color is not None:
        color = sns.diverging_palette(21, 20, n=12)
    else:
        color = sns.husl_palette(n_colors=12, h=0.01, s=0.6, l=0.6)
    g = sns.violinplot(df2.score, df2.group, bw=bw, ax=ax1, color=color)
    # remove the upper and right spines
    sns.despine(offset=3, trim=True)
    # Rotate the xlabels
    if rotate_xlabels:
        plt.setp(g.xaxis.get_majorticklabels(), rotation=45, ha='right')

    if title is not None:
        ax1.set_title(title)
    if ylabel is None:
        ax1.set_ylabel(colname)
    else:
        ax1.set_ylabel(ylabel)
    if xlabel is None:
        ax1.set_xlabel(groupname)
    else:
        ax1.set_xlabel(xlabel)

    if titlefontsize is not None:
        ax1.title.set_fontsize(titlefontsize)
    if ylabelfontsize is not None:
        ax1.yaxis.label.set_size(ylabelfontsize)
    if xlabelfontsize is not None:
        ax1.yaxis.label.set_size(xlabelfontsize)


def get_rid_of_non_ints(string, replace_with=np.nan):
    '''Attempt to convert a string into an integer. If input is not
      convertable into an integer, return the value stored in the
      `replace_with` argument (defaults to numpy float NaN)

    Args:
      string (str): String to attempt to convert into an integer
      replace_with (obj, optional): Object to return if string
        cannot be converted into an integer. Default: numpy NaN
    '''
    if string:  # Ensure not length 0 string
        try:
            ret = int(string)
        except:
            ret = replace_with
    else:
        ret = replace_with

    return ret


def grouped_distplot(df, xcol, groupbycol, sortby=None,
                     x_get_rid_of_non_ints=True, make_upper=False,
                     xclip=None, xlim=None, n_to_take=30, logy=False,
                     kde=True, hist=False, rug=False,
                     kde_kws=None, hist_kws=None, rug_kws=None,
                     hist2line=False, hist2linebins=None, zeros2nan=True,
                     fit=None,
                     xname=None, xunits=None, groupname=None):
    """Return a grouped and optionally ordered distribution plot
      (e.g. histogram or KDE). AKA "Ease of repair" plot

    TODO: Add example

    Args:
      df (pandas.DataFrame): Pandas dataframe containing the data
      xcol (str): Column label of df referring to the data to be plotted
        along the x-axis
      groupbycol (str): Column label of df referring to the data to be
        grouped and plotted as individual distributions
      sortby (str or None, optional): How to sort the grouped distributions
        based on their x-axis values. Default is None which will likely
        sort the values alphabetically.  Available options are
        'count_nonzero', 'mean', 'std', 'median', or None
      make_upper (bool, optional): whether to convert the values in the
        groupby column to all uppercase before doing the grouping
      x_get_rid_of_non_ints (bool, optional): try to convert all x values
        to integers before performing remainder of function
      xclip (tuple, optional): 2-tuple in format (xmin, xmax) which
        clips the x-axis values to (xmin <= x <= xmax). NOTE this is
        done BEFORE normalization, is performed, so any values outside
        this range are effectively treated as outliers and ignored.
      xlim (tuple, optional): Plot limit 2-tuple in format (xmin, xmax)
        which sets the range of values to plot. This is done AFTER
        normalization and only affects the visual plot limits, not the
        underlying calculation.
      n_to_take (int, optional): Number of unique group values to consider.
        The n most common groups are plotted.
      logy (bool, optional): Change y-axis scale to logarithmic
      kde (bool, optional): Keyword for sns.distplot
      hist (bool, optional): Keyword for sns.distplot
      rug (bool, optional): Keyword for sns.distplot
      kde_kws (dict, optional): Keyword for sns.distplot
      hist_kws (dict, optional): Keyword for sns.distplot
      rug_kws (dict, optional): Keyword for sns.distplot
      hist2line (bool, optional):
      hist2linebins (list, optional):
      zeros2nan (bool, optional):
      fit (function, optional): Keyword for sns.distplot
      xname (str, optional): Human-friendly name of x-values
      xunits (str, optional): Optional units for x-values to appear in plot
      groupname (str, optional): Human-friendly name of group-values

    Returns:
      pandas.DataFrame: Aggregate x-values for each group containing the
        columns 'count_nonzero', 'mean', 'std', and 'median'.

    Raises:
      TypeError:
        If n_to_take is not an integer
        If xclip has length != 2
        If xlim has length != 2
      ValueError:
        If n_to_take is not positive
        If sortby not in ['count_nonzero', 'mean', 'std', 'median', None]
    """
    if not isinstance(n_to_take, int):
        raise TypeError("n_to_take must be a positive integer.")
    if n_to_take <= 0:
        raise ValueError("n_to_take must be positive.")
    acceptable_sorts = ['count_nonzero', 'mean', 'std', 'median', None]
    if sortby not in acceptable_sorts:
        raise ValueError("sortby must be one of {}".format(acceptable_sorts))
    if xclip is not None:
        if len(xclip) != 2:
            raise TypeError("xclip must be of length 2")
    if xlim is not None:
        if len(xlim) != 2:
            raise TypeError("xlim must be of length 2")

    jd2 = df[[groupbycol, xcol]]

    # TODO: This should be done before passing to this function
    if x_get_rid_of_non_ints:
        jd2.loc[:, xcol] = jd2.loc[:, xcol].apply(get_rid_of_non_ints)

    # Make upper case before grouping.
    # TODO: This should be done before passing to this function
    if make_upper:
        jd2.loc[:, groupbycol] = jd2.loc[:, groupbycol].str.upper()

    if xclip is not None:
        jd2 = jd2[(jd2[xcol] > xclip[0]) & (jd2[xcol] < xclip[1])]

    mygroup = jd2.groupby(groupbycol)

    # Find the most common groups
    # Could perhaps speed this up by filtering beforehand
    # Take the count of the groups, sort by the resultant count values xcol
    top_inds = mygroup.count().sort(columns=xcol,
                                    ascending=False)[0:n_to_take].index

    gragg = mygroup.agg([np.count_nonzero, np.mean, np.std, np.median])[xcol]

    # TODO: Is there a way to do the median only on the filtered groups first?
    if sortby is not None:
        sorted_inds = gragg.loc[top_inds].sort(columns=sortby).index

    # Use a sequential color pallete like coolwarm or RdBu_r
    with sns.color_palette("RdBu_r", len(sorted_inds)):
        plt.figure(figsize=(15, 11))

        if hist2line:
            for ind in sorted_inds:
                # Generate histogram, take center points,
                # and then do a lineplot.
                if hist2linebins is None:
                    hist2linebins = np.arange(len(jd2[xcol].value_counts()))
                thing_to_hist = jd2[(jd2[groupbycol] == ind)][xcol]
                y, x = np.histogram(thing_to_hist,
                                    bins=hist2linebins+1,
                                    normed=True)
                if zeros2nan:
                    # Replace zeros in the histogram with nans
                    y[y == 0] = np.nan
                plt.plot(x[0:-1], y, label=ind)

        else:
            for ind in sorted_inds:
                sns.distplot(jd2[(jd2[groupbycol] == ind)][xcol], label=ind,
                             kde=kde, hist=hist, rug=rug,
                             kde_kws=kde_kws, hist_kws=hist_kws,
                             rug_kws=rug_kws, fit=fit)

        if logy:
            plt.gca().set_yscale('log')
        plt.legend(fontsize=13, loc='upper right')
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])

        if xname is not None and groupname is not None:
            ylab = 'P({0}|{1})'.format(xname, groupname)
            xlab = xname
            if xunits:
                xlab += ' ({})'.format(xunits)
            plt.ylabel(ylab, fontsize=22)
            plt.xlabel(xlab, fontsize=22)
            title = 'Distribution of {0} by {1}'.format(xname, groupname)
            plt.suptitle(title, fontsize=24)
            subtitle = ('Sorted by {0} {1} for {2} most common'
                        ' {3}s'.format(str(sortby), xname,
                                       n_to_take, groupname))
            plt.title(subtitle, fontsize=16)
    return gragg.loc[sorted_inds]


def normed_distplot(df, xcol, groupbycol, xinds=None,
                    sortby='count_nonzero',
                    xclip=None, xlim=None, n_to_take=15, pal=None,
                    ignore_groups=[],
                    xlabel=None, xunits=None, grouplabel=None,
                    rotate_xlabels=False):
    """Return a normalized stacked area chart, aka 'Geometric Abstraction
      plot' or 'Matisse plot'

    TODO: Test this more. See if can reproduce Tai's granularity plot.
    TODO: Add example

    Args:
      df (pandas.DataFrame): Pandas dataframe containing the data
      xcol (str): Column label of df referring to the data to be plotted
        along the x-axis
      groupbycol (str): Column label of df referring to the data to be
        grouped and plotted as sections of the area plot
      xinds (list, optional): Optional list of indices to take for the
        x-axis values
      sortby (str or None, optional): How to sort the grouped area sections
        based on their x-axis values. Default is 'count_nonzero' which
        places the largest areas on the bottom.  Available options are
        'count_nonzero', 'mean', 'std', 'median', or None
      xclip (tuple, optional): 2-tuple in format (xmin, xmax) which
        clips the x-axis values to (xmin <= x <= xmax). NOTE this is
        done BEFORE normalization, is performed, so any values outside
        this range are effectively treated as outliers and ignored.
      xlim (tuple, optional): Plot limit 2-tuple in format (xmin, xmax)
        which sets the range of values to plot. This is done AFTER
        normalization and only affects the visual plot limits, not the
        underlying calculation.
      n_to_take (int, optional): Number of unique group values to consider.
        The n most common groups are plotted. Note: if n > 15, the default
        color palette will have repeated colors.
      pal (list, optional): Color palette to use for plotting. Can be list
        of hex color codes, RGB tuples, or seaborn color_palette objects.
        Default color palette is a combo of colorbrewer Set1 and Dark2.
      ignore_groups (list, optional): List of groups to ignore when plotting,
        even if they are within the n_to_take most common groups.  Useful if
        e.g. "other" is a common category, and you want to keep these from
        being plotted as a labeled group.
      xlabel (str, optional): Human-friendly name of x-values
      xunits (str, optional): Optional units for x-values to appear in plot
      grouplabel (str, optional): Human-friendly name of group-values
      rotate_xlabels (bool, optional): Rotate the xlabels 45 degrees

    Returns:
      pandas.DataFrame: Aggregate x-values for each group containing the
        columns 'count_nonzero', 'mean', 'std', and 'median'.

    Raises:
      TypeError:
        If n_to_take is not an integer
        If xclip has length != 2
        If xlim has length != 2
      ValueError:
        If n_to_take is not positive
        If sortby not in ['count_nonzero', 'mean', 'std', 'median', None]
    """
    if not isinstance(n_to_take, int):
        raise TypeError("n_to_take must be a positive integer.")
    if n_to_take <= 0:
        raise ValueError("n_to_take must be positive.")
    acceptable_sorts = ['count_nonzero', 'mean', 'std', 'median', None]
    if sortby not in acceptable_sorts:
        raise ValueError("sortby must be one of {}".format(acceptable_sorts))
    if xclip is not None:
        if len(xclip) != 2:
            raise TypeError("xclip must be of length 2")
    if xlim is not None:
        if len(xlim) != 2:
            raise TypeError("xlim must be of length 2")

    if pal is None:
        # Default color palette is a combo of colorbrewer Set1 and Dark2
        pal = sns.color_palette("Set1", 8, desat=0.8)
        pal += sns.color_palette("Dark2", 7, desat=0.8)

    # Clip out desired columns
    if xclip is not None:
        df = df[(df[xcol] >= xclip[0]) & (df[xcol] <= xclip[1])]

    # Find the most common groups
    # Could perhaps speed this up by filtering beforehand
    # Take the count of the groups, sort by the resultant count values xcol
    mygroup = df.dropna().groupby([groupbycol])

    top_inds = mygroup.count().sort(columns=xcol,
                                    ascending=False)[0:n_to_take].index

    gragg = mygroup.agg([np.count_nonzero, np.mean, np.std, np.median])[xcol]

    # Sort by desired index
    sorted_inds = gragg.loc[top_inds].sort(columns=sortby,
                                           ascending=False).index

    # Remove unwanted groups:
    for ig_grp in ignore_groups:
        if ig_grp in sorted_inds:
            sorted_inds = sorted_inds.drop(ig_grp)

    if xinds:
        xxx = pd.DataFrame(index=xinds)
    else:
        defaultind = df[xcol].value_counts().index
        xxx = pd.DataFrame(index=defaultind)

    for name, group in mygroup:
        a = pd.Series(group[xcol].value_counts())
        xxx[name] = a.sort_index()

    xxxnorm = xxx.copy()
    for ind, row in xxxnorm.iterrows():
        xxxnorm.loc[ind] = row/row.fillna(0).sum()

    # Making stacked area plot, only feed the top index types
    finalstackdf = xxxnorm.fillna(0).loc[:, sorted_inds].copy()

    sns.set_style("ticks",
                  {"axes.grid": False,
                   "grid.linewidth": 0.0,
                   "axes.facecolor": "#999999"
                   })

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    finalstackdf.fillna(0).plot(ax=ax, kind='area', colors=pal, legend=False)

    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend((handles[::-1]), (labels[::-1]), bbox_to_anchor=(1.32, .8))

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if rotate_xlabels:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    if xlabel is not None and grouplabel is not None:
        ylab = 'P({0}|{1})'.format(xlabel, grouplabel)
        xlab = xlabel
        if xunits:
            xlab += ' ({})'.format(xunits)
        plt.ylabel(ylab, fontsize=22)
        plt.xlabel(xlab, fontsize=22)
        # TODO: Does this name need to be fixed?
        title = 'Distribution of {0} by {1}'.format(xlabel, grouplabel)
        plt.suptitle(title, fontsize=24)

        subtitle = ('Sorted by {0} {1} for {2} most common'
                    ' {3}s'.format(str(sortby), xlabel,
                                   n_to_take, grouplabel))
        plt.title(subtitle, fontsize=16)
    return gragg.loc[sorted_inds]

qpal = sns.color_palette("Set1", 8, desat=0.8)
qpal += sns.color_palette("Dark2", 8, desat=0.8)


def MultiTimeseries(df, column_dict_or_list=None, title=None):
    '''Plot multiple columns in a dataframe with a shared x-axis.

    Arguments:
      df (pandas.DataFrame): DataFrame containing data to plot
      column_dict_or_list (dict or list): Dictionary of format specified
        below, or a list of column names to plot.
      title (str): Title to place at the top of the figure.

    Each key in the configuration dictionary is a column to be plotted
    in the df. The value is a dictionary of keywords, including the desired
    color, marker, line width, and alpha of the points, the label name, and
    the id of the axis to plot.  If one or more of these is not specified
    for one of the columns, then default values are used. The number of
    axes needed for the figure will be automatically determined based on
    the coldict.

    Also within coldict, you can now set the following axis-specific
    keywords:
    axkwdict = {'ylabel': '',
                'xlabel': '',
                'title': '',
                'ylim': None,
                'legend': 1}
    Warning: if multiple plots are being placed on the same axis,
    this could lead to unexpected labeling behavior.

    Example coldict:
      coldict = {
          'Column_A':{
              'marker':'o',
              'color':'red',
              'ax_id':0,
              'label':'Feature A',
              'alpha':0.3,
              'lw':0
              },
          'Column_B':{
              'marker':'.',
              'color':'black',
              'ax_id':0,
              'label':'Feature B',
              'alpha':0.7,
              'lw':1
              },
          'Column_C':{
              'ax_id':1,
              'label':'Feature C',
              'lw':2
              }
          }

    Example Usage:
      see MultiTimeseriesExample() and MultiTimeseriesExample2()
    '''
    # typechecking
    if type(column_dict_or_list) is not dict:
        # if None, use all columns to make dict
        if column_dict_or_list is None:
            print "Column config not specified, using defaults on all columns"
            column_dict_or_list = df.columns
        # If not provided with dict, try making empty dictionary of list items
        coldict = dict(
            zip(
                (column_dict_or_list),
                [{}]*len(column_dict_or_list)
                )
            )
    else:
        coldict = column_dict_or_list

    naxes = 1
    count = 0

    # clean dictionary and find number of axes to plot
    cols_to_plot = {}
    axes_to_adjust = {}
    for colname, colvals in coldict.iteritems():
        if colname in df:

            # defaults to use if not specified by user
            kwdict = {
                'ax_id': count,
                'marker': '.',
                'color': qpal[np.mod(count, len(qpal))],
                'label': colname,
                'alpha': 0.5,
                'lw': 0
            }
            # custom vals
            for kw in kwdict.iterkeys():
                if kw in colvals:
                    kwdict.update({kw: colvals[kw]})

            cols_to_plot.update({colname: kwdict})

            # defaults
            axdict = {}
            for axid in range(naxes):
                axkwdict = {'ylabel': '',
                            'xlabel': '',
                            'title': '',
                            'ylim': None,
                            'legend': 1}
                for kw in axkwdict.iterkeys():
                    if kw in colvals:
                        axkwdict.update({kw: colvals[kw]})

            axes_to_adjust.update({kwdict['ax_id']: axkwdict})
            # store maximum number of axes while looping
            # TODO: fix edge case where ax_id is specified for some but
            #       not all entries in coldict
            naxes = max((naxes, kwdict['ax_id'] + 1))
            count += 1
        else:
            print "Warning: '{}' not in dataframe. Skipping.".format(colname)

    fig = plt.figure(figsize=(16, 2.375*naxes))
    fig.subplots_adjust(hspace=0.2)

    axlist = [fig.add_subplot(naxes, 1, nn) for nn in (range(1, naxes+1))]

    for colname, col in cols_to_plot.iteritems():
        df[colname].plot(ax=axlist[col['ax_id']],
                         lw=col['lw'],
                         marker=col['marker'],
                         alpha=col['alpha'],
                         label=col['label'],
                         color=col['color'])

    for axid, axkws in axes_to_adjust.iteritems():
        ax = axlist[axid]
        for kw, val in axkws.iteritems():
            if val:
                if kw == 'title':
                    ax.set_title(val)
                elif kw == 'xlabel':
                    ax.set_xlabel(val)
                elif kw == 'ylabel':
                    ax.set_ylabel(val)
                elif kw == 'ylim':
                    ax.set_ylim(val[0], val[1])
                elif kw == 'legend':
                    ax.legend(loc=val)

    # remove xticks from mid-plots
    xticklabels = []
    for ax in axlist[:-1]:
        xticklabels += ax.get_xticklabels()
    qqq = plt.setp(xticklabels, visible=False)

    sns.despine()
    if title:
        fig.suptitle(title, fontsize=20)

    # fig.show()
    return fig


def MultiTimeseriesExample():
    '''Example of how to use MultiTimeseries() with a highly customized
    configuration dictionary'''

    from NotebookTools import Paths
    paths = Paths('PBMining')
    paths.add_filepath(
        'datafilename',
        "brushy_creek_milling_merged_130101-140520.pkl",
        base='reduced')

    merged_df = pd.read_pickle(paths.datafilename)

    coldict = {
        'MillFeedXRAYData_ZincAssay': {
            'marker': 'o',
            'color': qpal[0],
            'ax_id': 0,
            'label': 'Mill Feed Zn',
            'alpha': 0.3,
            'lw': 0
            },
        'PbTailXRAYData_ZincAssay': {
            'marker': 'o',
            'color': 'black',
            'ax_id': 0,
            'label': 'Pb Tail Zn',
            'alpha': 0.3,
            'lw': 1
            },
        'CuSO4toZnCond': {
            'marker': 'o',
            'color': qpal[2],
            'ax_id': 1,
            'label': 'CuSO4 to Zn Cond',
            'alpha': 0.3,
            'lw': 0
            },
        'ZnXanthate': {
            'marker': '.',
            'color': qpal[7],
            'ax_id': 2,
            'label': 'Zn Xanthate',
            'alpha': 0.9,
            'lw': 1,
            'ylabel': 'Amount'
            },
        'PbTailXRAYData_IronAssay': {
            'marker': 'o',
            'color': qpal[4],
            'ax_id': 3,
            'label': 'Pb Tail Iron',
            'alpha': 0.7,
            'lw': 0,
            'title': 'YO',
            'ylim': (1.3, 2.0),
            'xlabel': 'Time'
            },
        }

    tw = '5-11-2014'

    fig = MultiTimeseries(merged_df[tw], coldict, title=tw)
    fig.show()


def MultiTimeseriesExample2():
    '''Example of how to use MultiTimeseries() by just feeding a default list
    of columns.
    '''
    from NotebookTools import Paths
    paths = Paths('PBMining')
    paths.add_filepath(
        'datafilename',
        "brushy_creek_milling_merged_130101-140520.pkl",
        base='reduced')

    merged_df = pd.read_pickle(paths.datafilename)
    tw = '5-11-2014'
    columns = ['PbTailXRAYData_ZincAssay',
               'ZnConcX_RayData__ZnValue',
               'ZnTailXRAYData_ZincAssay']
    fig = MultiTimeseries(merged_df[tw], columns, title=tw)
    fig.show()
