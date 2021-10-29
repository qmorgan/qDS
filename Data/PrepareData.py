import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Helpers.NotebookTools import Paths
from .Aggregate import replace_values


class DataBlock(object):
    '''Generalized Data Block Class

    Arguments:
      df (pandas.DataFrame):
      coldict (dict):

    Methods:
      ...

    Attributes:
      coldict (dict)
      coldf (pandas.DataFrame)
      df (pandas.DataFrame)
      N_raw (int)
      bool_duplicate_indices_removed (bool)
      bool_time_shifts_applied (bool)
      bool_missing_values_removed (bool): True if remove_missing_values() has
        already been run.
      bool_repeated_values_removed (bool): True if remove_repeated_values() has
        already been run.
      bool_suspicious_values_converted (bool): True if
        convert_suspicious_values() has already been run.
      log (list): List of operations applied to the DataBlock
      paths (NotebookTools.Paths)
      name: String name of datablock 
    
      TODO: Move NotebookTools.Paths to this parent class
      TODO: Complete Documentation
      TODO: Merge with Plotomatic.MegaDF?

    Example Usage:
      >>>

    '''
    def __init__(self,
                 df=None,
                 coldict=None,
                 name=''
                 ):

        if not hasattr(self, 'df'):
            if df is not None:
                self.df = df
            else:
                print("Please specify df")
                return

        # TODO: Default behavior if coldict not specified?
        if not coldict:
            print("coldict not specified; using all columns with no parameters")
            coldict = {}
            for key in df.columns:
                coldict.update({key:{}})

        self.coldict = coldict
        self.coldf = pd.DataFrame(coldict).T
        self.name = name
        

        self.update_N()
        self.N_raw = self.N
        self.bool_duplicate_indices_removed = False
        self.bool_time_shifts_applied = False
        self.bool_missing_values_removed = False
        self.bool_repeated_values_removed = False
        self.bool_suspicious_values_converted = False
        self.log = []

        # TODO: Make paths object part of this parent class
        self.paths = None

    def add_paths(self):
        '''Add paths to the Paths object, which defines where to look to load
        the data.
        
        TODO: Add documentation!!
        '''
        if not hasattr(self, 'pathlist'):
            print("DataBlock does not have pathlist specified")
            return
        if not hasattr(self, 'paths'):
            print("DataBlock does not contain Paths object")
        for path in self.pathlist:
            self.paths.add_filepath(path['name'],
                                    path['path'],
                                    base=path['base'])

    def describe(self, 
                 include=None, 
                 exclude=None, 
                 dropna=False,
                 print_metadata=True):
        # TODO: This is probably inefficient
        if (include is None and exclude is None):
            cols = self.df.columns
        else:
            cols = list(self.df.select_dtypes(include=include, exclude=exclude).columns)

        desc = getattr(self,'_desc',None)
        if desc is None:
            self.update_describe()
            desc = getattr(self,'_desc')
        
        if print_metadata:
            print("Summary for DataFrame '{}'".format(self.name))
        
        if dropna:
            return desc[cols].dropna()
        else:
            return desc[cols]
        
        
    def update_describe(self):
        # np.generic includes all types, including object, datetime, etc
        # np.number would just be numeric types
        desc = self.df.describe(include='all')
        desc.loc['dtypes'] = self.df.dtypes
        desc.loc['missing'] = len(self.df) - desc.loc['count']
        desc.loc['missing_frac'] = desc.loc['missing']/float(len(self.df))
        # includes nan! unlike the default unique for object types. 
        # TODO: This might be very slow for large datasets of numeric data
        desc.loc['unique_vals'] = self.df.apply(lambda x: len(pd.unique(x))) 
        desc.loc['top_frac'] = desc.loc['freq']/float(len(self.df))
        self._desc = desc
        
    def update_N(self):
        '''Re-determine the number of non-null values in self.df
        '''
        self.update_describe()
        self.N = self.describe(print_metadata=False).loc['count']

    def drop_duplicate_indicies(self):
        '''Checks if there are any duplicate indices (e.g. timestamps)
        
        In the off-chance there are duplicates, take the first value.
        Prints a warning if the duplicate indices do not have identical values.
        '''
        if self.bool_duplicate_indices_removed:
            print ('WARNING Duplicate entries had already been removed')

        grouped = self.df.groupby(level=0)
        groupedfirst = grouped.first()

        # confirm that the repeat indices are actually the same values
        if not (groupedfirst-grouped.last()).sum().any():
            print ("WARNING: Duplicate indices do not contain identical values")

        outstr = ("Removed {} duplicate indices"
                  "".format(len(self.df) - len(groupedfirst)))
        print(outstr)
        self.log.append(outstr)

        # TODO: update_N()?

        self.df = groupedfirst
        groupedfirst = 0

        self.bool_duplicate_indices_removed = True

    def remove_missing_values(self, to_replace=0):
        '''
        Replaces all instances in self.df with values of to_replace with np.nan

        Arguments:
          to_replace: Value in dataframe to consider as "missing" and replace
            with NaN.

        TODO: Should this just be integrated with convert_suspicious_values()
          or left separate?

        TODO: Currently applies to entire dataframe. Make it column specific?
        '''
        if self.bool_missing_values_removed:
            print ('WARNING Missing entries had already been removed')

        N_before = self.N
        self.df = self.df.replace(to_replace=to_replace, value=np.nan)
        self.update_N()
        outstr = "Marked the following # of values as missing: \n"
        outstr += str(N_before - self.N)
        print(outstr)
        self.log.append(outstr)

        self.bool_missing_values_removed = True

    def remove_repeated_values(self, jump_limit=0.75):
        '''
        For certain timeseries datasets, such as those which have been
        upsampled to a frequency that is shorter than the frequency at which
        new measurements are taken, the values may have been forward-filled to
        fill in the gaps of the resampled timeseries.

        This method attempts to undo the forward-filling, and replace the
        repeated values with NaNs.

        This operation is only applied to attributes in the coldict which have
        the value 'remove_repeats' == True.

        TODO: Separate the Percent Jump removal to different method!! Could
          perhaps do this without doing another loop through the columns
          since self.df.loc[:,col+'_delta'] is stored.

        For each of these timeseries vectors, sequential jumps which are too
        large to be considered realistic are also removed.
        The cutoff is defined as abs((x[i] - x[i-1])/(x[i])) > 0.75

        If the cutoff is violated, both x[i] and x[i+1] are removed.

        TODO: Check if this is correct, and the desired behavior! Should it
          be x[i-1]?

        '''
        if self.bool_repeated_values_removed:
            print ('WARNING repeated entries had already been removed')

        col_list = self.coldf.index[self.coldf['remove_repeats'] == True]
        for col in col_list:
            series = self.df[col]
            len1 = len(series.dropna())
            resid = series-series.shift(1)
            # remove values where residual is zero
            self.df.loc[:, col][resid == 0] = np.nan
            len2 = len(self.df.loc[:, col].dropna())
            outstr = "{}: {} repeats removed".format(col, len1 - len2)
            print(outstr)
            self.log.append(outstr)

            # TODO: update_N()?

            # Store the x[i] - x[i-1] values as '<colname>_delta' in self.df
            resid = resid.replace(to_replace=0, value=np.nan)
            self.df.loc[:, col+'_delta'] = resid

            # remove large jumps
            # TODO: Have jump_limit configurable in coldict? 
            self._remove_large_jumps(col, jump_limit=jump_limit)

            len3 = len(self.df.loc[:, col].dropna())
            outstr = "{}: {} large deltas removed".format(col, len2 - len3)
            print(outstr)
            self.log.append(outstr)

            # TODO: update_N()?

        self.bool_repeated_values_removed = True
    
    def _remove_large_jumps(self, col, jump_limit=0.75):
        '''Mark large changes over short time periods as missing data points.
        
        # TODO: Have jump_limit configurable in coldict? 
        '''
        series = self.df[col]
        resid = self.df.loc[:, col+'_delta']

        outstr = "Removing large (t-1) deltas greater than {}%".format(jump_limit*100)
        print(outstr)
        self.log.append(outstr)

        percent_jump = (resid/series).abs()
        pct_bool = (percent_jump > jump_limit)

        self.df.loc[:, col][pct_bool] = np.nan
        self.df.loc[:, col][pct_bool.shift(-1).fillna(False)] = np.nan
    
    def convert_suspicious_values(self):
        '''
        If coldict contains a rule_list for a parameter, replace values
        using the replace_values() method from Aggregate.py according to
        the rules. 
        
        rule_list: dict
            operator: str in [<, >, <=, >=, ==, !=]
            comparison_val:
            replace_val:
        
        '''
        if self.bool_suspicious_values_converted:
            print ('WARNING suspicious entries had already been converted')

        rules = self.coldf['rule_list'].dropna()
        for col, rule_list in rules.items():
            outstr = 'Converted suspicious values for {}'.format(col)
            print(outstr)
            self.log.append(outstr)

            for rule_dict in rule_list:
                self.df.loc[:, col] = replace_values(self.df.loc[:, col],
                                                     rule_dict)

        # TODO: update_N()?

        self.bool_suspicious_values_converted = True

    def impute_missing_values(self, columns=None, value=None, method=None, limit=None):
        '''Impute missing values in self.df by either specifying values
        directly, or by specifying a fillna or imputation method.

        Arguments:
          columns (list or None, optional): List of columns to apply
            imputation function to. If None, apply to all columns in self.df
          value (multiple options): Value to use to fill missing values
            (e.g. 0), alternately a dict/Series/DataFrame of values specifying
            which value to use for each index (for a Series) or column (for a
            DataFrame). e.g. to fill columns 'B' thru 'C' with their respective
            means, use value = df.mean()['B':'C']
          method (str, optional): method taken by either pd.DataFrame.fillna(),
            or pd.DataFrame.interpolate()
              fillna methods:
                Method	           Action
                pad / ffill	       Fill values forward
                bfill / backfill   Fill values backward
              interpolate methods:
                'linear', 'time', 'index', 'values', 'nearest', 'zero',
                'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh',
                'polynomial', 'spline', 'piecewise_polynomial', 'pchip'
        '''
        #TODO: Certain interpolate methods require additional keywords, 
        # e.g. 'spline' requires 'order'  

        fillna_methods = ['pad', 'ffill', 'bfill', 'backfill']
        interpolate_methods = ['linear', 'time', 'index', 'values', 'nearest',
                               'zero', 'slinear', 'quadratic', 'cubic',
                               'barycentric', 'krogh', 'polynomial', 'spline',
                               'piecewise_polynomial', 'pchip']

        if columns is None:
            print("Columns unspecified; applying to all columns")
            columns = self.df.columns
        self.check_columns(columns)
        
        # check that there are actually NaNs to fill:
        completely_filled_columns = self.df[columns].columns[self.df[columns].isnull().sum() == 0]
        if len(completely_filled_columns) > 0:
            outstr = "Warning! no NaN values to compute in following columns: "
            for col in completely_filled_columns: 
                outstr += ''
                outstr += ' * {}'.format(col)
            print(outstr)
            self.log += outstr
        
        if value is not None:
            # apply single value to columns
            if method is not None:
                raise ValueError("Either specify value OR method, not both")
            outstr = ('Imputing {} columns with value {}').format(len(columns), value)
            print(outstr) 
            self.log += outstr
            self.df[columns].fillna(value, inplace=True)

        elif method is not None:
            if method in fillna_methods:
                outstr = ('Imputing {} columns using fillna method {} '
                          'and limit {}').format(len(columns), method, limit)
                print(outstr) 
                self.log += outstr
                # Seem to need to do it this way to update in place
                self.df.loc[:,columns] = self.df.loc[:,columns].fillna(method=method, axis=0, limit=limit)
            elif method in interpolate_methods:
                outstr = ('Imputing {} columns using interpolate method {} '
                          'and limit {}').format(len(columns), method, limit)
                print(outstr) 
                self.log += outstr
                self.df.loc[:,columns] = self.df.loc[:,columns].interpolate(method=method, axis=0, limit=limit)
            elif method == 'cubic_spline_0_smoothing':
                # Special method calling cubic spline with no smoothing, which
                # should be similar to the 'cubic' interpolate method but is
                # much faster (http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html#scipy.interpolate.UnivariateSpline)
                outstr = ('Imputing {} columns using interpolate method {} '
                          'and limit {}').format(len(columns), method, limit)
                print(outstr) 
                self.log += outstr
                self.df.loc[:,columns] = self.df.loc[:,columns].interpolate(method='spline', order=3, s=0., axis=0, limit=limit)
                
            else:
                raise ValueError("Method {} not in acceptable fillna/"
                                 "interpolate methods".format(method))
        else:
            print("Nothing to do.. specify value or method")
            self.log += "WARNING: Tried to impute.. but no method specified"

        # TODO: update_N()?

    def correct_systematic_offsets(self):
        raise NotImplementedError()

    def apply_time_shifts(self, freq='5min'):
        '''
        TODO: Ensure that data is timeseries! 
        
        Creates column of shifted values according to shift in coldict

        In [45]: db.df[['PbTailXRAYData_ZincAssay','PbTailXRAYData_ZincAssay-3x5min']][510:520]
        Out[45]:
                             PbTailXRAYData_ZincAssay  PbTailXRAYData_ZincAssay-3x5min
        2013-01-02 18:30:00                  0.385685                         0.406947
        2013-01-02 18:35:00                  0.385000                         0.389018
        2013-01-02 18:40:00                  0.414458                         0.387351
        2013-01-02 18:45:00                  0.435000                         0.385685
        2013-01-02 18:50:00                       NaN                         0.385000
        2013-01-02 18:55:00                  0.426531                         0.414458
        2013-01-02 19:00:00                  0.419704                         0.435000
        2013-01-02 19:05:00                  0.418142                              NaN
        2013-01-02 19:10:00                  0.416579                         0.426531
        2013-01-02 19:15:00                  0.415938                         0.419704

        '''
        # rng = pd.date_range(start="2014-10-07",periods=10,freq='2min')
        # ts = pd.Series(data = list(range(10)), index = rng)
        # ts.resample('2min',base=1)
        # ts2 = ts[ts !=5]  # Handling missing values
        # ts2.shift(1)
        # ts2.shift(1,freq='2min')
        # ts2.shift(1,freq='2min').resample('2min')
        # ts2.shift(2,freq='2min').resample('2min') # same as:
        # ts2.shift(1,freq='4min').resample('2min')

        if self.bool_time_shifts_applied:
            print("Warning: Time shifts already applied!")
            # raise Exception("Time shifts already applied; cannot re-apply")
        # keep original
        # self.df_unshifted = self.df

        col_list = self.coldf.index[self.coldf['t_shift'].isnull() == False]
        for col in col_list:
            timelag = self.coldf.loc[col]['t_shift']
            newstr = "{0}-{1:d}x{2}".format(col, int(timelag), freq)
            print("Adding {}".format(newstr))
            self.log.append("Adding {}".format(newstr))

            self.df[newstr] = self.df[col].shift(timelag,
                                                 freq=freq).resample(freq)

        self.bool_time_shifts_applied = True

        # TODO: update_N()?

    def apply_ewma(self, col, span=5, ignore_na=False,
                   persist_missing=True, make_previous_missing=True):
        '''
        Create new feature applying an exponential weighted moving average
        (ewma) to a given column.  
        
        TODO: First ensure that the dataframe is a timeseries! 
        
        Arguments:
            col: column upon which to apply
            span: ewma keyword (see pandas.ewma)
            ignore_na: ewma keyword (see pandas.ewma)
            presist_missing: mark originally missing values as missing.
                By default, pandas ewma seems to interpolate missing values,
                So this will remove all values that were missing originally 
            make_previous_missing: mark values with missing (t-1) value as 
                missing
        '''
        newname = col+'_ewma'+str(span)
        
        self.add_feature(newname,
                         pd.ewma(self.df[col],
                                 span=span,
                                 ignore_na=ignore_na
                                 )
                         )
        
        # Mark original missing values as missing
        if persist_missing:
            outstr = "{}: Marking original missing vals as missing".format(newname)
            print(outstr)
            self.log.append(outstr)
            self.df.loc[self.df[col].isnull(), newname] = np.nan

        # mark where the previous value is missing as missing as well
        if make_previous_missing:
            outstr = "{}: Marking values with missing (t-1) value as missing".format(newname)
            print(outstr)
            self.log.append(outstr)
            self.df.loc[self.df[col].shift(1).isnull(), newname] = np.nan


    def downsample_values(self, freq='15min', how='median'):
        '''
        Resample a timeseries dataframe to a new frequency.  
        
        Overwrites self.df with a new dataframe resampled to the new frequency
        
        Arguments:
            freq: 
                Frequency to be sampled to. 
                dtype: str
                Default: '15min'
            how: 
                Method by which to do the resampling
                dtype: str
                default: 'median'
         
        '''
        # TODO: allow for time lag
        # TODO: allow for weighted means
        # Condense the data by taking the median at each 15 min interval, in
        #  order to reduce null values.
        # aggXY = pd.DataFrame(columns = XY.columns)
        self.df_copy = self.df.copy()  # TODO: change this name
        self.df = pd.DataFrame(columns=self.df_copy.columns)
        for col in self.df.columns:
            print('Resampling {}'.format(col))
            self.df[col] = self.df_copy[col].resample(freq, how=how)
        outstr = "{} resampled to {}".format(how, freq)
        print(outstr)
        self.log.append(outstr)

        # TODO: update_N()?

    def add_feature(self, feature_name, function):
        '''Generic function to operate on the dataframe and add new column
        '''
        self.df.loc[:, feature_name] = function
        outstr = 'Adding feature: {}'.format(feature_name)
        print(outstr)
        self.log.append(outstr)

        # TODO: update_N()?

    def print_n_missing(self):
        '''Print the number of missing values for each column in self.df
        '''
        N = self.N
        self.update_N()
        
        # if not, I should have updated somewhere
        if (N.sum() != self.N.sum()):
            print("WARNING: Should have updated N?")
        
        msg = "Remaining Missing Values:\n"
        self.log.append(msg)
        print(msg)

        # N_missing = self.N*-1 + len(self.df)
        N_missing = self.describe().loc['missing']
        print(N_missing)
        self.log.append(N_missing)

    def drop_missing(self, columns=None):
        '''Drop rows where entries in any of the specified columns are
        missing'''

        if columns is None:
            print("Columns unspecified; applying to all columns")
            columns = self.df.columns
        self.check_columns(columns)
        old_len = len(self.df)
        self.df.dropna(axis=0, subset=columns, inplace=True)
        new_len = len(self.df)
        outstr = ("Dropped {} rows with NaN in columns:{}"
                  "".format(old_len-new_len, columns))
        print(outstr)
        self.log.append(outstr)

        # TODO: update_N()?

    def rename_columns(self):
        raise NotImplementedError()
        # df.rename(columns={'$a': 'a', '$b': 'b'}, inplace=True)

    def check_columns(self, columns=None):
        '''When called with no arguments, print the names of all available 
        columns.
        
        Given a list of column names, check whether every column exists in 
        self.df 
        
        If not, raise an error. 
        '''
        if columns is None:
            print("Available columns: ")
            for col in self.df.columns:
                print("'{}',".format(col))
            return
        else:
            columns = list(columns)
            for col in columns:
                if col not in self.df:
                    raise ValueError('Column {} not in dataframe!'.format(col))


    def select_columns_to_keep(self, columns=None):
        '''Remove columns from self.df to clear up some memory.  
    
        TODO: Add documentation!!
        '''
        self.check_columns(columns=columns)
        old_length = len(self.df.columns)
        if columns:
            self.df = self.df[columns]
            new_length = len(self.df.columns)
            outstr = ("Retained {} columns ({} removed)"
                      "".format(new_length, old_length-new_length))
            print(outstr)
            self.log.append(outstr)

        # TODO: update_N()?

    def select_training_columns(self, columns=None):
        '''
            TODO: Make this a method of Model Class instead of DataBlock?
            TODO: Add documentation!!
        '''
        print("WARNING: DEPRECIATED. Column Selection now in Models module.")
        self.check_columns(columns=columns)
        if columns:
            self.training_columns = columns

    def select_target_column(self, column=None):
        '''
            TODO: Make this a method of Model Class instead of DataBlock?
            TODO: Add documentation!!
        '''
        print("WARNING: DEPRECIATED. Column Selection now in Models module.")
        self.check_columns(columns=[column])
        if column:
            if hasattr(self, 'training_columns'):
                if column in self.training_columns:
                    raise ValueError("Target cannot be a training column!")
            self.target_column = column


class BCData(DataBlock):
    '''Subclass of DataBlock specific to brushy-creek dataset 

        TODO: Add documentation!!
    '''
    def __init__(self,
                 pathlist=[{'name': 'dictfilename',
                            'path': "brushy_creek_milling_processes.pkl",
                            'base': 'reduced'},
                           {'name': 'datafilename',
                            'path': "brushy_creek_milling_merged_130101-140520.pkl",
                            'base': 'reduced'}],
                 coldict=None
                 ):

        if not coldict:
            print("Please specify coldict")
            return

        self.paths = Paths('PBMining')
        self.pathlist = pathlist
        self.add_paths()

        # Try to load the data
        try:
            merged_df = pd.read_pickle(self.paths.datafilename)
        except:
            print("Cannot load the data...")
            raise IOError("Cannot load the data")

        try:
            datadict_df = pd.read_pickle(self.paths.dictfilename)
        except:
            print("Cannot load the data dict.. ")

        self.df = merged_df[list(coldict.keys())]
        self.df_raw = self.df.copy()

        DataBlock.__init__(self, coldict=coldict)

    def add_previous_assay_values(self):
        '''Helper function to grab the previous assay value for the zinc 
        concentrate and tailing.  Saves the current values as 
            current_Ezn
            current_Dzn
        and the previous values as 
            previous_Ezn
            previous_Dzn
        and the difference between the two as 
            current_delta_Ezn
            current_delta_Dzn
                
        TODO: Will need to add additional assays when we move to those circuits
        '''
        timelag = 1
        # adding the change in Y as an attribute
        self.df['current_Ezn'] = self.df.ZnConcX_RayData__ZnValue
        self.df['previous_Ezn'] = self.df.ZnConcX_RayData__ZnValue.shift(timelag)
        self.df['current_delta_Ezn'] = self.df['current_Ezn'] - self.df['previous_Ezn']

        self.df['current_Dzn'] = self.df.ZnTailXRAYData_ZincAssay
        self.df['previous_Dzn'] = self.df.ZnTailXRAYData_ZincAssay.shift(timelag)
        self.df['current_delta_Dzn'] = self.df['current_Dzn'] - self.df['previous_Dzn']

        self.add_previous_B_values()

    def add_previous_B_values(self):
        '''Helper function to grab the previous assay value for the zinc 
        concentrate and tailing.  
        
        Bit hacky.. depends whether resampling has been done or not
        
        Saves the current values as
            current_Bzn
            previous_Bzn_preresampling
            previous_Bzn_postresampling
        '''
        
        try:
            self.df['current_Bzn'] = self.df['PbTailXRAYData_ZincAssay']
            self.df['previous_Bzn_preresampling'] = self.df['PbTailXRAYData_ZincAssay'].shift(timelag)
        except:
            print("Problem with grabbing pre-resampledPbTail Assay")
        try:
            self.df['previous_Bzn_postresampling'] = self.df['PbTailXRAYData_ZincAssay-3x5min']
        except:
            print("Problem with grabbing post-resampled PbTail Assay")

    def test_function(self):
        return self.df.ZnConcX_RayData__ZnValue

    def test_add_feature(self):
        self.add_feature('rename', self.test_function())

