from base import data
try:
    import pandas as pd
except ImportError:
    pd = None
import numpy as np
import warnings

if pd is None:
    warnings.warn("Pandas is not installed, the 'tabular' "
                  "class has reduced functionality.")


class tabular(data):
    """
    A class for holding tabular (e.g. 'spreadsheet') type data.

    This data-type is assumed to be planar (2-D, rows and columns)
    only.
    """

    if pd is None:

        def to_dataframe(self, ):
            raise Exception("Install pandas to write to a dataframe.")

        def to_excel(self, ):
            raise Exception("Install pandas to write to excel files.")

    else:

        def to_dataframe(self,):
            siteout = None
            for nm, val in self.iteritems():
                if val.ndim == 1:
                    if siteout is None:
                        siteout = pd.DataFrame(val, columns=[nm])
                    else:
                        siteout.loc[:, nm] = pd.Series(val)
                else:
                    siteout[nm] = pd.DataFrame(val)
            return siteout

        def to_excel(self, fname):
            out = {}
            buf = pd.io.excel.ExcelWriter(fname)
            siteout = self.to_dataframe()
            siteout.to_excel(buf, sheet_name='Site')
            for nm in out:
                if np.iscomplex(out[nm]).any():
                    out[nm].astype('S').to_excel(buf, sheet_name=nm)
                else:
                    out[nm].to_excel(buf, sheet_name=nm)
            buf.close()
