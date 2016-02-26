'''
File: example3.py
Description: Example of importing self-defining function into autoD
                    calculate d/dxdy^2[x^2 ln(y)]
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: Dwindz 26Feb2016           - Created
'''

import numpy as np
import autoD_v2 as ad

x=ad.Scalar(index=0)

y=ad.Scalar(index=1)

a=ad.Multiply([ad.Power(x,2.),ad.Ln(y)])

l1=0.2
l2=2.2
print(a.cal(np.array([l1,l2]),np.array([1,2])))
print(-2.*l1/l2/l2)