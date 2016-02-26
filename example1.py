'''
File: example1.py
Description: Simple example
                calculate d/dx[ 2cos(x^2)-d/dx[sin(ln(x))] ] at x=0.2
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: Dwindz 26Feb2016           - Created
'''

import numpy as np
import autoD_v1 as ad

x=ad.Scalar()
a=ad.Multiply([ad.Constant(2.),ad.Cos(ad.Power(x,2.))])
b=ad.Differentiate(ad.Sin(ad.Ln(x)),1)
fx=ad.Addition([a,ad.Multiply([ad.Constant(-1),b])])
print(fx.cal(0.2,1))