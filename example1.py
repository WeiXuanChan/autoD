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
import autoD as ad

x=ad.Scalar('x')
a=ad.Multiply([2.,ad.Cos(ad.Power(x,2.))])
b=ad.Differentiate(ad.Sin(ad.Ln(x)),1)
fx=ad.Addition([a,ad.Multiply([-1,b])])
print(fx.cal({'x':0.2},{'x':1}))
