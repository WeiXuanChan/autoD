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
import autoD as ad

x=ad.Scalar('x')

y=ad.Scalar('y')

a=x**2.*ad.Ln(y)

inputPoint={'x':0.2,'y':2.2}
print(a.cal(inputPoint,{'x':1,'y':2}))
print(-2.*inputPoint['x']/inputPoint['y']/inputPoint['y'])
