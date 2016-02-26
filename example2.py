'''
File: example2.py
Description: Example of importing self-defining function into autoD
                calculate d/dx[ln(integral of x)] and d/dx[ln(integral of x^2)] at x=0.2, starting integration at x=0.
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: Dwindz 26Feb2016           - Created
'''

import numpy as np
import autoD_v1 as ad

def func(x,dOrder,fx,startIntegration):
    if dOrder>0:
        return fx.cal(x,dOrder-1)
    else:
        xsteps=np.array(range(101))/100*(x-startIntegration)+startIntegration
        tosum=np.zeros(101)
        for n in range(101):
            tosum[n]=fx.cal(xsteps[n],0)
        return np.trapz(tosum,x=xsteps)


x=ad.Scalar()
a=ad.Power(x,2.)
integral=ad.Function(func,x,0.)
b=ad.Ln(integral)
print(b.cal(0.2,1))
integral.changeArgs(a,0.)
print(b.cal(0.2,1))