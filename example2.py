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
import autoD as ad

def func(x,dOrder,fx,startIntegration):
    if dOrder['x']>0:
      new_dOrder=dOrder.copy()
      new_dOrder['x']=dOrder['x']-1
      return fx.cal(x,new_dOrder)
    else:
        xsteps=np.array(range(101))/100*(x-startIntegration)+startIntegration
        tosum=np.zeros(101)
        for n in range(101):
            tosum[n]=fx.cal(xsteps[n],0)
        return np.trapz(tosum,x=xsteps)


x=ad.Scalar('x')
a=x**2.
integral=ad.Function(func,x,0.)
b=ad.Ln(integral)
print(b.cal({'x':0.2},{'x':1}))
integral.changeArgs(a,0.)
print(b.cal({'x':0.2},{'x':1}))
