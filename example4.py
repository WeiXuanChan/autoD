'''
File: example4.py
Description: calculating matrix for differentiation equation in the space x=(0,200)
                    x (d^2 y/dx^2)-2(dy/dx)-3y =0
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: Dwindz 26Feb2016           - Created
'''

import numpy as np
import autoD_v2 as ad

def caly(x,dOrder,steps,xmin,xmax):
    eqnarray=np.zeros(steps+1)
    deltax=(xmax-xmin)/steps
    #fill eqnarray with finite difference
    index=int((x['x']-xmin)/(xmax-xmin)*steps)
    #for clarity this part is manual, you change change it to a function which calls onto itself for higher dOrder
    if dOrder['x']==0:
        eqnarray[index]=1.
    elif dOrder['x']==1:
        eqnarray[index+1]=1./(2.*deltax)
        eqnarray[index-1]=-1./(2.*deltax)
    elif dOrder['x']==2:
        eqnarray[index+1]=1./deltax/deltax
        eqnarray[index]=-2./deltax/deltax
        eqnarray[index-1]=1./deltax/deltax
    return eqnarray
        
steps=100
xmin=0.
xmax=200.
symbolic_x=ad.Scalar('x')
x=np.array(range(steps+1))/steps*(xmax-xmin)+xmin
y=ad.Function(caly,steps,xmin,xmax)
d0=-3.*y
d1=-2.*ad.Differentiate(y,1)
d2=symbolic_x*ad.Differentiate(y,2)
func=d0+d1+d2
matrix=np.zeros((steps+1,steps+1))

#the first and last row is remove to add in boundary equation
for n in range(1,steps):
    matrix[n,:]=func.cal({'x':x[n]},{})

print(matrix)
