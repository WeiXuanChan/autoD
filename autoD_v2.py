'''
File: autoD.py
Description: Single Scalar automatic differentiation
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: dwindz 24Feb2016           - Created
  Author: dwindz 25Feb2016           - v2
                                        -include multi-variable entry

'''

'''
Standardized class def: 
func            class object      class object must contain the following function
                                    def cal(self,x,dOrder):
                                        x        float/numpy.1darray
                                        dOrder   float/numpy.1darray
                                        return float results or call to other class object .cal function

Note:
I divided the Class into three types; basic functions, base-end functions and flexible functions.
Basic functions contain class objects that deals with differentiating operations
Base end functions returns the result for any order of differentiation without call to other functions.
Flexible functions accepts user-defined function and turn them into callable objects "func" in this module.
'''

import numpy as np

'''
#---------------Basic Functions-------------------------------#
'''

class Differentiate:
    def __init__(self,func,order):
        self.inputFunc=func
        self.inputorder=order
    def cal(self,x,dOrder):
        return self.inputFunc.cal(x,self.inputorder+dOrder)
        
class Addition:
    def __init__(self,funcList):
        self.funcList=funcList
    def cal(self,x,dOrder):
        
        temp=[]
        for n in range(len(self.funcList)):
            temp.append(self.funcList[n].cal(x,dOrder))
        result=sum(temp)
        return result
              
class Multiply:
    def __init__(self,funcList):
        self.funcList=funcList
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            result=1.
            for n in range(len(self.funcList)):
                result=result*self.funcList[n].cal(x,next_cal)
            return result
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_dOrder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            new_mul=[]
            for n in range(len(self.funcList)):
                newList=[]
                for m in range(len(self.funcList)):
                    if m==n:
                        newList.append(Differentiate(self.funcList[n],diffchange))
                    else:
                        newList.append(self.funcList[m])
                new_mul.append(Multiply(newList))
            new_add=Addition(new_mul)
            return new_add.cal(x,new_dOrder) 

class Power:
    def __init__(self,func,pow):
        self.func=func
        self.pow=pow
        
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            if isinstance(self.pow, (int, float)):
                return self.func.cal(x,next_cal)**self.pow
            else:
                return self.func.cal(x,next_cal)**self.pow.cal(x,next_cal)
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_dOrder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            if isinstance(self.pow, (int, float)):
                new_const=Constant(self.pow)
                new_pow=Power(self.func,self.pow-1.)
                new_diff=Differentiate(self.func,diffchange)
                new_mul=Multiply([new_pow,new_diff,new_const])
                return new_mul.cal(x,new_dOrder)
            else:
                new_exp=Exp(Multiply([Ln(self.func),self.pow]))
                return new_exp.cal(x,dOrder)

class Exp:
    def __init__(self,func):
        self.func=func
        
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            return np.exp(self.func.cal(x,next_cal))
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_dOrder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            new_mul=Multiply([Exp(self.func),Differentiate(self.func,diffchange)])
            return new_mul.cal(x,new_dOrder)
class Ln:
    def __init__(self,func):
        self.func=func
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            return np.log(self.func.cal(x,next_cal))
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_dOrder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            new_power=Power(self.func,-1.)
            new_diff=Differentiate(self.func,diffchange)
            new_mul=Multiply([new_power,new_diff])
            return new_mul.cal(x,new_dOrder)

class Log:
    def __init__(self,func,base):
        self.func=func
        self.base=base
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            if isinstance(self.base, (int, float)):
                return np.log(self.func.cal(x,next_cal))/np.log(self.base)
            else:
                return np.log(self.func.cal(x,next_cal))/np.log(self.base.cal(x,next_cal))
        else:
            if isinstance(self.base, (int, float)):
                return np.log(self.func.cal(x,dOrder))/np.log(self.base)
            else:
                new_mul=Multiply([Ln(self.func),Power(Ln(self.base),-1.)])
                return new_mul.cal(x,dOrder)
            
class Cos:
    def __init__(self,func):
        self.func=func
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            return np.cos(self.func.cal(x,next_cal))
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_dOrder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            new_sin=Sin(self.func)
            new_diff=Differentiate(self.func,diffchange)
            new_mul=Multiply([new_sin,new_diff,Constant(-1.)])
            return new_mul.cal(x,new_dOrder)
        
class Sin:
    def __init__(self,func):
        self.func=func
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            if isinstance(dOrder, (int,float)):
                next_cal=0
            else:
                next_cal=np.zeros(len(dOrder))
            return np.sin(self.func.cal(x,next_cal))
        else:
            if isinstance(dOrder, (int,float)):
                diffchange=1
                new_drOder=dOrder-1
            else:
                diffchange=np.zeros(len(dOrder))
                diffchange[check[0]]=1
                new_dOrder=dOrder-diffchange
            new_sin=Cos(self.func)
            new_diff=Differentiate(self.func,diffchange)
            new_mul=Multiply([new_sin,new_diff])
            return new_mul.cal(x,new_dOrder)
        
'''
#---------------Base End Functions-------------------------------#
'''
            
class Constant:
    def __init__(self,const):
        self.const=const
    def cal(self,x,dOrder):
        check=np.nonzero(dOrder)[0]
        if check.size==0:
            return self.const
        else:             
            return 0.          
class Scalar:
    def __init__(self,name='x',index=0):
        self.name=name
        self.index=index
    def cal(self,x,dOrder):
        if isinstance(dOrder, (int,float)): 
            if dOrder==0:
                return x
            elif dOrder==1:
                return 1.
            else:
                return 0.
        else:
            for n in range(len(dOrder)):
                if n!= self.index:
                    if dOrder[n]>0:
                        return 0.
            if dOrder[self.index]==0:
                return x[self.index]
            elif dOrder[self.index]==1:
                return 1.
            else:
                return 0.
'''
#---------------Flexible Functions-----------------#
'''
class Function:
    def __init__(self,func,*args):
        self.func=func
        self.args=args
    def cal(self,x,dOrder):
        args=self.args
        return self.func(x,dOrder,*args)
        
    def changeArgs(self,*new_args):
        self.args=new_args
    
    
    
    