'''
File: autoD.py
Description: Single Scalar automatic differentiation
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: dwindz 24Feb2016           - Created

'''

'''
Standardized class def: 
func            class object      class object must contain the following function
                                    def cal(self,x,dOrder):
                                        x        float
                                        dOrder   float
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
        
        temp=np.zeros(len(self.funcList))
        for n in range(len(self.funcList)):
            temp[n]=self.funcList[n].cal(x,dOrder)
        result=np.sum(temp)
        return result
              
class Multiply:
    def __init__(self,funcList):
        self.funcList=funcList
    def cal(self,x,dOrder):
        if dOrder==0:
            result=1.
            for n in range(len(self.funcList)):
                result=result*self.funcList[n].cal(x,0)
            return result
        else:
            
            new_mul=[]
            for n in range(len(self.funcList)):
                newList=[]
                for m in range(len(self.funcList)):
                    if m==n:
                        newList.append(Differentiate(self.funcList[n],1))
                    else:
                        newList.append(self.funcList[m])
                new_mul.append(Multiply(newList))
            new_add=Addition(new_mul)
            return new_add.cal(x,dOrder-1) 

class Power:
    def __init__(self,func,pow):
        self.func=func
        self.pow=pow
        
    def cal(self,x,dOrder):
        if dOrder==0:
            if isinstance(self.pow, (int, float)):
                return self.func.cal(x,0)**self.pow
            else:
                return self.func.cal(x,0)**self.pow.cal(x,0)
        else:
            if isinstance(self.pow, (int, float)):
                new_const=Constant(self.pow)
                new_pow=Power(self.func,self.pow-1)
                new_diff=Differentiate(self.func,1)
                new_mul=Multiply([new_pow,new_diff,new_const])
                return new_mul.cal(x,dOrder-1)
            else:
                new_exp=Exp(Multiply([Ln(self.func),self.pow]))
                return new_exp.cal(x,dOrder)

class Exp:
    def __init__(self,func):
        self.func=func
        
    def cal(self,x,dOrder):
        if dOrder==0:
            return np.exp(self.func.cal(x,0.))
        else:
            new_mul=Multiply([Exp(self.func),Differentiate(self.func,1)])
            return new_mul.cal(x,dOrder-1)
class Ln:
    def __init__(self,func):
        self.func=func
    def cal(self,x,dOrder):
        if dOrder==0:
            return np.log(self.func.cal(x,0))
        else:
            new_power=Power(self.func,-1.)
            new_diff=Differentiate(self.func,1)
            new_mul=Multiply([new_power,new_diff])
            return new_mul.cal(x,dOrder-1)

class Log:
    def __init__(self,func,base):
        self.func=func
        self.base=base
    def cal(self,x,dOrder):
        if dOrder==0:
            if isinstance(self.base, (int, float)):
                return np.log(self.func.cal(x,0))/np.log(self.base)
            else:
                return np.log(self.func.cal(x,0))/np.log(self.base.cal(x,0))
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
        if dOrder==0:
            return np.cos(self.func.cal(x,0))
        else:
            new_sin=Sin(self.func)
            new_diff=Differentiate(self.func,1)
            new_mul=Multiply([new_sin,new_diff,Constant(-1.)])
            return new_mul.cal(x,dOrder-1)
        
class Sin:
    def __init__(self,func):
        self.func=func
    def cal(self,x,dOrder):
        if dOrder==0:
            return np.sin(self.func.cal(x,0))
        else:
            new_sin=Cos(self.func)
            new_diff=Differentiate(self.func,1)
            new_mul=Multiply([new_sin,new_diff])
            return new_mul.cal(x,dOrder-1)

'''
#---------------Base-end Functions-------------------------------#
'''

            
class Constant:
        def __init__(self,const):
            self.const=const
        def cal(self,x,dOrder):
            if dOrder==0:
                return self.const
            else:
                return 0.          
class Scalar:
    def __init__(self,name='x'):
        self.name=name
    def cal(self,x,dOrder):
        if dOrder==0:
            return x
        elif dOrder==1:
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
    
    
    
    