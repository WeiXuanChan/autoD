# autoD
autoD is a lightweight, flexible automatic differentiation for python3 based on numpy. It enable user to convert your user-defined functions into differentiatable object. Thus, it will be able to do integration and matrix filling for you (see examples). To calculate the differential, call any the function ".cal(x,dOrder)" of any class in this module, where x is the 

###Function description:
#####Addition(funcList): objects in list can be float(for v3_1 and above)
input list of objects you want to add. funcList=[func1,func2,func3,...]

#####Multiply(funcList): objects in list can be float(for v3_1 and above)
input list of objects you want to multiply. funcList=[func1,func2,func3,...]

#####Power(func,pow):    
input an object and the power for power operation (pow can be a float or another func object).

#####Log(func,base):     
input an object and the base for logarithmic operation (base can be a float or another func object).

#####Exp(func):          
input object you want to do the operation e^.

#####Ln(func):           
input object you want to do the natural logarithmic operation.

#####Cos(func):          
input object you want to do the cosine operation.

#####Sin(func):          
input object you want to do the sin operation.

#####Constant(const):
change any float to a callable class object.

#####Scalar(name):
A scalar variable (each scalar must be independant of other variables)

#####Function(fx,*args,dependent='ALL'): 
input self-defined function to convert it to usable class object for differentiation.
Self-defined function must be able to accept the input in the form (x,dOrder,*args).
x is the value of the variable you want to differentiate wrt.
dOrder is the order of differentiation.
You can change your args even after definine by calling fx.changeArgs(*new_args).

###Versions

#####autoD_v1:
Use this version if you just need to differentiate one variable.

#####autoD_v2_1 and above:
These version uses python dictionary input for x and dOrder.
e.g x={'x':1.,'y':2.:'z':3.}, the dictionary key must be the same when defining Scalar

#####autoD_v3_1 and above:
These version optimized not neccessary calculation. When converting self-defined function into differentiatable class (autoD.Function(fx,*args,dependent='ALL')), optional input enable selecting dependent scalar. when dOrder contains a scalar name not detected in dependent, output automates to 0.

###note
I tried Theano (http://deeplearning.net/software/theano/) but I have no idea why it is clogging up my system RAM (~7GB, which is almost all I have). This code is easy to edit and depends on only Numpy. However you cannot use symbols such as '+', '-', '\*', '' or '**' as I have not figured out a way to decode such symbols while keeping the size of the code small (less prone to error). If you need more functionallity (I do not know what else there is to automatic differentiation as I have not explored Theano fully), know how to decode symbols easily, want to include more functions or any other issues, please leave your comments :) . I am using python3, so I do not know how well it works for python2. Thanks.
