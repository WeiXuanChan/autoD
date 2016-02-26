# autoD
autoD is a lightweight, flexible automatic differentiation for python3 based on numpy. 

###autoD_v1:
Use this version if you just need to differentiate one variable.

###Function description:
#####Addition(funcList):
input list of objects you want to add. funcList=[func1,func2,func3,...]

#####Multiply(funcList): 
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

#####Scalar(name='x'):   
you can just use it without input.
e.g x=autoD.Scalar().

#####Function(fx,*args): 
input self-defined function to convert it to usable class object for differentiation.
Self-defined function must be able to accept the input in the form (x,dOrder,*args).
x is the value of the variable you want to differentiate wrt.
dOrder is the order of differentiation.
You can change your args even after definine by calling fx.changeArgs(*new_args).

###autoD_v2:
This version is in beta. It accepts multivariable by giving input with numpy.1darray for both 'x' and 'dOrder'. Both 'x' and 'dOrder' must have the same length. When using scalar, you have to input the index of the array this scalar corresponds to.
e.g. x=numpy.array([x0,x1,x2]) x1=Scalar(index=1)

###note
I tried Theano (http://deeplearning.net/software/theano/) but I have no idea why it is clogging up my system RAM (~7GB, which I almost all I have). This code is easy to edit and depends on only Numpy. However you cannot use symbols such as '+', '-', '\*', '' or '**' as I have not figured out a way to decode such symbols while keeping the size of the code small (less prone to error). If you need more functionallity (I do not know what else there is to automatic differentiation as I have not explored Theano fully), know how to decode symbols easily, want to include more functions or any other issues, please leave your comments :) . I am using python3, so I do not know how well it works for python2. Thanks.
