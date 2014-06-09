import theano.tensor as T
from theano import function
from theano import pp

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f= function( [x, y], z)

print f(2, 3)