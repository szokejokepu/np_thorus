# np_thorus
Implementation of a Thorus (donut) in Numpy. 

The idea is that the arrays wrap around, for an array that is a=[1,2,3] 
which normally has values between a[0:3] == [1,2,3] you can take the values:
 * that wrap around in the negative values a[-2:2] = [2,3,1,2]
 * and can wrap around forward a[1:4] = [2,3,1]

It also works for 2D arrays, I haven't tried with more dimensions, 
because I didn't need that, but it should.
