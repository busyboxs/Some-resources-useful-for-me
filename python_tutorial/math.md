# [math](https://docs.python.org/2.7/library/math.html) — Mathematical functions

该模块始终可用。它提供对由C标准定义的数学函数的访问。

这些功能不能用于复数;如果您需要支持复数，请使用 [cmath](https://docs.python.org/2.7/library/cmath.html#module-cmath) 模块中的同名功能。支持复数的功能和不支持的功能之间的区别是由于大多数用户不想学习理解复数所需的太多数学。接收一个异常而不是一个复数的结果，可以更早地检测出用作参数的异常复数，这样程序员就可以确定首先产生的方式和原因。

以下功能由该模块提供。除非另外明确指出，否则所有返回值都是浮点数。

## Number-theoretic and representation functions（数论和表示函数）

|Function|Description|
|:----|:----|
|math.ceil(x)|Return the ceiling of x as a float, the smallest integer value greater than or equal to x.|
|math.copysign(x, y)|Return x with the sign of y. On a platform that supports signed zeros, `copysign(1.0, -0.0)` returns -1.0.|
|math.fabs(x)|Return the absolute value of x.|
|math.factorial(x)|Return x factorial. Raises ValueError if x is not integral or is negative.|
|math.floor(x)|Return the floor of x as a float, the largest integer value less than or equal to x.|
|math.fmod(x, y)|Return `fmod(x, y)`, as defined by the platform C library. Note that the Python expression `x % y` may not return the same result. The intent of the C standard is that `fmod(x, y)` be exactly (mathematically; to infinite precision) equal to `x - n*y` for some integer n such that the result has the same sign as x and magnitude less than `abs(y)`. Python’s `x % y` returns a result with the sign of y instead, and may not be exactly computable for float arguments. For example, `fmod(-1e-100, 1e100)` is `-1e-100`, but the result of Python’s `-1e-100 % 1e100` is `1e100-1e-100`, which cannot be represented exactly as a float, and rounds to the surprising `1e100`. For this reason, function `fmod()` is generally preferred when working with floats, while Python’s `x % y` is preferred when working with integers.|
|math.frexp(x)|Return the mantissa and exponent of x as the pair `(m, e)`. `m` is a float and `e` is an integer such that `x == m * 2**e` exactly. If `x` is zero, returns `(0.0, 0)`, otherwise `0.5 <= abs(m) < 1`. This is used to “pick apart” the internal representation of a float in a portable way.|
|math.fsum(iterable)|Return an accurate floating point sum of values in the iterable. |
|math.isinf(x)|Check if the float x is positive or negative infinity.|
|math.isnan(x)|Check if the float x is a NaN (not a number). |
|math.ldexp(x, i)|Return `x * (2**i)`. This is essentially the inverse of function frexp().|
|math.modf(x)|Return the fractional and integer parts of x. Both results carry the sign of x and are floats.|
|math.trunc(x)|Return the Real value x truncated to an Integral (usually a long integer). Uses the __trunc__ method.|

## Power and logarithmic functions

|Function|Description|
|:----|:----|
|math.exp(x)|Return $e^x$.|
|math.expm1(x)|Return $e^x-1$. For small floats x, the subtraction in exp(x) - 1 can result in a significant loss of precision; the expm1() function provides a way to compute this quantity to full precision:
|math.log(x[, base])|With one argument, return the natural logarithm of x (to base e). With two arguments, return the logarithm of x to the given base, calculated as log(x)/log(base).|
|math.log1p(x)|Return the natural logarithm of 1+x (base e). The result is calculated in a way which is accurate for x near zero.|
|math.log10(x)|Return the base-10 logarithm of x. This is usually more accurate than log(x, 10).|
|math.pow(x, y)|Return x raised to the power y. Exceptional cases follow Annex ‘F’ of the C99 standard as far as possible. In particular, `pow(1.0, x)` and `pow(x, 0.0)` always return 1.0, even when x is a zero or a NaN. If both x and y are finite, x is negative, and y is not an integer then `pow(x, y)` is undefined, and raises ValueError. Unlike the built-in `**` operator, `math.pow()` converts both its arguments to type float. Use `**` or the built-in `pow()` function for computing exact integer powers.|
|math.sqrt(x)|Return the square root of x.|

## Trigonometric functions

|Function|Description|
|:----|:----|
|math.acos(x)|Return the arc cosine of x, in radians.|
|math.asin(x)|Return the arc sine of x, in radians.|
|math.atan(x)|Return the arc tangent of x, in radians.|
|math.atan2(y, x)|Return `atan(y / x)`, in radians. The result is between `-pi` and `pi`. The vector in the plane from the origin to point `(x, y)` makes this angle with the positive X axis. The point of `atan2()` is that the signs of both inputs are known to it, so it can compute the correct quadrant for the angle. For example, `atan(1)` and `atan2(1, 1)` are both `pi/4`, but `atan2(-1, -1)` is `-3*pi/4`.|
|math.cos(x)|Return the cosine of x radians.|
|math.hypot(x, y)|Return the Euclidean norm, `sqrt(x*x + y*y)`. This is the length of the vector from the origin to `point (x, y)`.|
|math.sin(x)|Return the sine of x radians.|
|math.tan(x)|Return the tangent of x radians.|

## Angular conversion

|Function|Description|
|:----|:----|
|math.degrees(x)|Convert angle x from radians to degrees.|
|math.radians(x)|Convert angle x from degrees to radians.|

## Hyperbolic functions

|Function|Description|
|:----|:----|
|math.acosh(x)|Return the inverse hyperbolic cosine of x.|
|math.asinh(x)|Return the inverse hyperbolic sine of x.|
|math.atanh(x)|Return the inverse hyperbolic tangent of x.|
|math.cosh(x)|Return the hyperbolic cosine of x.|
|math.sinh(x)|Return the hyperbolic sine of x.|
|math.tanh(x)|Return the hyperbolic tangent of x.|

## Special functions

|Function|Description|
|:----|:----|
|math.erf(x)|Return the error function at x.|
|math.erfc(x)|Return the complementary error function at x.|
|math.gamma(x)|Return the Gamma function at x.|
|math.lgamma(x)|Return the natural logarithm of the absolute value of the Gamma function at x.|

## Constants

|Function|Description|
|:----|:----|
|math.pi|The mathematical constant π = 3.141592…, to available precision.|
|math.e|The mathematical constant e = 2.718281…, to available precision.|

