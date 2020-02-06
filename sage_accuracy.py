import sage.all
from sage.knots.knot import Knots
import sys

dt_code_1 = sys.argv[1]
dt_code_2 = sys.argv[2]

code1 = eval(eval(dt_code_1))
code2 = eval(eval(dt_code_2))

K = Knots()
w1 = K.from_dowker_code(code1)
w2 = K.from_dowker_code(code2)
print(w1 == w2)

