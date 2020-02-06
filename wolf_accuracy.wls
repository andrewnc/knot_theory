#!/usr/bin/env wolframscript
<< KnotTheory`




data = ToExpression[$ScriptCommandLine[[2;;]]]
code1 = data[[1]]
code2 = data[[2]]

(*Print[Jones[DTCode[-6, -2, -4]][t] == Jones[DTCode[6, 2, 4]][t]]*)
(*Export["knot.png", Show[DrawPD[TorusKnot[4, 3]]], "PNG"];*)
(*Export["knot.png", Show[DrawPD[PD[DTCode[4, 6, 2]]]], "PNG"];*)

Print[Apply[DTCode, code1] == Apply[DTCode, code2]]

