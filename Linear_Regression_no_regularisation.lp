\ Model Linear_Regression_no_regularisation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.1428571428571428 abs_0 + 0.1428571428571428 abs_1
   + 0.1428571428571428 abs_2 + 0.1428571428571428 abs_3
   + 0.1428571428571428 abs_4 + 0.1428571428571428 abs_5
   + 0.1428571428571428 abs_6
Subject To
 R0: - w0 - 1.174345729437642 w1 + C3 = -1.433832755766481
 R1: - w0 - 0.3862178522203878 w1 + C5 = -1.103649871721759
 R2: - w0 - 1.599214658346796 w1 + C7 = -1.793267854403062
 R3: - w0 - 0.0298233386325421 w1 + C9 = -1.04864279353429
 R4: - w0 - 1.76745729886752 w1 + C11 = -2.080284361757528
 R5: - w0 - 0.4399202965792479 w1 + C13 = -1.192137746784001
 R6: - w0 - 1.566704152458803 w1 + C15 = -1.806135531247109
Bounds
General Constraints
 absconstr_0: abs_0 = ABS ( C3 )
 absconstr_1: abs_1 = ABS ( C5 )
 absconstr_2: abs_2 = ABS ( C7 )
 absconstr_3: abs_3 = ABS ( C9 )
 absconstr_4: abs_4 = ABS ( C11 )
 absconstr_5: abs_5 = ABS ( C13 )
 absconstr_6: abs_6 = ABS ( C15 )
End
