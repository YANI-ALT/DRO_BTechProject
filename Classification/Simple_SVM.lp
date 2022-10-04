\ Model Simple_SVM
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ w0 ^2 + w1 ^2 ] / 2 
Subject To
 R0: 4.8 w0 + 3 w1 <= -1
 R1: 5.1 w0 + 3.8 w1 <= -1
 R2: 4.6 w0 + 3.2 w1 <= -1
 R3: 5.3 w0 + 3.7 w1 <= -1
 R4: 5 w0 + 3.3 w1 <= -1
 R5: - 6.3 w0 - 3.3 w1 <= -1
 R6: - 5.8 w0 - 2.7 w1 <= -1
 R7: - 7.1 w0 - 3 w1 <= -1
 R8: - 6.3 w0 - 2.9 w1 <= -1
 R9: - 6.5 w0 - 3 w1 <= -1
Bounds
 w0 free
 w1 free
End
