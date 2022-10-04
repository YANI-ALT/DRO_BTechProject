\ Model Support_Vector_Machine
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Lambda + 0.1 s_0 + 0.1 s_1 + 0.1 s_2 + 0.1 s_3 + 0.1 s_4 + 0.1 s_5
   + 0.1 s_6 + 0.1 s_7 + 0.1 s_8 + 0.1 s_9
Subject To
 data[0]_constr1: w0 + 4.8 w1 + 3 w2 - s_0 <= -1
 data[0]_constr2: - w0 - 4.8 w1 - 3 w2 - 0.1 Lambda - s_0 <= -1
 data[1]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_1 <= -1
 data[1]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_1 <= -1
 data[2]_constr1: w0 + 4.6 w1 + 3.2 w2 - s_2 <= -1
 data[2]_constr2: - w0 - 4.6 w1 - 3.2 w2 - 0.1 Lambda - s_2 <= -1
 data[3]_constr1: w0 + 5.3 w1 + 3.7 w2 - s_3 <= -1
 data[3]_constr2: - w0 - 5.3 w1 - 3.7 w2 - 0.1 Lambda - s_3 <= -1
 data[4]_constr1: w0 + 5 w1 + 3.3 w2 - s_4 <= -1
 data[4]_constr2: - w0 - 5 w1 - 3.3 w2 - 0.1 Lambda - s_4 <= -1
 data[5]_constr1: - w0 - 6.3 w1 - 3.3 w2 - s_5 <= -1
 data[5]_constr2: w0 + 6.3 w1 + 3.3 w2 - 0.1 Lambda - s_5 <= -1
 data[6]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_6 <= -1
 data[6]_constr2: w0 + 5.8 w1 + 2.7 w2 - 0.1 Lambda - s_6 <= -1
 data[7]_constr1: - w0 - 7.1 w1 - 3 w2 - s_7 <= -1
 data[7]_constr2: w0 + 7.1 w1 + 3 w2 - 0.1 Lambda - s_7 <= -1
 data[8]_constr1: - w0 - 6.3 w1 - 2.9 w2 - s_8 <= -1
 data[8]_constr2: w0 + 6.3 w1 + 2.9 w2 - 0.1 Lambda - s_8 <= -1
 data[9]_constr1: - w0 - 6.5 w1 - 3 w2 - s_9 <= -1
 data[9]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_9 <= -1
 s_0>=0: s_0 >= 0
 s_1>=0: s_1 >= 0
 s_2>=0: s_2 >= 0
 s_3>=0: s_3 >= 0
 s_4>=0: s_4 >= 0
 s_5>=0: s_5 >= 0
 s_6>=0: s_6 >= 0
 s_7>=0: s_7 >= 0
 s_8>=0: s_8 >= 0
 s_9>=0: s_9 >= 0
 qc0: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc1: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc2: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc3: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc4: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc5: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc6: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc7: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc8: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
 qc9: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
Bounds
End
