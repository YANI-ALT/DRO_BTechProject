\ Model Support_vector_reg
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1000 Lambda + 0.0333333333333333 s_0 + 0.0333333333333333 s_1
   + 0.0333333333333333 s_2 + 0.0333333333333333 s_3
   + 0.0333333333333333 s_4 + 0.0333333333333333 s_5
   + 0.0333333333333333 s_6 + 0.0333333333333333 s_7
   + 0.0333333333333333 s_8 + 0.0333333333333333 s_9
   + 0.0333333333333333 s_10 + 0.0333333333333333 s_11
   + 0.0333333333333333 s_12 + 0.0333333333333333 s_13
   + 0.0333333333333333 s_14 + 0.0333333333333333 s_15
   + 0.0333333333333333 s_16 + 0.0333333333333333 s_17
   + 0.0333333333333333 s_18 + 0.0333333333333333 s_19
   + 0.0333333333333333 s_20 + 0.0333333333333333 s_21
   + 0.0333333333333333 s_22 + 0.0333333333333333 s_23
   + 0.0333333333333333 s_24 + 0.0333333333333333 s_25
   + 0.0333333333333333 s_26 + 0.0333333333333333 s_27
   + 0.0333333333333333 s_28 + 0.0333333333333333 s_29
Subject To
 data[0]_constr1: - w0 - 0.392208262207379 w1 - s_0
   + 1.762389692014152 p_0_plus[0] + 2.370181429806772 p_0_plus[1]
   <= -0.9574017079209991
 data[0]_constr2: w0 + 0.392208262207379 w1 - s_0
   + 1.762389692014152 p_0_minus[0] + 2.370181429806772 p_0_minus[1]
   <= 1.957401707920999
 2Norm-constr_1_data[0]: - Lambda + C153 <= 0
 2Norm-constr_2_data[0]: - Lambda + C154 <= 0
 data[1]_constr1: - w0 - 1.130457172152803 w1 - s_1
   + 1.620246404274269 p_1_plus[0] + 1.489789232121466 p_1_plus[1]
   <= -1.099544995660882
 data[1]_constr2: w0 + 1.130457172152803 w1 - s_1
   + 1.620246404274269 p_1_minus[0] + 1.489789232121466 p_1_minus[1]
   <= 2.099544995660882
 2Norm-constr_1_data[1]: - Lambda + C157 <= 0
 2Norm-constr_2_data[1]: - Lambda + C158 <= 0
 data[2]_constr1: - w0 - 1.483268722476929 w1 - s_2
   + 1.593786366904444 p_2_plus[0] + 1.110517644427516 p_2_plus[1]
   <= -1.126005033030706
 data[2]_constr2: w0 + 1.483268722476929 w1 - s_2
   + 1.593786366904444 p_2_minus[0] + 1.110517644427516 p_2_minus[1]
   <= 2.126005033030706
 2Norm-constr_1_data[2]: - Lambda + C161 <= 0
 2Norm-constr_2_data[2]: - Lambda + C162 <= 0
 data[3]_constr1: - w0 - 0.0186516493584516 w1 - s_3
   + 2.09131751133116 p_3_plus[0] + 3.072665861972709 p_3_plus[1]
   <= -0.6284738886039904
 data[3]_constr2: w0 + 0.0186516493584516 w1 - s_3
   + 2.09131751133116 p_3_minus[0] + 3.072665861972709 p_3_minus[1]
   <= 1.62847388860399
 2Norm-constr_1_data[3]: - Lambda + C165 <= 0
 2Norm-constr_2_data[3]: - Lambda + C166 <= 0
 data[4]_constr1: - w0 - 0.4087249884125821 w1 - s_4
   + 2.074171055921745 p_4_plus[0] + 2.665446067509163 p_4_plus[1]
   <= -0.6456203440134058
 data[4]_constr2: w0 + 0.4087249884125821 w1 - s_4
   + 2.074171055921745 p_4_minus[0] + 2.665446067509163 p_4_minus[1]
   <= 1.645620344013406
 2Norm-constr_1_data[4]: - Lambda + C169 <= 0
 2Norm-constr_2_data[4]: - Lambda + C170 <= 0
 data[5]_constr1: - w0 - 1.760900238877326 w1 - s_5
   + 1.473322420140542 p_5_plus[0] + 0.7124221812632163 p_5_plus[1]
   <= -1.246468979794608
 data[5]_constr2: w0 + 1.760900238877326 w1 - s_5
   + 1.473322420140542 p_5_minus[0] + 0.7124221812632163 p_5_minus[1]
   <= 2.246468979794608
 2Norm-constr_1_data[5]: - Lambda + C173 <= 0
 2Norm-constr_2_data[5]: - Lambda + C174 <= 0
 data[6]_constr1: - w0 - 0.2326528481323859 w1 - s_6
   + 2.128486432664913 p_6_plus[0] + 2.895833584532527 p_6_plus[1]
   <= -0.5913049672702375
 data[6]_constr2: w0 + 0.2326528481323859 w1 - s_6
   + 2.128486432664913 p_6_minus[0] + 2.895833584532527 p_6_minus[1]
   <= 1.591304967270238
 2Norm-constr_1_data[6]: - Lambda + C177 <= 0
 2Norm-constr_2_data[6]: - Lambda + C178 <= 0
 data[7]_constr1: - w0 - 1.168908804338115 w1 - s_7
   + 1.550443576707631 p_7_plus[0] + 1.381534772369517 p_7_plus[1]
   <= -1.169347823227519
 data[7]_constr2: w0 + 1.168908804338115 w1 - s_7
   + 1.550443576707631 p_7_minus[0] + 1.381534772369517 p_7_minus[1]
   <= 2.169347823227519
 2Norm-constr_1_data[7]: - Lambda + C181 <= 0
 2Norm-constr_2_data[7]: - Lambda + C182 <= 0
 data[8]_constr1: - w0 - 1.626902483831781 w1 - s_8
   + 1.500472963539456 p_8_plus[0] + 0.8735704797076751 p_8_plus[1]
   <= -1.219318436395694
 data[8]_constr2: w0 + 1.626902483831781 w1 - s_8
   + 1.500472963539456 p_8_minus[0] + 0.8735704797076751 p_8_minus[1]
   <= 2.219318436395694
 2Norm-constr_1_data[8]: - Lambda + C185 <= 0
 2Norm-constr_2_data[8]: - Lambda + C186 <= 0
 data[9]_constr1: - w0 - 1.155205366220341 w1 - s_9
   + 2.043274220200727 p_9_plus[0] + 1.888068853980386 p_9_plus[1]
   <= -0.6765171797344238
 data[9]_constr2: w0 + 1.155205366220341 w1 - s_9
   + 2.043274220200727 p_9_minus[0] + 1.888068853980386 p_9_minus[1]
   <= 1.676517179734424
 2Norm-constr_1_data[9]: - Lambda + C189 <= 0
 2Norm-constr_2_data[9]: - Lambda + C190 <= 0
 data[10]_constr1: - w0 - 1.2602519782007 w1 - s_10
   + 1.66150448399638 p_10_plus[0] + 1.40125250579568 p_10_plus[1]
   <= -1.05828691593877
 data[10]_constr2: w0 + 1.2602519782007 w1 - s_10
   + 1.66150448399638 p_10_minus[0] + 1.40125250579568 p_10_minus[1]
   <= 2.05828691593877
 2Norm-constr_1_data[10]: - Lambda + C193 <= 0
 2Norm-constr_2_data[10]: - Lambda + C194 <= 0
 data[11]_constr1: - w0 - 0.3979353714140454 w1 - s_11
   + 2.083768369493249 p_11_plus[0] + 2.685832998079203 p_11_plus[1]
   <= -0.6360230304419019
 data[11]_constr2: w0 + 0.3979353714140454 w1 - s_11
   + 2.083768369493249 p_11_minus[0] + 2.685832998079203 p_11_minus[1]
   <= 1.636023030441902
 2Norm-constr_1_data[11]: - Lambda + C197 <= 0
 2Norm-constr_2_data[11]: - Lambda + C198 <= 0
 data[12]_constr1: - w0 - 0.8237874768426068 w1 - s_12
   + 1.932385944549373 p_12_plus[0] + 2.108598467706766 p_12_plus[1]
   <= -0.7874054553857774
 data[12]_constr2: w0 + 0.8237874768426068 w1 - s_12
   + 1.932385944549373 p_12_minus[0] + 2.108598467706766 p_12_minus[1]
   <= 1.787405455385777
 2Norm-constr_1_data[12]: - Lambda + C201 <= 0
 2Norm-constr_2_data[12]: - Lambda + C202 <= 0
 data[13]_constr1: - w0 - 0.6378283189121157 w1 - s_13
   + 1.756650367415121 p_13_plus[0] + 2.118822048503005 p_13_plus[1]
   <= -0.9631410325200298
 data[13]_constr2: w0 + 0.6378283189121157 w1 - s_13
   + 1.756650367415121 p_13_minus[0] + 2.118822048503005 p_13_minus[1]
   <= 1.96314103252003
 2Norm-constr_1_data[13]: - Lambda + C205 <= 0
 2Norm-constr_2_data[13]: - Lambda + C206 <= 0
 data[14]_constr1: - w0 - 0.6173306923529991 w1 - s_14
   + 2.034841623615349 p_14_plus[0] + 2.41751093126235 p_14_plus[1]
   <= -0.6849497763198014
 data[14]_constr2: w0 + 0.6173306923529991 w1 - s_14
   + 2.034841623615349 p_14_minus[0] + 2.41751093126235 p_14_minus[1]
   <= 1.684949776319801
 2Norm-constr_1_data[14]: - Lambda + C209 <= 0
 2Norm-constr_2_data[14]: - Lambda + C210 <= 0
 data[15]_constr1: - w0 - 1.873023612583461 w1 - s_15
   + 1.056414960586804 p_15_plus[0] + 0.1833913480033433 p_15_plus[1]
   <= -1.663376439348347
 data[15]_constr2: w0 + 1.873023612583461 w1 - s_15
   + 1.056414960586804 p_15_minus[0] + 0.1833913480033433 p_15_minus[1]
   <= 2.663376439348347
 2Norm-constr_1_data[15]: - Lambda + C213 <= 0
 2Norm-constr_2_data[15]: - Lambda + C214 <= 0
 data[16]_constr1: - w0 - 1.89155369509694 w1 - s_16
   + 1.392038868016668 p_16_plus[0] + 0.5004851729197277 p_16_plus[1]
   <= -1.327752531918482
 data[16]_constr2: w0 + 1.89155369509694 w1 - s_16
   + 1.392038868016668 p_16_minus[0] + 0.5004851729197277 p_16_minus[1]
   <= 2.327752531918482
 2Norm-constr_1_data[16]: - Lambda + C217 <= 0
 2Norm-constr_2_data[16]: - Lambda + C218 <= 0
 data[17]_constr1: - w0 - 1.583933760292234 w1 - s_17
   + 1.435716974891806 p_17_plus[0] + 0.8517832145995716 p_17_plus[1]
   <= -1.284074425043345
 data[17]_constr2: w0 + 1.583933760292234 w1 - s_17
   + 1.435716974891806 p_17_minus[0] + 0.8517832145995716 p_17_minus[1]
   <= 2.284074425043345
 2Norm-constr_1_data[17]: - Lambda + C221 <= 0
 2Norm-constr_2_data[17]: - Lambda + C222 <= 0
 data[18]_constr1: - w0 - 0.1003604264083728 w1 - s_18
   + 2.346352466951098 p_18_plus[0] + 3.245992040542725 p_18_plus[1]
   <= -0.3734389329840524
 data[18]_constr2: w0 + 0.1003604264083728 w1 - s_18
   + 2.346352466951098 p_18_minus[0] + 3.245992040542725 p_18_minus[1]
   <= 1.373438932984052
 2Norm-constr_1_data[18]: - Lambda + C225 <= 0
 2Norm-constr_2_data[18]: - Lambda + C226 <= 0
 data[19]_constr1: - w0 - 0.4572352930251671 w1 - s_19
   + 1.823461705675967 p_19_plus[0] + 2.366226412650799 p_19_plus[1]
   <= -0.8963296942591841
 data[19]_constr2: w0 + 0.4572352930251671 w1 - s_19
   + 1.823461705675967 p_19_minus[0] + 2.366226412650799 p_19_minus[1]
   <= 1.896329694259184
 2Norm-constr_1_data[19]: - Lambda + C229 <= 0
 2Norm-constr_2_data[19]: - Lambda + C230 <= 0
 data[20]_constr1: - w0 - 0.9733035704821686 w1 - s_20
   + 1.714786755968855 p_20_plus[0] + 1.741483185486686 p_20_plus[1]
   <= -1.005004643966296
 data[20]_constr2: w0 + 0.9733035704821686 w1 - s_20
   + 1.714786755968855 p_20_minus[0] + 1.741483185486686 p_20_minus[1]
   <= 2.005004643966296
 2Norm-constr_1_data[20]: - Lambda + C233 <= 0
 2Norm-constr_2_data[20]: - Lambda + C234 <= 0
 data[21]_constr1: - w0 - 1.982995835254107 w1 - s_21
   + 1.44618083745438 p_21_plus[0] + 0.4631850022002728 p_21_plus[1]
   <= -1.273610562480771
 data[21]_constr2: w0 + 1.982995835254107 w1 - s_21
   + 1.44618083745438 p_21_minus[0] + 0.4631850022002728 p_21_minus[1]
   <= 2.273610562480771
 2Norm-constr_1_data[21]: - Lambda + C237 <= 0
 2Norm-constr_2_data[21]: - Lambda + C238 <= 0
 data[22]_constr1: - w0 - 0.8322315831179621 w1 - s_22
   + 1.780845755961181 p_22_plus[0] + 1.948614172843219 p_22_plus[1]
   <= -0.9389456439739692
 data[22]_constr2: w0 + 0.8322315831179621 w1 - s_22
   + 1.780845755961181 p_22_minus[0] + 1.948614172843219 p_22_minus[1]
   <= 1.938945643973969
 2Norm-constr_1_data[22]: - Lambda + C241 <= 0
 2Norm-constr_2_data[22]: - Lambda + C242 <= 0
 data[23]_constr1: - w0 - 0.6137941315825155 w1 - s_23
   + 1.713143380493547 p_23_plus[0] + 2.099349248911032 p_23_plus[1]
   <= -1.006648019441603
 data[23]_constr2: w0 + 0.6137941315825155 w1 - s_23
   + 1.713143380493547 p_23_minus[0] + 2.099349248911032 p_23_minus[1]
   <= 2.006648019441603
 2Norm-constr_1_data[23]: - Lambda + C245 <= 0
 2Norm-constr_2_data[23]: - Lambda + C246 <= 0
 data[24]_constr1: - w0 - 1.941858014334926 w1 - s_24
   + 0.9829958352541066 p_24_plus[0] + 0.0411378209191811 p_24_plus[1]
   <= -1.736795564681044
 data[24]_constr2: w0 + 1.941858014334926 w1 - s_24
   + 0.9829958352541066 p_24_minus[0] + 0.0411378209191811 p_24_minus[1]
   <= 2.736795564681044
 2Norm-constr_1_data[24]: - Lambda + C249 <= 0
 2Norm-constr_2_data[24]: - Lambda + C250 <= 0
 data[25]_constr1: - w0 - 1.042685929410966 w1 - s_25
   + 1.658693962195811 p_25_plus[0] + 1.616008032784845 p_25_plus[1]
   <= -1.06109743773934
 data[25]_constr2: w0 + 1.042685929410966 w1 - s_25
   + 1.658693962195811 p_25_minus[0] + 1.616008032784845 p_25_minus[1]
   <= 2.06109743773934
 2Norm-constr_1_data[25]: - Lambda + C253 <= 0
 2Norm-constr_2_data[25]: - Lambda + C254 <= 0
 data[26]_constr1: - w0 - 0.7645502433165927 w1 - s_26
   + 1.54797318434041 p_26_plus[0] + 1.783422941023818 p_26_plus[1]
   <= -1.17181821559474
 data[26]_constr2: w0 + 0.7645502433165927 w1 - s_26
   + 1.54797318434041 p_26_minus[0] + 1.783422941023818 p_26_minus[1]
   <= 2.17181821559474
 2Norm-constr_1_data[26]: - Lambda + C257 <= 0
 2Norm-constr_2_data[26]: - Lambda + C258 <= 0
 data[27]_constr1: - w0 - 0.9671776914661928 w1 - s_27
   + 1.283997179822367 p_27_plus[0] + 1.316819488356174 p_27_plus[1]
   <= -1.435794220112784
 data[27]_constr2: w0 + 0.9671776914661928 w1 - s_27
   + 1.283997179822367 p_27_minus[0] + 1.316819488356174 p_27_minus[1]
   <= 2.435794220112784
 2Norm-constr_1_data[27]: - Lambda + C261 <= 0
 2Norm-constr_2_data[27]: - Lambda + C262 <= 0
 data[28]_constr1: - w0 - 0.3772864304130567 w1 - s_28
   + 1.530825141777142 p_28_plus[0] + 2.153538711364086 p_28_plus[1]
   <= -1.188966258158008
 data[28]_constr2: w0 + 0.3772864304130567 w1 - s_28
   + 1.530825141777142 p_28_minus[0] + 2.153538711364086 p_28_minus[1]
   <= 2.188966258158008
 2Norm-constr_1_data[28]: - Lambda + C265 <= 0
 2Norm-constr_2_data[28]: - Lambda + C266 <= 0
 data[29]_constr1: - w0 - 1.257727506708613 w1 - s_29
   + 1.81028173767287 p_29_plus[0] + 1.552554230964257 p_29_plus[1]
   <= -0.9095096622622809
 data[29]_constr2: w0 + 1.257727506708613 w1 - s_29
   + 1.81028173767287 p_29_minus[0] + 1.552554230964257 p_29_minus[1]
   <= 1.909509662262281
 2Norm-constr_1_data[29]: - Lambda + C269 <= 0
 2Norm-constr_2_data[29]: - Lambda + C270 <= 0
 s_0>=0: s_0 >= 0
 p_[00]_plus>=0: p_0_plus[0] >= 0
 p_[00]_minus>=0: p_0_minus[0] >= 0
 p_[01]_plus>=0: p_0_plus[1] >= 0
 p_[01]_minus>=0: p_0_minus[1] >= 0
 s_1>=0: s_1 >= 0
 p_[10]_plus>=0: p_1_plus[0] >= 0
 p_[10]_minus>=0: p_1_minus[0] >= 0
 p_[11]_plus>=0: p_1_plus[1] >= 0
 p_[11]_minus>=0: p_1_minus[1] >= 0
 s_2>=0: s_2 >= 0
 p_[20]_plus>=0: p_2_plus[0] >= 0
 p_[20]_minus>=0: p_2_minus[0] >= 0
 p_[21]_plus>=0: p_2_plus[1] >= 0
 p_[21]_minus>=0: p_2_minus[1] >= 0
 s_3>=0: s_3 >= 0
 p_[30]_plus>=0: p_3_plus[0] >= 0
 p_[30]_minus>=0: p_3_minus[0] >= 0
 p_[31]_plus>=0: p_3_plus[1] >= 0
 p_[31]_minus>=0: p_3_minus[1] >= 0
 s_4>=0: s_4 >= 0
 p_[40]_plus>=0: p_4_plus[0] >= 0
 p_[40]_minus>=0: p_4_minus[0] >= 0
 p_[41]_plus>=0: p_4_plus[1] >= 0
 p_[41]_minus>=0: p_4_minus[1] >= 0
 s_5>=0: s_5 >= 0
 p_[50]_plus>=0: p_5_plus[0] >= 0
 p_[50]_minus>=0: p_5_minus[0] >= 0
 p_[51]_plus>=0: p_5_plus[1] >= 0
 p_[51]_minus>=0: p_5_minus[1] >= 0
 s_6>=0: s_6 >= 0
 p_[60]_plus>=0: p_6_plus[0] >= 0
 p_[60]_minus>=0: p_6_minus[0] >= 0
 p_[61]_plus>=0: p_6_plus[1] >= 0
 p_[61]_minus>=0: p_6_minus[1] >= 0
 s_7>=0: s_7 >= 0
 p_[70]_plus>=0: p_7_plus[0] >= 0
 p_[70]_minus>=0: p_7_minus[0] >= 0
 p_[71]_plus>=0: p_7_plus[1] >= 0
 p_[71]_minus>=0: p_7_minus[1] >= 0
 s_8>=0: s_8 >= 0
 p_[80]_plus>=0: p_8_plus[0] >= 0
 p_[80]_minus>=0: p_8_minus[0] >= 0
 p_[81]_plus>=0: p_8_plus[1] >= 0
 p_[81]_minus>=0: p_8_minus[1] >= 0
 s_9>=0: s_9 >= 0
 p_[90]_plus>=0: p_9_plus[0] >= 0
 p_[90]_minus>=0: p_9_minus[0] >= 0
 p_[91]_plus>=0: p_9_plus[1] >= 0
 p_[91]_minus>=0: p_9_minus[1] >= 0
 s_10>=0: s_10 >= 0
 p_[100]_plus>=0: p_10_plus[0] >= 0
 p_[100]_minus>=0: p_10_minus[0] >= 0
 p_[101]_plus>=0: p_10_plus[1] >= 0
 p_[101]_minus>=0: p_10_minus[1] >= 0
 s_11>=0: s_11 >= 0
 p_[110]_plus>=0: p_11_plus[0] >= 0
 p_[110]_minus>=0: p_11_minus[0] >= 0
 p_[111]_plus>=0: p_11_plus[1] >= 0
 p_[111]_minus>=0: p_11_minus[1] >= 0
 s_12>=0: s_12 >= 0
 p_[120]_plus>=0: p_12_plus[0] >= 0
 p_[120]_minus>=0: p_12_minus[0] >= 0
 p_[121]_plus>=0: p_12_plus[1] >= 0
 p_[121]_minus>=0: p_12_minus[1] >= 0
 s_13>=0: s_13 >= 0
 p_[130]_plus>=0: p_13_plus[0] >= 0
 p_[130]_minus>=0: p_13_minus[0] >= 0
 p_[131]_plus>=0: p_13_plus[1] >= 0
 p_[131]_minus>=0: p_13_minus[1] >= 0
 s_14>=0: s_14 >= 0
 p_[140]_plus>=0: p_14_plus[0] >= 0
 p_[140]_minus>=0: p_14_minus[0] >= 0
 p_[141]_plus>=0: p_14_plus[1] >= 0
 p_[141]_minus>=0: p_14_minus[1] >= 0
 s_15>=0: s_15 >= 0
 p_[150]_plus>=0: p_15_plus[0] >= 0
 p_[150]_minus>=0: p_15_minus[0] >= 0
 p_[151]_plus>=0: p_15_plus[1] >= 0
 p_[151]_minus>=0: p_15_minus[1] >= 0
 s_16>=0: s_16 >= 0
 p_[160]_plus>=0: p_16_plus[0] >= 0
 p_[160]_minus>=0: p_16_minus[0] >= 0
 p_[161]_plus>=0: p_16_plus[1] >= 0
 p_[161]_minus>=0: p_16_minus[1] >= 0
 s_17>=0: s_17 >= 0
 p_[170]_plus>=0: p_17_plus[0] >= 0
 p_[170]_minus>=0: p_17_minus[0] >= 0
 p_[171]_plus>=0: p_17_plus[1] >= 0
 p_[171]_minus>=0: p_17_minus[1] >= 0
 s_18>=0: s_18 >= 0
 p_[180]_plus>=0: p_18_plus[0] >= 0
 p_[180]_minus>=0: p_18_minus[0] >= 0
 p_[181]_plus>=0: p_18_plus[1] >= 0
 p_[181]_minus>=0: p_18_minus[1] >= 0
 s_19>=0: s_19 >= 0
 p_[190]_plus>=0: p_19_plus[0] >= 0
 p_[190]_minus>=0: p_19_minus[0] >= 0
 p_[191]_plus>=0: p_19_plus[1] >= 0
 p_[191]_minus>=0: p_19_minus[1] >= 0
 s_20>=0: s_20 >= 0
 p_[200]_plus>=0: p_20_plus[0] >= 0
 p_[200]_minus>=0: p_20_minus[0] >= 0
 p_[201]_plus>=0: p_20_plus[1] >= 0
 p_[201]_minus>=0: p_20_minus[1] >= 0
 s_21>=0: s_21 >= 0
 p_[210]_plus>=0: p_21_plus[0] >= 0
 p_[210]_minus>=0: p_21_minus[0] >= 0
 p_[211]_plus>=0: p_21_plus[1] >= 0
 p_[211]_minus>=0: p_21_minus[1] >= 0
 s_22>=0: s_22 >= 0
 p_[220]_plus>=0: p_22_plus[0] >= 0
 p_[220]_minus>=0: p_22_minus[0] >= 0
 p_[221]_plus>=0: p_22_plus[1] >= 0
 p_[221]_minus>=0: p_22_minus[1] >= 0
 s_23>=0: s_23 >= 0
 p_[230]_plus>=0: p_23_plus[0] >= 0
 p_[230]_minus>=0: p_23_minus[0] >= 0
 p_[231]_plus>=0: p_23_plus[1] >= 0
 p_[231]_minus>=0: p_23_minus[1] >= 0
 s_24>=0: s_24 >= 0
 p_[240]_plus>=0: p_24_plus[0] >= 0
 p_[240]_minus>=0: p_24_minus[0] >= 0
 p_[241]_plus>=0: p_24_plus[1] >= 0
 p_[241]_minus>=0: p_24_minus[1] >= 0
 s_25>=0: s_25 >= 0
 p_[250]_plus>=0: p_25_plus[0] >= 0
 p_[250]_minus>=0: p_25_minus[0] >= 0
 p_[251]_plus>=0: p_25_plus[1] >= 0
 p_[251]_minus>=0: p_25_minus[1] >= 0
 s_26>=0: s_26 >= 0
 p_[260]_plus>=0: p_26_plus[0] >= 0
 p_[260]_minus>=0: p_26_minus[0] >= 0
 p_[261]_plus>=0: p_26_plus[1] >= 0
 p_[261]_minus>=0: p_26_minus[1] >= 0
 s_27>=0: s_27 >= 0
 p_[270]_plus>=0: p_27_plus[0] >= 0
 p_[270]_minus>=0: p_27_minus[0] >= 0
 p_[271]_plus>=0: p_27_plus[1] >= 0
 p_[271]_minus>=0: p_27_minus[1] >= 0
 s_28>=0: s_28 >= 0
 p_[280]_plus>=0: p_28_plus[0] >= 0
 p_[280]_minus>=0: p_28_minus[0] >= 0
 p_[281]_plus>=0: p_28_plus[1] >= 0
 p_[281]_minus>=0: p_28_minus[1] >= 0
 s_29>=0: s_29 >= 0
 p_[290]_plus>=0: p_29_plus[0] >= 0
 p_[290]_minus>=0: p_29_minus[0] >= 0
 p_[291]_plus>=0: p_29_plus[1] >= 0
 p_[291]_minus>=0: p_29_minus[1] >= 0
 qc0: 2 p_0_plus[0] + 2 p_0_plus[1] + C155 + [ - w0 ^2 - 2 w0 * p_0_plus[0]
   - w1 ^2 - 2 w1 * p_0_plus[1] - 2 p_0_plus[0] ^2
   - 2 p_0_plus[0] * p_0_plus[1] - 2 p_0_plus[1] ^2 ] = 1
 qc1: - 2 p_0_minus[0] - 2 p_0_minus[1] + C156 + [ - w0 ^2
   + 2 w0 * p_0_minus[0] - w1 ^2 + 2 w1 * p_0_minus[1] - 2 p_0_minus[0] ^2
   - 2 p_0_minus[0] * p_0_minus[1] - 2 p_0_minus[1] ^2 ] = 1
 qc2: 2 p_1_plus[0] + 2 p_1_plus[1] + C159 + [ - w0 ^2 - 2 w0 * p_1_plus[0]
   - w1 ^2 - 2 w1 * p_1_plus[1] - 2 p_1_plus[0] ^2
   - 2 p_1_plus[0] * p_1_plus[1] - 2 p_1_plus[1] ^2 ] = 1
 qc3: - 2 p_1_minus[0] - 2 p_1_minus[1] + C160 + [ - w0 ^2
   + 2 w0 * p_1_minus[0] - w1 ^2 + 2 w1 * p_1_minus[1] - 2 p_1_minus[0] ^2
   - 2 p_1_minus[0] * p_1_minus[1] - 2 p_1_minus[1] ^2 ] = 1
 qc4: 2 p_2_plus[0] + 2 p_2_plus[1] + C163 + [ - w0 ^2 - 2 w0 * p_2_plus[0]
   - w1 ^2 - 2 w1 * p_2_plus[1] - 2 p_2_plus[0] ^2
   - 2 p_2_plus[0] * p_2_plus[1] - 2 p_2_plus[1] ^2 ] = 1
 qc5: - 2 p_2_minus[0] - 2 p_2_minus[1] + C164 + [ - w0 ^2
   + 2 w0 * p_2_minus[0] - w1 ^2 + 2 w1 * p_2_minus[1] - 2 p_2_minus[0] ^2
   - 2 p_2_minus[0] * p_2_minus[1] - 2 p_2_minus[1] ^2 ] = 1
 qc6: 2 p_3_plus[0] + 2 p_3_plus[1] + C167 + [ - w0 ^2 - 2 w0 * p_3_plus[0]
   - w1 ^2 - 2 w1 * p_3_plus[1] - 2 p_3_plus[0] ^2
   - 2 p_3_plus[0] * p_3_plus[1] - 2 p_3_plus[1] ^2 ] = 1
 qc7: - 2 p_3_minus[0] - 2 p_3_minus[1] + C168 + [ - w0 ^2
   + 2 w0 * p_3_minus[0] - w1 ^2 + 2 w1 * p_3_minus[1] - 2 p_3_minus[0] ^2
   - 2 p_3_minus[0] * p_3_minus[1] - 2 p_3_minus[1] ^2 ] = 1
 qc8: 2 p_4_plus[0] + 2 p_4_plus[1] + C171 + [ - w0 ^2 - 2 w0 * p_4_plus[0]
   - w1 ^2 - 2 w1 * p_4_plus[1] - 2 p_4_plus[0] ^2
   - 2 p_4_plus[0] * p_4_plus[1] - 2 p_4_plus[1] ^2 ] = 1
 qc9: - 2 p_4_minus[0] - 2 p_4_minus[1] + C172 + [ - w0 ^2
   + 2 w0 * p_4_minus[0] - w1 ^2 + 2 w1 * p_4_minus[1] - 2 p_4_minus[0] ^2
   - 2 p_4_minus[0] * p_4_minus[1] - 2 p_4_minus[1] ^2 ] = 1
 qc10: 2 p_5_plus[0] + 2 p_5_plus[1] + C175 + [ - w0 ^2
   - 2 w0 * p_5_plus[0] - w1 ^2 - 2 w1 * p_5_plus[1] - 2 p_5_plus[0] ^2
   - 2 p_5_plus[0] * p_5_plus[1] - 2 p_5_plus[1] ^2 ] = 1
 qc11: - 2 p_5_minus[0] - 2 p_5_minus[1] + C176 + [ - w0 ^2
   + 2 w0 * p_5_minus[0] - w1 ^2 + 2 w1 * p_5_minus[1] - 2 p_5_minus[0] ^2
   - 2 p_5_minus[0] * p_5_minus[1] - 2 p_5_minus[1] ^2 ] = 1
 qc12: 2 p_6_plus[0] + 2 p_6_plus[1] + C179 + [ - w0 ^2
   - 2 w0 * p_6_plus[0] - w1 ^2 - 2 w1 * p_6_plus[1] - 2 p_6_plus[0] ^2
   - 2 p_6_plus[0] * p_6_plus[1] - 2 p_6_plus[1] ^2 ] = 1
 qc13: - 2 p_6_minus[0] - 2 p_6_minus[1] + C180 + [ - w0 ^2
   + 2 w0 * p_6_minus[0] - w1 ^2 + 2 w1 * p_6_minus[1] - 2 p_6_minus[0] ^2
   - 2 p_6_minus[0] * p_6_minus[1] - 2 p_6_minus[1] ^2 ] = 1
 qc14: 2 p_7_plus[0] + 2 p_7_plus[1] + C183 + [ - w0 ^2
   - 2 w0 * p_7_plus[0] - w1 ^2 - 2 w1 * p_7_plus[1] - 2 p_7_plus[0] ^2
   - 2 p_7_plus[0] * p_7_plus[1] - 2 p_7_plus[1] ^2 ] = 1
 qc15: - 2 p_7_minus[0] - 2 p_7_minus[1] + C184 + [ - w0 ^2
   + 2 w0 * p_7_minus[0] - w1 ^2 + 2 w1 * p_7_minus[1] - 2 p_7_minus[0] ^2
   - 2 p_7_minus[0] * p_7_minus[1] - 2 p_7_minus[1] ^2 ] = 1
 qc16: 2 p_8_plus[0] + 2 p_8_plus[1] + C187 + [ - w0 ^2
   - 2 w0 * p_8_plus[0] - w1 ^2 - 2 w1 * p_8_plus[1] - 2 p_8_plus[0] ^2
   - 2 p_8_plus[0] * p_8_plus[1] - 2 p_8_plus[1] ^2 ] = 1
 qc17: - 2 p_8_minus[0] - 2 p_8_minus[1] + C188 + [ - w0 ^2
   + 2 w0 * p_8_minus[0] - w1 ^2 + 2 w1 * p_8_minus[1] - 2 p_8_minus[0] ^2
   - 2 p_8_minus[0] * p_8_minus[1] - 2 p_8_minus[1] ^2 ] = 1
 qc18: 2 p_9_plus[0] + 2 p_9_plus[1] + C191 + [ - w0 ^2
   - 2 w0 * p_9_plus[0] - w1 ^2 - 2 w1 * p_9_plus[1] - 2 p_9_plus[0] ^2
   - 2 p_9_plus[0] * p_9_plus[1] - 2 p_9_plus[1] ^2 ] = 1
 qc19: - 2 p_9_minus[0] - 2 p_9_minus[1] + C192 + [ - w0 ^2
   + 2 w0 * p_9_minus[0] - w1 ^2 + 2 w1 * p_9_minus[1] - 2 p_9_minus[0] ^2
   - 2 p_9_minus[0] * p_9_minus[1] - 2 p_9_minus[1] ^2 ] = 1
 qc20: 2 p_10_plus[0] + 2 p_10_plus[1] + C195 + [ - w0 ^2
   - 2 w0 * p_10_plus[0] - w1 ^2 - 2 w1 * p_10_plus[1] - 2 p_10_plus[0] ^2
   - 2 p_10_plus[0] * p_10_plus[1] - 2 p_10_plus[1] ^2 ] = 1
 qc21: - 2 p_10_minus[0] - 2 p_10_minus[1] + C196 + [ - w0 ^2
   + 2 w0 * p_10_minus[0] - w1 ^2 + 2 w1 * p_10_minus[1]
   - 2 p_10_minus[0] ^2 - 2 p_10_minus[0] * p_10_minus[1]
   - 2 p_10_minus[1] ^2 ] = 1
 qc22: 2 p_11_plus[0] + 2 p_11_plus[1] + C199 + [ - w0 ^2
   - 2 w0 * p_11_plus[0] - w1 ^2 - 2 w1 * p_11_plus[1] - 2 p_11_plus[0] ^2
   - 2 p_11_plus[0] * p_11_plus[1] - 2 p_11_plus[1] ^2 ] = 1
 qc23: - 2 p_11_minus[0] - 2 p_11_minus[1] + C200 + [ - w0 ^2
   + 2 w0 * p_11_minus[0] - w1 ^2 + 2 w1 * p_11_minus[1]
   - 2 p_11_minus[0] ^2 - 2 p_11_minus[0] * p_11_minus[1]
   - 2 p_11_minus[1] ^2 ] = 1
 qc24: 2 p_12_plus[0] + 2 p_12_plus[1] + C203 + [ - w0 ^2
   - 2 w0 * p_12_plus[0] - w1 ^2 - 2 w1 * p_12_plus[1] - 2 p_12_plus[0] ^2
   - 2 p_12_plus[0] * p_12_plus[1] - 2 p_12_plus[1] ^2 ] = 1
 qc25: - 2 p_12_minus[0] - 2 p_12_minus[1] + C204 + [ - w0 ^2
   + 2 w0 * p_12_minus[0] - w1 ^2 + 2 w1 * p_12_minus[1]
   - 2 p_12_minus[0] ^2 - 2 p_12_minus[0] * p_12_minus[1]
   - 2 p_12_minus[1] ^2 ] = 1
 qc26: 2 p_13_plus[0] + 2 p_13_plus[1] + C207 + [ - w0 ^2
   - 2 w0 * p_13_plus[0] - w1 ^2 - 2 w1 * p_13_plus[1] - 2 p_13_plus[0] ^2
   - 2 p_13_plus[0] * p_13_plus[1] - 2 p_13_plus[1] ^2 ] = 1
 qc27: - 2 p_13_minus[0] - 2 p_13_minus[1] + C208 + [ - w0 ^2
   + 2 w0 * p_13_minus[0] - w1 ^2 + 2 w1 * p_13_minus[1]
   - 2 p_13_minus[0] ^2 - 2 p_13_minus[0] * p_13_minus[1]
   - 2 p_13_minus[1] ^2 ] = 1
 qc28: 2 p_14_plus[0] + 2 p_14_plus[1] + C211 + [ - w0 ^2
   - 2 w0 * p_14_plus[0] - w1 ^2 - 2 w1 * p_14_plus[1] - 2 p_14_plus[0] ^2
   - 2 p_14_plus[0] * p_14_plus[1] - 2 p_14_plus[1] ^2 ] = 1
 qc29: - 2 p_14_minus[0] - 2 p_14_minus[1] + C212 + [ - w0 ^2
   + 2 w0 * p_14_minus[0] - w1 ^2 + 2 w1 * p_14_minus[1]
   - 2 p_14_minus[0] ^2 - 2 p_14_minus[0] * p_14_minus[1]
   - 2 p_14_minus[1] ^2 ] = 1
 qc30: 2 p_15_plus[0] + 2 p_15_plus[1] + C215 + [ - w0 ^2
   - 2 w0 * p_15_plus[0] - w1 ^2 - 2 w1 * p_15_plus[1] - 2 p_15_plus[0] ^2
   - 2 p_15_plus[0] * p_15_plus[1] - 2 p_15_plus[1] ^2 ] = 1
 qc31: - 2 p_15_minus[0] - 2 p_15_minus[1] + C216 + [ - w0 ^2
   + 2 w0 * p_15_minus[0] - w1 ^2 + 2 w1 * p_15_minus[1]
   - 2 p_15_minus[0] ^2 - 2 p_15_minus[0] * p_15_minus[1]
   - 2 p_15_minus[1] ^2 ] = 1
 qc32: 2 p_16_plus[0] + 2 p_16_plus[1] + C219 + [ - w0 ^2
   - 2 w0 * p_16_plus[0] - w1 ^2 - 2 w1 * p_16_plus[1] - 2 p_16_plus[0] ^2
   - 2 p_16_plus[0] * p_16_plus[1] - 2 p_16_plus[1] ^2 ] = 1
 qc33: - 2 p_16_minus[0] - 2 p_16_minus[1] + C220 + [ - w0 ^2
   + 2 w0 * p_16_minus[0] - w1 ^2 + 2 w1 * p_16_minus[1]
   - 2 p_16_minus[0] ^2 - 2 p_16_minus[0] * p_16_minus[1]
   - 2 p_16_minus[1] ^2 ] = 1
 qc34: 2 p_17_plus[0] + 2 p_17_plus[1] + C223 + [ - w0 ^2
   - 2 w0 * p_17_plus[0] - w1 ^2 - 2 w1 * p_17_plus[1] - 2 p_17_plus[0] ^2
   - 2 p_17_plus[0] * p_17_plus[1] - 2 p_17_plus[1] ^2 ] = 1
 qc35: - 2 p_17_minus[0] - 2 p_17_minus[1] + C224 + [ - w0 ^2
   + 2 w0 * p_17_minus[0] - w1 ^2 + 2 w1 * p_17_minus[1]
   - 2 p_17_minus[0] ^2 - 2 p_17_minus[0] * p_17_minus[1]
   - 2 p_17_minus[1] ^2 ] = 1
 qc36: 2 p_18_plus[0] + 2 p_18_plus[1] + C227 + [ - w0 ^2
   - 2 w0 * p_18_plus[0] - w1 ^2 - 2 w1 * p_18_plus[1] - 2 p_18_plus[0] ^2
   - 2 p_18_plus[0] * p_18_plus[1] - 2 p_18_plus[1] ^2 ] = 1
 qc37: - 2 p_18_minus[0] - 2 p_18_minus[1] + C228 + [ - w0 ^2
   + 2 w0 * p_18_minus[0] - w1 ^2 + 2 w1 * p_18_minus[1]
   - 2 p_18_minus[0] ^2 - 2 p_18_minus[0] * p_18_minus[1]
   - 2 p_18_minus[1] ^2 ] = 1
 qc38: 2 p_19_plus[0] + 2 p_19_plus[1] + C231 + [ - w0 ^2
   - 2 w0 * p_19_plus[0] - w1 ^2 - 2 w1 * p_19_plus[1] - 2 p_19_plus[0] ^2
   - 2 p_19_plus[0] * p_19_plus[1] - 2 p_19_plus[1] ^2 ] = 1
 qc39: - 2 p_19_minus[0] - 2 p_19_minus[1] + C232 + [ - w0 ^2
   + 2 w0 * p_19_minus[0] - w1 ^2 + 2 w1 * p_19_minus[1]
   - 2 p_19_minus[0] ^2 - 2 p_19_minus[0] * p_19_minus[1]
   - 2 p_19_minus[1] ^2 ] = 1
 qc40: 2 p_20_plus[0] + 2 p_20_plus[1] + C235 + [ - w0 ^2
   - 2 w0 * p_20_plus[0] - w1 ^2 - 2 w1 * p_20_plus[1] - 2 p_20_plus[0] ^2
   - 2 p_20_plus[0] * p_20_plus[1] - 2 p_20_plus[1] ^2 ] = 1
 qc41: - 2 p_20_minus[0] - 2 p_20_minus[1] + C236 + [ - w0 ^2
   + 2 w0 * p_20_minus[0] - w1 ^2 + 2 w1 * p_20_minus[1]
   - 2 p_20_minus[0] ^2 - 2 p_20_minus[0] * p_20_minus[1]
   - 2 p_20_minus[1] ^2 ] = 1
 qc42: 2 p_21_plus[0] + 2 p_21_plus[1] + C239 + [ - w0 ^2
   - 2 w0 * p_21_plus[0] - w1 ^2 - 2 w1 * p_21_plus[1] - 2 p_21_plus[0] ^2
   - 2 p_21_plus[0] * p_21_plus[1] - 2 p_21_plus[1] ^2 ] = 1
 qc43: - 2 p_21_minus[0] - 2 p_21_minus[1] + C240 + [ - w0 ^2
   + 2 w0 * p_21_minus[0] - w1 ^2 + 2 w1 * p_21_minus[1]
   - 2 p_21_minus[0] ^2 - 2 p_21_minus[0] * p_21_minus[1]
   - 2 p_21_minus[1] ^2 ] = 1
 qc44: 2 p_22_plus[0] + 2 p_22_plus[1] + C243 + [ - w0 ^2
   - 2 w0 * p_22_plus[0] - w1 ^2 - 2 w1 * p_22_plus[1] - 2 p_22_plus[0] ^2
   - 2 p_22_plus[0] * p_22_plus[1] - 2 p_22_plus[1] ^2 ] = 1
 qc45: - 2 p_22_minus[0] - 2 p_22_minus[1] + C244 + [ - w0 ^2
   + 2 w0 * p_22_minus[0] - w1 ^2 + 2 w1 * p_22_minus[1]
   - 2 p_22_minus[0] ^2 - 2 p_22_minus[0] * p_22_minus[1]
   - 2 p_22_minus[1] ^2 ] = 1
 qc46: 2 p_23_plus[0] + 2 p_23_plus[1] + C247 + [ - w0 ^2
   - 2 w0 * p_23_plus[0] - w1 ^2 - 2 w1 * p_23_plus[1] - 2 p_23_plus[0] ^2
   - 2 p_23_plus[0] * p_23_plus[1] - 2 p_23_plus[1] ^2 ] = 1
 qc47: - 2 p_23_minus[0] - 2 p_23_minus[1] + C248 + [ - w0 ^2
   + 2 w0 * p_23_minus[0] - w1 ^2 + 2 w1 * p_23_minus[1]
   - 2 p_23_minus[0] ^2 - 2 p_23_minus[0] * p_23_minus[1]
   - 2 p_23_minus[1] ^2 ] = 1
 qc48: 2 p_24_plus[0] + 2 p_24_plus[1] + C251 + [ - w0 ^2
   - 2 w0 * p_24_plus[0] - w1 ^2 - 2 w1 * p_24_plus[1] - 2 p_24_plus[0] ^2
   - 2 p_24_plus[0] * p_24_plus[1] - 2 p_24_plus[1] ^2 ] = 1
 qc49: - 2 p_24_minus[0] - 2 p_24_minus[1] + C252 + [ - w0 ^2
   + 2 w0 * p_24_minus[0] - w1 ^2 + 2 w1 * p_24_minus[1]
   - 2 p_24_minus[0] ^2 - 2 p_24_minus[0] * p_24_minus[1]
   - 2 p_24_minus[1] ^2 ] = 1
 qc50: 2 p_25_plus[0] + 2 p_25_plus[1] + C255 + [ - w0 ^2
   - 2 w0 * p_25_plus[0] - w1 ^2 - 2 w1 * p_25_plus[1] - 2 p_25_plus[0] ^2
   - 2 p_25_plus[0] * p_25_plus[1] - 2 p_25_plus[1] ^2 ] = 1
 qc51: - 2 p_25_minus[0] - 2 p_25_minus[1] + C256 + [ - w0 ^2
   + 2 w0 * p_25_minus[0] - w1 ^2 + 2 w1 * p_25_minus[1]
   - 2 p_25_minus[0] ^2 - 2 p_25_minus[0] * p_25_minus[1]
   - 2 p_25_minus[1] ^2 ] = 1
 qc52: 2 p_26_plus[0] + 2 p_26_plus[1] + C259 + [ - w0 ^2
   - 2 w0 * p_26_plus[0] - w1 ^2 - 2 w1 * p_26_plus[1] - 2 p_26_plus[0] ^2
   - 2 p_26_plus[0] * p_26_plus[1] - 2 p_26_plus[1] ^2 ] = 1
 qc53: - 2 p_26_minus[0] - 2 p_26_minus[1] + C260 + [ - w0 ^2
   + 2 w0 * p_26_minus[0] - w1 ^2 + 2 w1 * p_26_minus[1]
   - 2 p_26_minus[0] ^2 - 2 p_26_minus[0] * p_26_minus[1]
   - 2 p_26_minus[1] ^2 ] = 1
 qc54: 2 p_27_plus[0] + 2 p_27_plus[1] + C263 + [ - w0 ^2
   - 2 w0 * p_27_plus[0] - w1 ^2 - 2 w1 * p_27_plus[1] - 2 p_27_plus[0] ^2
   - 2 p_27_plus[0] * p_27_plus[1] - 2 p_27_plus[1] ^2 ] = 1
 qc55: - 2 p_27_minus[0] - 2 p_27_minus[1] + C264 + [ - w0 ^2
   + 2 w0 * p_27_minus[0] - w1 ^2 + 2 w1 * p_27_minus[1]
   - 2 p_27_minus[0] ^2 - 2 p_27_minus[0] * p_27_minus[1]
   - 2 p_27_minus[1] ^2 ] = 1
 qc56: 2 p_28_plus[0] + 2 p_28_plus[1] + C267 + [ - w0 ^2
   - 2 w0 * p_28_plus[0] - w1 ^2 - 2 w1 * p_28_plus[1] - 2 p_28_plus[0] ^2
   - 2 p_28_plus[0] * p_28_plus[1] - 2 p_28_plus[1] ^2 ] = 1
 qc57: - 2 p_28_minus[0] - 2 p_28_minus[1] + C268 + [ - w0 ^2
   + 2 w0 * p_28_minus[0] - w1 ^2 + 2 w1 * p_28_minus[1]
   - 2 p_28_minus[0] ^2 - 2 p_28_minus[0] * p_28_minus[1]
   - 2 p_28_minus[1] ^2 ] = 1
 qc58: 2 p_29_plus[0] + 2 p_29_plus[1] + C271 + [ - w0 ^2
   - 2 w0 * p_29_plus[0] - w1 ^2 - 2 w1 * p_29_plus[1] - 2 p_29_plus[0] ^2
   - 2 p_29_plus[0] * p_29_plus[1] - 2 p_29_plus[1] ^2 ] = 1
 qc59: - 2 p_29_minus[0] - 2 p_29_minus[1] + C272 + [ - w0 ^2
   + 2 w0 * p_29_minus[0] - w1 ^2 + 2 w1 * p_29_minus[1]
   - 2 p_29_minus[0] ^2 - 2 p_29_minus[0] * p_29_minus[1]
   - 2 p_29_minus[1] ^2 ] = 1
Bounds
 w0 free
 w1 free
General Constraints
 GC0: C153 = POW ( C155 ^ 0.5 )
 GC1: C154 = POW ( C156 ^ 0.5 )
 GC2: C157 = POW ( C159 ^ 0.5 )
 GC3: C158 = POW ( C160 ^ 0.5 )
 GC4: C161 = POW ( C163 ^ 0.5 )
 GC5: C162 = POW ( C164 ^ 0.5 )
 GC6: C165 = POW ( C167 ^ 0.5 )
 GC7: C166 = POW ( C168 ^ 0.5 )
 GC8: C169 = POW ( C171 ^ 0.5 )
 GC9: C170 = POW ( C172 ^ 0.5 )
 GC10: C173 = POW ( C175 ^ 0.5 )
 GC11: C174 = POW ( C176 ^ 0.5 )
 GC12: C177 = POW ( C179 ^ 0.5 )
 GC13: C178 = POW ( C180 ^ 0.5 )
 GC14: C181 = POW ( C183 ^ 0.5 )
 GC15: C182 = POW ( C184 ^ 0.5 )
 GC16: C185 = POW ( C187 ^ 0.5 )
 GC17: C186 = POW ( C188 ^ 0.5 )
 GC18: C189 = POW ( C191 ^ 0.5 )
 GC19: C190 = POW ( C192 ^ 0.5 )
 GC20: C193 = POW ( C195 ^ 0.5 )
 GC21: C194 = POW ( C196 ^ 0.5 )
 GC22: C197 = POW ( C199 ^ 0.5 )
 GC23: C198 = POW ( C200 ^ 0.5 )
 GC24: C201 = POW ( C203 ^ 0.5 )
 GC25: C202 = POW ( C204 ^ 0.5 )
 GC26: C205 = POW ( C207 ^ 0.5 )
 GC27: C206 = POW ( C208 ^ 0.5 )
 GC28: C209 = POW ( C211 ^ 0.5 )
 GC29: C210 = POW ( C212 ^ 0.5 )
 GC30: C213 = POW ( C215 ^ 0.5 )
 GC31: C214 = POW ( C216 ^ 0.5 )
 GC32: C217 = POW ( C219 ^ 0.5 )
 GC33: C218 = POW ( C220 ^ 0.5 )
 GC34: C221 = POW ( C223 ^ 0.5 )
 GC35: C222 = POW ( C224 ^ 0.5 )
 GC36: C225 = POW ( C227 ^ 0.5 )
 GC37: C226 = POW ( C228 ^ 0.5 )
 GC38: C229 = POW ( C231 ^ 0.5 )
 GC39: C230 = POW ( C232 ^ 0.5 )
 GC40: C233 = POW ( C235 ^ 0.5 )
 GC41: C234 = POW ( C236 ^ 0.5 )
 GC42: C237 = POW ( C239 ^ 0.5 )
 GC43: C238 = POW ( C240 ^ 0.5 )
 GC44: C241 = POW ( C243 ^ 0.5 )
 GC45: C242 = POW ( C244 ^ 0.5 )
 GC46: C245 = POW ( C247 ^ 0.5 )
 GC47: C246 = POW ( C248 ^ 0.5 )
 GC48: C249 = POW ( C251 ^ 0.5 )
 GC49: C250 = POW ( C252 ^ 0.5 )
 GC50: C253 = POW ( C255 ^ 0.5 )
 GC51: C254 = POW ( C256 ^ 0.5 )
 GC52: C257 = POW ( C259 ^ 0.5 )
 GC53: C258 = POW ( C260 ^ 0.5 )
 GC54: C261 = POW ( C263 ^ 0.5 )
 GC55: C262 = POW ( C264 ^ 0.5 )
 GC56: C265 = POW ( C267 ^ 0.5 )
 GC57: C266 = POW ( C268 ^ 0.5 )
 GC58: C269 = POW ( C271 ^ 0.5 )
 GC59: C270 = POW ( C272 ^ 0.5 )
End
