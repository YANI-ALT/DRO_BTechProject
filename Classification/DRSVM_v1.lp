\ Model Support_Vector_Machine
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 Lambda + 0.01 s_0 + 0.01 s_1 + 0.01 s_2 + 0.01 s_3 + 0.01 s_4
   + 0.01 s_5 + 0.01 s_6 + 0.01 s_7 + 0.01 s_8 + 0.01 s_9 + 0.01 s_10
   + 0.01 s_11 + 0.01 s_12 + 0.01 s_13 + 0.01 s_14 + 0.01 s_15 + 0.01 s_16
   + 0.01 s_17 + 0.01 s_18 + 0.01 s_19 + 0.01 s_20 + 0.01 s_21 + 0.01 s_22
   + 0.01 s_23 + 0.01 s_24 + 0.01 s_25 + 0.01 s_26 + 0.01 s_27 + 0.01 s_28
   + 0.01 s_29 + 0.01 s_30 + 0.01 s_31 + 0.01 s_32 + 0.01 s_33 + 0.01 s_34
   + 0.01 s_35 + 0.01 s_36 + 0.01 s_37 + 0.01 s_38 + 0.01 s_39 + 0.01 s_40
   + 0.01 s_41 + 0.01 s_42 + 0.01 s_43 + 0.01 s_44 + 0.01 s_45 + 0.01 s_46
   + 0.01 s_47 + 0.01 s_48 + 0.01 s_49 + 0.01 s_50 + 0.01 s_51 + 0.01 s_52
   + 0.01 s_53 + 0.01 s_54 + 0.01 s_55 + 0.01 s_56 + 0.01 s_57 + 0.01 s_58
   + 0.01 s_59 + 0.01 s_60 + 0.01 s_61 + 0.01 s_62 + 0.01 s_63 + 0.01 s_64
   + 0.01 s_65 + 0.01 s_66 + 0.01 s_67 + 0.01 s_68 + 0.01 s_69 + 0.01 s_70
   + 0.01 s_71 + 0.01 s_72 + 0.01 s_73 + 0.01 s_74 + 0.01 s_75 + 0.01 s_76
   + 0.01 s_77 + 0.01 s_78 + 0.01 s_79 + 0.01 s_80 + 0.01 s_81 + 0.01 s_82
   + 0.01 s_83 + 0.01 s_84 + 0.01 s_85 + 0.01 s_86 + 0.01 s_87 + 0.01 s_88
   + 0.01 s_89 + 0.01 s_90 + 0.01 s_91 + 0.01 s_92 + 0.01 s_93 + 0.01 s_94
   + 0.01 s_95 + 0.01 s_96 + 0.01 s_97 + 0.01 s_98 + 0.01 s_99
Subject To
 data[0]_constr1: w0 + 5.1 w1 + 3.5 w2 - s_0 + 7.9 p_0_plus[0]
   + 3.8 p_0_plus[1] + 5.4 p_0_plus[2] <= -1
 data[0]_constr2: - w0 - 5.1 w1 - 3.5 w2 - 0.1 Lambda - s_0
   + 7.9 p_0_minus[0] + 3.8 p_0_minus[1] + 5.4 p_0_minus[2] <= -1
 data[1]_constr1: w0 + 4.9 w1 + 3 w2 - s_1 + 7.9 p_1_plus[0]
   + 4 p_1_plus[1] + 5.9 p_1_plus[2] <= -1
 data[1]_constr2: - w0 - 4.9 w1 - 3 w2 - 0.1 Lambda - s_1
   + 7.9 p_1_minus[0] + 4 p_1_minus[1] + 5.9 p_1_minus[2] <= -1
 data[2]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_2 + 7.9 p_2_plus[0]
   + 4.2 p_2_plus[1] + 5.7 p_2_plus[2] <= -1
 data[2]_constr2: - w0 - 4.7 w1 - 3.2 w2 - 0.1 Lambda - s_2
   + 7.9 p_2_minus[0] + 4.2 p_2_minus[1] + 5.7 p_2_minus[2] <= -1
 data[3]_constr1: w0 + 4.6 w1 + 3.1 w2 - s_3 + 7.9 p_3_plus[0]
   + 4.3 p_3_plus[1] + 5.8 p_3_plus[2] <= -1
 data[3]_constr2: - w0 - 4.6 w1 - 3.1 w2 - 0.1 Lambda - s_3
   + 7.9 p_3_minus[0] + 4.3 p_3_minus[1] + 5.8 p_3_minus[2] <= -1
 data[4]_constr1: w0 + 5 w1 + 3.6 w2 - s_4 + 7.9 p_4_plus[0]
   + 3.9 p_4_plus[1] + 5.3 p_4_plus[2] <= -1
 data[4]_constr2: - w0 - 5 w1 - 3.6 w2 - 0.1 Lambda - s_4
   + 7.9 p_4_minus[0] + 3.9 p_4_minus[1] + 5.3 p_4_minus[2] <= -1
 data[5]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_5 + 7.9 p_5_plus[0]
   + 3.5 p_5_plus[1] + 5 p_5_plus[2] <= -1
 data[5]_constr2: - w0 - 5.4 w1 - 3.9 w2 - 0.1 Lambda - s_5
   + 7.9 p_5_minus[0] + 3.5 p_5_minus[1] + 5 p_5_minus[2] <= -1
 data[6]_constr1: w0 + 4.6 w1 + 3.4 w2 - s_6 + 7.9 p_6_plus[0]
   + 4.3 p_6_plus[1] + 5.5 p_6_plus[2] <= -1
 data[6]_constr2: - w0 - 4.6 w1 - 3.4 w2 - 0.1 Lambda - s_6
   + 7.9 p_6_minus[0] + 4.3 p_6_minus[1] + 5.5 p_6_minus[2] <= -1
 data[7]_constr1: w0 + 5 w1 + 3.4 w2 - s_7 + 7.9 p_7_plus[0]
   + 3.9 p_7_plus[1] + 5.5 p_7_plus[2] <= -1
 data[7]_constr2: - w0 - 5 w1 - 3.4 w2 - 0.1 Lambda - s_7
   + 7.9 p_7_minus[0] + 3.9 p_7_minus[1] + 5.5 p_7_minus[2] <= -1
 data[8]_constr1: w0 + 4.4 w1 + 2.9 w2 - s_8 + 7.9 p_8_plus[0]
   + 4.5 p_8_plus[1] + 6 p_8_plus[2] <= -1
 data[8]_constr2: - w0 - 4.4 w1 - 2.9 w2 - 0.1 Lambda - s_8
   + 7.9 p_8_minus[0] + 4.5 p_8_minus[1] + 6 p_8_minus[2] <= -1
 data[9]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_9 + 7.9 p_9_plus[0]
   + 4 p_9_plus[1] + 5.8 p_9_plus[2] <= -1
 data[9]_constr2: - w0 - 4.9 w1 - 3.1 w2 - 0.1 Lambda - s_9
   + 7.9 p_9_minus[0] + 4 p_9_minus[1] + 5.8 p_9_minus[2] <= -1
 data[10]_constr1: w0 + 5.4 w1 + 3.7 w2 - s_10 + 7.9 p_10_plus[0]
   + 3.5 p_10_plus[1] + 5.2 p_10_plus[2] <= -1
 data[10]_constr2: - w0 - 5.4 w1 - 3.7 w2 - 0.1 Lambda - s_10
   + 7.9 p_10_minus[0] + 3.5 p_10_minus[1] + 5.2 p_10_minus[2] <= -1
 data[11]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_11 + 7.9 p_11_plus[0]
   + 4.1 p_11_plus[1] + 5.5 p_11_plus[2] <= -1
 data[11]_constr2: - w0 - 4.8 w1 - 3.4 w2 - 0.1 Lambda - s_11
   + 7.9 p_11_minus[0] + 4.1 p_11_minus[1] + 5.5 p_11_minus[2] <= -1
 data[12]_constr1: w0 + 4.8 w1 + 3 w2 - s_12 + 7.9 p_12_plus[0]
   + 4.1 p_12_plus[1] + 5.9 p_12_plus[2] <= -1
 data[12]_constr2: - w0 - 4.8 w1 - 3 w2 - 0.1 Lambda - s_12
   + 7.9 p_12_minus[0] + 4.1 p_12_minus[1] + 5.9 p_12_minus[2] <= -1
 data[13]_constr1: w0 + 4.3 w1 + 3 w2 - s_13 + 7.9 p_13_plus[0]
   + 4.6 p_13_plus[1] + 5.9 p_13_plus[2] <= -1
 data[13]_constr2: - w0 - 4.3 w1 - 3 w2 - 0.1 Lambda - s_13
   + 7.9 p_13_minus[0] + 4.6 p_13_minus[1] + 5.9 p_13_minus[2] <= -1
 data[14]_constr1: w0 + 5.8 w1 + 4 w2 - s_14 + 7.9 p_14_plus[0]
   + 3.1 p_14_plus[1] + 4.9 p_14_plus[2] <= -1
 data[14]_constr2: - w0 - 5.8 w1 - 4 w2 - 0.1 Lambda - s_14
   + 7.9 p_14_minus[0] + 3.1 p_14_minus[1] + 4.9 p_14_minus[2] <= -1
 data[15]_constr1: w0 + 5.7 w1 + 4.4 w2 - s_15 + 7.9 p_15_plus[0]
   + 3.2 p_15_plus[1] + 4.5 p_15_plus[2] <= -1
 data[15]_constr2: - w0 - 5.7 w1 - 4.4 w2 - 0.1 Lambda - s_15
   + 7.9 p_15_minus[0] + 3.2 p_15_minus[1] + 4.5 p_15_minus[2] <= -1
 data[16]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_16 + 7.9 p_16_plus[0]
   + 3.5 p_16_plus[1] + 5 p_16_plus[2] <= -1
 data[16]_constr2: - w0 - 5.4 w1 - 3.9 w2 - 0.1 Lambda - s_16
   + 7.9 p_16_minus[0] + 3.5 p_16_minus[1] + 5 p_16_minus[2] <= -1
 data[17]_constr1: w0 + 5.1 w1 + 3.5 w2 - s_17 + 7.9 p_17_plus[0]
   + 3.8 p_17_plus[1] + 5.4 p_17_plus[2] <= -1
 data[17]_constr2: - w0 - 5.1 w1 - 3.5 w2 - 0.1 Lambda - s_17
   + 7.9 p_17_minus[0] + 3.8 p_17_minus[1] + 5.4 p_17_minus[2] <= -1
 data[18]_constr1: w0 + 5.7 w1 + 3.8 w2 - s_18 + 7.9 p_18_plus[0]
   + 3.2 p_18_plus[1] + 5.1 p_18_plus[2] <= -1
 data[18]_constr2: - w0 - 5.7 w1 - 3.8 w2 - 0.1 Lambda - s_18
   + 7.9 p_18_minus[0] + 3.2 p_18_minus[1] + 5.1 p_18_minus[2] <= -1
 data[19]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_19 + 7.9 p_19_plus[0]
   + 3.8 p_19_plus[1] + 5.1 p_19_plus[2] <= -1
 data[19]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_19
   + 7.9 p_19_minus[0] + 3.8 p_19_minus[1] + 5.1 p_19_minus[2] <= -1
 data[20]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_20 + 7.9 p_20_plus[0]
   + 3.5 p_20_plus[1] + 5.5 p_20_plus[2] <= -1
 data[20]_constr2: - w0 - 5.4 w1 - 3.4 w2 - 0.1 Lambda - s_20
   + 7.9 p_20_minus[0] + 3.5 p_20_minus[1] + 5.5 p_20_minus[2] <= -1
 data[21]_constr1: w0 + 5.1 w1 + 3.7 w2 - s_21 + 7.9 p_21_plus[0]
   + 3.8 p_21_plus[1] + 5.2 p_21_plus[2] <= -1
 data[21]_constr2: - w0 - 5.1 w1 - 3.7 w2 - 0.1 Lambda - s_21
   + 7.9 p_21_minus[0] + 3.8 p_21_minus[1] + 5.2 p_21_minus[2] <= -1
 data[22]_constr1: w0 + 4.6 w1 + 3.6 w2 - s_22 + 7.9 p_22_plus[0]
   + 4.3 p_22_plus[1] + 5.3 p_22_plus[2] <= -1
 data[22]_constr2: - w0 - 4.6 w1 - 3.6 w2 - 0.1 Lambda - s_22
   + 7.9 p_22_minus[0] + 4.3 p_22_minus[1] + 5.3 p_22_minus[2] <= -1
 data[23]_constr1: w0 + 5.1 w1 + 3.3 w2 - s_23 + 7.9 p_23_plus[0]
   + 3.8 p_23_plus[1] + 5.6 p_23_plus[2] <= -1
 data[23]_constr2: - w0 - 5.1 w1 - 3.3 w2 - 0.1 Lambda - s_23
   + 7.9 p_23_minus[0] + 3.8 p_23_minus[1] + 5.6 p_23_minus[2] <= -1
 data[24]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_24 + 7.9 p_24_plus[0]
   + 4.1 p_24_plus[1] + 5.5 p_24_plus[2] <= -1
 data[24]_constr2: - w0 - 4.8 w1 - 3.4 w2 - 0.1 Lambda - s_24
   + 7.9 p_24_minus[0] + 4.1 p_24_minus[1] + 5.5 p_24_minus[2] <= -1
 data[25]_constr1: w0 + 5 w1 + 3 w2 - s_25 + 7.9 p_25_plus[0]
   + 3.9 p_25_plus[1] + 5.9 p_25_plus[2] <= -1
 data[25]_constr2: - w0 - 5 w1 - 3 w2 - 0.1 Lambda - s_25
   + 7.9 p_25_minus[0] + 3.9 p_25_minus[1] + 5.9 p_25_minus[2] <= -1
 data[26]_constr1: w0 + 5 w1 + 3.4 w2 - s_26 + 7.9 p_26_plus[0]
   + 3.9 p_26_plus[1] + 5.5 p_26_plus[2] <= -1
 data[26]_constr2: - w0 - 5 w1 - 3.4 w2 - 0.1 Lambda - s_26
   + 7.9 p_26_minus[0] + 3.9 p_26_minus[1] + 5.5 p_26_minus[2] <= -1
 data[27]_constr1: w0 + 5.2 w1 + 3.5 w2 - s_27 + 7.9 p_27_plus[0]
   + 3.7 p_27_plus[1] + 5.4 p_27_plus[2] <= -1
 data[27]_constr2: - w0 - 5.2 w1 - 3.5 w2 - 0.1 Lambda - s_27
   + 7.9 p_27_minus[0] + 3.7 p_27_minus[1] + 5.4 p_27_minus[2] <= -1
 data[28]_constr1: w0 + 5.2 w1 + 3.4 w2 - s_28 + 7.9 p_28_plus[0]
   + 3.7 p_28_plus[1] + 5.5 p_28_plus[2] <= -1
 data[28]_constr2: - w0 - 5.2 w1 - 3.4 w2 - 0.1 Lambda - s_28
   + 7.9 p_28_minus[0] + 3.7 p_28_minus[1] + 5.5 p_28_minus[2] <= -1
 data[29]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_29 + 7.9 p_29_plus[0]
   + 4.2 p_29_plus[1] + 5.7 p_29_plus[2] <= -1
 data[29]_constr2: - w0 - 4.7 w1 - 3.2 w2 - 0.1 Lambda - s_29
   + 7.9 p_29_minus[0] + 4.2 p_29_minus[1] + 5.7 p_29_minus[2] <= -1
 data[30]_constr1: w0 + 4.8 w1 + 3.1 w2 - s_30 + 7.9 p_30_plus[0]
   + 4.1 p_30_plus[1] + 5.8 p_30_plus[2] <= -1
 data[30]_constr2: - w0 - 4.8 w1 - 3.1 w2 - 0.1 Lambda - s_30
   + 7.9 p_30_minus[0] + 4.1 p_30_minus[1] + 5.8 p_30_minus[2] <= -1
 data[31]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_31 + 7.9 p_31_plus[0]
   + 3.5 p_31_plus[1] + 5.5 p_31_plus[2] <= -1
 data[31]_constr2: - w0 - 5.4 w1 - 3.4 w2 - 0.1 Lambda - s_31
   + 7.9 p_31_minus[0] + 3.5 p_31_minus[1] + 5.5 p_31_minus[2] <= -1
 data[32]_constr1: w0 + 5.2 w1 + 4.1 w2 - s_32 + 7.9 p_32_plus[0]
   + 3.7 p_32_plus[1] + 4.8 p_32_plus[2] <= -1
 data[32]_constr2: - w0 - 5.2 w1 - 4.1 w2 - 0.1 Lambda - s_32
   + 7.9 p_32_minus[0] + 3.7 p_32_minus[1] + 4.8 p_32_minus[2] <= -1
 data[33]_constr1: w0 + 5.5 w1 + 4.2 w2 - s_33 + 7.9 p_33_plus[0]
   + 3.4 p_33_plus[1] + 4.7 p_33_plus[2] <= -1
 data[33]_constr2: - w0 - 5.5 w1 - 4.2 w2 - 0.1 Lambda - s_33
   + 7.9 p_33_minus[0] + 3.4 p_33_minus[1] + 4.7 p_33_minus[2] <= -1
 data[34]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_34 + 7.9 p_34_plus[0]
   + 4 p_34_plus[1] + 5.8 p_34_plus[2] <= -1
 data[34]_constr2: - w0 - 4.9 w1 - 3.1 w2 - 0.1 Lambda - s_34
   + 7.9 p_34_minus[0] + 4 p_34_minus[1] + 5.8 p_34_minus[2] <= -1
 data[35]_constr1: w0 + 5 w1 + 3.2 w2 - s_35 + 7.9 p_35_plus[0]
   + 3.9 p_35_plus[1] + 5.7 p_35_plus[2] <= -1
 data[35]_constr2: - w0 - 5 w1 - 3.2 w2 - 0.1 Lambda - s_35
   + 7.9 p_35_minus[0] + 3.9 p_35_minus[1] + 5.7 p_35_minus[2] <= -1
 data[36]_constr1: w0 + 5.5 w1 + 3.5 w2 - s_36 + 7.9 p_36_plus[0]
   + 3.4 p_36_plus[1] + 5.4 p_36_plus[2] <= -1
 data[36]_constr2: - w0 - 5.5 w1 - 3.5 w2 - 0.1 Lambda - s_36
   + 7.9 p_36_minus[0] + 3.4 p_36_minus[1] + 5.4 p_36_minus[2] <= -1
 data[37]_constr1: w0 + 4.9 w1 + 3.6 w2 - s_37 + 7.9 p_37_plus[0]
   + 4 p_37_plus[1] + 5.3 p_37_plus[2] <= -1
 data[37]_constr2: - w0 - 4.9 w1 - 3.6 w2 - 0.1 Lambda - s_37
   + 7.9 p_37_minus[0] + 4 p_37_minus[1] + 5.3 p_37_minus[2] <= -1
 data[38]_constr1: w0 + 4.4 w1 + 3 w2 - s_38 + 7.9 p_38_plus[0]
   + 4.5 p_38_plus[1] + 5.9 p_38_plus[2] <= -1
 data[38]_constr2: - w0 - 4.4 w1 - 3 w2 - 0.1 Lambda - s_38
   + 7.9 p_38_minus[0] + 4.5 p_38_minus[1] + 5.9 p_38_minus[2] <= -1
 data[39]_constr1: w0 + 5.1 w1 + 3.4 w2 - s_39 + 7.9 p_39_plus[0]
   + 3.8 p_39_plus[1] + 5.5 p_39_plus[2] <= -1
 data[39]_constr2: - w0 - 5.1 w1 - 3.4 w2 - 0.1 Lambda - s_39
   + 7.9 p_39_minus[0] + 3.8 p_39_minus[1] + 5.5 p_39_minus[2] <= -1
 data[40]_constr1: w0 + 5 w1 + 3.5 w2 - s_40 + 7.9 p_40_plus[0]
   + 3.9 p_40_plus[1] + 5.4 p_40_plus[2] <= -1
 data[40]_constr2: - w0 - 5 w1 - 3.5 w2 - 0.1 Lambda - s_40
   + 7.9 p_40_minus[0] + 3.9 p_40_minus[1] + 5.4 p_40_minus[2] <= -1
 data[41]_constr1: w0 + 4.5 w1 + 2.3 w2 - s_41 + 7.9 p_41_plus[0]
   + 4.4 p_41_plus[1] + 6.6 p_41_plus[2] <= -1
 data[41]_constr2: - w0 - 4.5 w1 - 2.3 w2 - 0.1 Lambda - s_41
   + 7.9 p_41_minus[0] + 4.4 p_41_minus[1] + 6.6 p_41_minus[2] <= -1
 data[42]_constr1: w0 + 4.4 w1 + 3.2 w2 - s_42 + 7.9 p_42_plus[0]
   + 4.5 p_42_plus[1] + 5.7 p_42_plus[2] <= -1
 data[42]_constr2: - w0 - 4.4 w1 - 3.2 w2 - 0.1 Lambda - s_42
   + 7.9 p_42_minus[0] + 4.5 p_42_minus[1] + 5.7 p_42_minus[2] <= -1
 data[43]_constr1: w0 + 5 w1 + 3.5 w2 - s_43 + 7.9 p_43_plus[0]
   + 3.9 p_43_plus[1] + 5.4 p_43_plus[2] <= -1
 data[43]_constr2: - w0 - 5 w1 - 3.5 w2 - 0.1 Lambda - s_43
   + 7.9 p_43_minus[0] + 3.9 p_43_minus[1] + 5.4 p_43_minus[2] <= -1
 data[44]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_44 + 7.9 p_44_plus[0]
   + 3.8 p_44_plus[1] + 5.1 p_44_plus[2] <= -1
 data[44]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_44
   + 7.9 p_44_minus[0] + 3.8 p_44_minus[1] + 5.1 p_44_minus[2] <= -1
 data[45]_constr1: w0 + 4.8 w1 + 3 w2 - s_45 + 7.9 p_45_plus[0]
   + 4.1 p_45_plus[1] + 5.9 p_45_plus[2] <= -1
 data[45]_constr2: - w0 - 4.8 w1 - 3 w2 - 0.1 Lambda - s_45
   + 7.9 p_45_minus[0] + 4.1 p_45_minus[1] + 5.9 p_45_minus[2] <= -1
 data[46]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_46 + 7.9 p_46_plus[0]
   + 3.8 p_46_plus[1] + 5.1 p_46_plus[2] <= -1
 data[46]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_46
   + 7.9 p_46_minus[0] + 3.8 p_46_minus[1] + 5.1 p_46_minus[2] <= -1
 data[47]_constr1: w0 + 4.6 w1 + 3.2 w2 - s_47 + 7.9 p_47_plus[0]
   + 4.3 p_47_plus[1] + 5.7 p_47_plus[2] <= -1
 data[47]_constr2: - w0 - 4.6 w1 - 3.2 w2 - 0.1 Lambda - s_47
   + 7.9 p_47_minus[0] + 4.3 p_47_minus[1] + 5.7 p_47_minus[2] <= -1
 data[48]_constr1: w0 + 5.3 w1 + 3.7 w2 - s_48 + 7.9 p_48_plus[0]
   + 3.6 p_48_plus[1] + 5.2 p_48_plus[2] <= -1
 data[48]_constr2: - w0 - 5.3 w1 - 3.7 w2 - 0.1 Lambda - s_48
   + 7.9 p_48_minus[0] + 3.6 p_48_minus[1] + 5.2 p_48_minus[2] <= -1
 data[49]_constr1: w0 + 5 w1 + 3.3 w2 - s_49 + 7.9 p_49_plus[0]
   + 3.9 p_49_plus[1] + 5.6 p_49_plus[2] <= -1
 data[49]_constr2: - w0 - 5 w1 - 3.3 w2 - 0.1 Lambda - s_49
   + 7.9 p_49_minus[0] + 3.9 p_49_minus[1] + 5.6 p_49_minus[2] <= -1
 data[50]_constr1: - w0 - 6.3 w1 - 3.3 w2 - s_50 + 7.9 p_50_plus[0]
   + 2.6 p_50_plus[1] + 5.6 p_50_plus[2] <= -1
 data[50]_constr2: w0 + 6.3 w1 + 3.3 w2 - 0.1 Lambda - s_50
   + 7.9 p_50_minus[0] + 2.6 p_50_minus[1] + 5.6 p_50_minus[2] <= -1
 data[51]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_51 + 7.9 p_51_plus[0]
   + 3.1 p_51_plus[1] + 6.2 p_51_plus[2] <= -1
 data[51]_constr2: w0 + 5.8 w1 + 2.7 w2 - 0.1 Lambda - s_51
   + 7.9 p_51_minus[0] + 3.1 p_51_minus[1] + 6.2 p_51_minus[2] <= -1
 data[52]_constr1: - w0 - 7.1 w1 - 3 w2 - s_52 + 7.9 p_52_plus[0]
   + 1.8 p_52_plus[1] + 5.9 p_52_plus[2] <= -1
 data[52]_constr2: w0 + 7.1 w1 + 3 w2 - 0.1 Lambda - s_52
   + 7.9 p_52_minus[0] + 1.8 p_52_minus[1] + 5.9 p_52_minus[2] <= -1
 data[53]_constr1: - w0 - 6.3 w1 - 2.9 w2 - s_53 + 7.9 p_53_plus[0]
   + 2.6 p_53_plus[1] + 6 p_53_plus[2] <= -1
 data[53]_constr2: w0 + 6.3 w1 + 2.9 w2 - 0.1 Lambda - s_53
   + 7.9 p_53_minus[0] + 2.6 p_53_minus[1] + 6 p_53_minus[2] <= -1
 data[54]_constr1: - w0 - 6.5 w1 - 3 w2 - s_54 + 7.9 p_54_plus[0]
   + 2.4 p_54_plus[1] + 5.9 p_54_plus[2] <= -1
 data[54]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_54
   + 7.9 p_54_minus[0] + 2.4 p_54_minus[1] + 5.9 p_54_minus[2] <= -1
 data[55]_constr1: - w0 - 7.6 w1 - 3 w2 - s_55 + 7.9 p_55_plus[0]
   + 1.3 p_55_plus[1] + 5.9 p_55_plus[2] <= -1
 data[55]_constr2: w0 + 7.6 w1 + 3 w2 - 0.1 Lambda - s_55
   + 7.9 p_55_minus[0] + 1.3 p_55_minus[1] + 5.9 p_55_minus[2] <= -1
 data[56]_constr1: - w0 - 4.9 w1 - 2.5 w2 - s_56 + 7.9 p_56_plus[0]
   + 4 p_56_plus[1] + 6.4 p_56_plus[2] <= -1
 data[56]_constr2: w0 + 4.9 w1 + 2.5 w2 - 0.1 Lambda - s_56
   + 7.9 p_56_minus[0] + 4 p_56_minus[1] + 6.4 p_56_minus[2] <= -1
 data[57]_constr1: - w0 - 7.3 w1 - 2.9 w2 - s_57 + 7.9 p_57_plus[0]
   + 1.6 p_57_plus[1] + 6 p_57_plus[2] <= -1
 data[57]_constr2: w0 + 7.3 w1 + 2.9 w2 - 0.1 Lambda - s_57
   + 7.9 p_57_minus[0] + 1.6 p_57_minus[1] + 6 p_57_minus[2] <= -1
 data[58]_constr1: - w0 - 6.7 w1 - 2.5 w2 - s_58 + 7.9 p_58_plus[0]
   + 2.2 p_58_plus[1] + 6.4 p_58_plus[2] <= -1
 data[58]_constr2: w0 + 6.7 w1 + 2.5 w2 - 0.1 Lambda - s_58
   + 7.9 p_58_minus[0] + 2.2 p_58_minus[1] + 6.4 p_58_minus[2] <= -1
 data[59]_constr1: - w0 - 7.2 w1 - 3.6 w2 - s_59 + 7.9 p_59_plus[0]
   + 1.7 p_59_plus[1] + 5.3 p_59_plus[2] <= -1
 data[59]_constr2: w0 + 7.2 w1 + 3.6 w2 - 0.1 Lambda - s_59
   + 7.9 p_59_minus[0] + 1.7 p_59_minus[1] + 5.3 p_59_minus[2] <= -1
 data[60]_constr1: - w0 - 6.5 w1 - 3.2 w2 - s_60 + 7.9 p_60_plus[0]
   + 2.4 p_60_plus[1] + 5.7 p_60_plus[2] <= -1
 data[60]_constr2: w0 + 6.5 w1 + 3.2 w2 - 0.1 Lambda - s_60
   + 7.9 p_60_minus[0] + 2.4 p_60_minus[1] + 5.7 p_60_minus[2] <= -1
 data[61]_constr1: - w0 - 6.4 w1 - 2.7 w2 - s_61 + 7.9 p_61_plus[0]
   + 2.5 p_61_plus[1] + 6.2 p_61_plus[2] <= -1
 data[61]_constr2: w0 + 6.4 w1 + 2.7 w2 - 0.1 Lambda - s_61
   + 7.9 p_61_minus[0] + 2.5 p_61_minus[1] + 6.2 p_61_minus[2] <= -1
 data[62]_constr1: - w0 - 6.8 w1 - 3 w2 - s_62 + 7.9 p_62_plus[0]
   + 2.1 p_62_plus[1] + 5.9 p_62_plus[2] <= -1
 data[62]_constr2: w0 + 6.8 w1 + 3 w2 - 0.1 Lambda - s_62
   + 7.9 p_62_minus[0] + 2.1 p_62_minus[1] + 5.9 p_62_minus[2] <= -1
 data[63]_constr1: - w0 - 5.7 w1 - 2.5 w2 - s_63 + 7.9 p_63_plus[0]
   + 3.2 p_63_plus[1] + 6.4 p_63_plus[2] <= -1
 data[63]_constr2: w0 + 5.7 w1 + 2.5 w2 - 0.1 Lambda - s_63
   + 7.9 p_63_minus[0] + 3.2 p_63_minus[1] + 6.4 p_63_minus[2] <= -1
 data[64]_constr1: - w0 - 5.8 w1 - 2.8 w2 - s_64 + 7.9 p_64_plus[0]
   + 3.1 p_64_plus[1] + 6.1 p_64_plus[2] <= -1
 data[64]_constr2: w0 + 5.8 w1 + 2.8 w2 - 0.1 Lambda - s_64
   + 7.9 p_64_minus[0] + 3.1 p_64_minus[1] + 6.1 p_64_minus[2] <= -1
 data[65]_constr1: - w0 - 6.4 w1 - 3.2 w2 - s_65 + 7.9 p_65_plus[0]
   + 2.5 p_65_plus[1] + 5.7 p_65_plus[2] <= -1
 data[65]_constr2: w0 + 6.4 w1 + 3.2 w2 - 0.1 Lambda - s_65
   + 7.9 p_65_minus[0] + 2.5 p_65_minus[1] + 5.7 p_65_minus[2] <= -1
 data[66]_constr1: - w0 - 6.5 w1 - 3 w2 - s_66 + 7.9 p_66_plus[0]
   + 2.4 p_66_plus[1] + 5.9 p_66_plus[2] <= -1
 data[66]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_66
   + 7.9 p_66_minus[0] + 2.4 p_66_minus[1] + 5.9 p_66_minus[2] <= -1
 data[67]_constr1: - w0 - 7.7 w1 - 3.8 w2 - s_67 + 7.9 p_67_plus[0]
   + 1.2 p_67_plus[1] + 5.1 p_67_plus[2] <= -1
 data[67]_constr2: w0 + 7.7 w1 + 3.8 w2 - 0.1 Lambda - s_67
   + 7.9 p_67_minus[0] + 1.2 p_67_minus[1] + 5.1 p_67_minus[2] <= -1
 data[68]_constr1: - w0 - 7.7 w1 - 2.6 w2 - s_68 + 7.9 p_68_plus[0]
   + 1.2 p_68_plus[1] + 6.3 p_68_plus[2] <= -1
 data[68]_constr2: w0 + 7.7 w1 + 2.6 w2 - 0.1 Lambda - s_68
   + 7.9 p_68_minus[0] + 1.2 p_68_minus[1] + 6.3 p_68_minus[2] <= -1
 data[69]_constr1: - w0 - 6 w1 - 2.2 w2 - s_69 + 7.9 p_69_plus[0]
   + 2.9 p_69_plus[1] + 6.7 p_69_plus[2] <= -1
 data[69]_constr2: w0 + 6 w1 + 2.2 w2 - 0.1 Lambda - s_69
   + 7.9 p_69_minus[0] + 2.9 p_69_minus[1] + 6.7 p_69_minus[2] <= -1
 data[70]_constr1: - w0 - 6.9 w1 - 3.2 w2 - s_70 + 7.9 p_70_plus[0]
   + 2 p_70_plus[1] + 5.7 p_70_plus[2] <= -1
 data[70]_constr2: w0 + 6.9 w1 + 3.2 w2 - 0.1 Lambda - s_70
   + 7.9 p_70_minus[0] + 2 p_70_minus[1] + 5.7 p_70_minus[2] <= -1
 data[71]_constr1: - w0 - 5.6 w1 - 2.8 w2 - s_71 + 7.9 p_71_plus[0]
   + 3.3 p_71_plus[1] + 6.1 p_71_plus[2] <= -1
 data[71]_constr2: w0 + 5.6 w1 + 2.8 w2 - 0.1 Lambda - s_71
   + 7.9 p_71_minus[0] + 3.3 p_71_minus[1] + 6.1 p_71_minus[2] <= -1
 data[72]_constr1: - w0 - 7.7 w1 - 2.8 w2 - s_72 + 7.9 p_72_plus[0]
   + 1.2 p_72_plus[1] + 6.1 p_72_plus[2] <= -1
 data[72]_constr2: w0 + 7.7 w1 + 2.8 w2 - 0.1 Lambda - s_72
   + 7.9 p_72_minus[0] + 1.2 p_72_minus[1] + 6.1 p_72_minus[2] <= -1
 data[73]_constr1: - w0 - 6.3 w1 - 2.7 w2 - s_73 + 7.9 p_73_plus[0]
   + 2.6 p_73_plus[1] + 6.2 p_73_plus[2] <= -1
 data[73]_constr2: w0 + 6.3 w1 + 2.7 w2 - 0.1 Lambda - s_73
   + 7.9 p_73_minus[0] + 2.6 p_73_minus[1] + 6.2 p_73_minus[2] <= -1
 data[74]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_74 + 7.9 p_74_plus[0]
   + 2.2 p_74_plus[1] + 5.6 p_74_plus[2] <= -1
 data[74]_constr2: w0 + 6.7 w1 + 3.3 w2 - 0.1 Lambda - s_74
   + 7.9 p_74_minus[0] + 2.2 p_74_minus[1] + 5.6 p_74_minus[2] <= -1
 data[75]_constr1: - w0 - 7.2 w1 - 3.2 w2 - s_75 + 7.9 p_75_plus[0]
   + 1.7 p_75_plus[1] + 5.7 p_75_plus[2] <= -1
 data[75]_constr2: w0 + 7.2 w1 + 3.2 w2 - 0.1 Lambda - s_75
   + 7.9 p_75_minus[0] + 1.7 p_75_minus[1] + 5.7 p_75_minus[2] <= -1
 data[76]_constr1: - w0 - 6.2 w1 - 2.8 w2 - s_76 + 7.9 p_76_plus[0]
   + 2.7 p_76_plus[1] + 6.1 p_76_plus[2] <= -1
 data[76]_constr2: w0 + 6.2 w1 + 2.8 w2 - 0.1 Lambda - s_76
   + 7.9 p_76_minus[0] + 2.7 p_76_minus[1] + 6.1 p_76_minus[2] <= -1
 data[77]_constr1: - w0 - 6.1 w1 - 3 w2 - s_77 + 7.9 p_77_plus[0]
   + 2.8 p_77_plus[1] + 5.9 p_77_plus[2] <= -1
 data[77]_constr2: w0 + 6.1 w1 + 3 w2 - 0.1 Lambda - s_77
   + 7.9 p_77_minus[0] + 2.8 p_77_minus[1] + 5.9 p_77_minus[2] <= -1
 data[78]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_78 + 7.9 p_78_plus[0]
   + 2.5 p_78_plus[1] + 6.1 p_78_plus[2] <= -1
 data[78]_constr2: w0 + 6.4 w1 + 2.8 w2 - 0.1 Lambda - s_78
   + 7.9 p_78_minus[0] + 2.5 p_78_minus[1] + 6.1 p_78_minus[2] <= -1
 data[79]_constr1: - w0 - 7.2 w1 - 3 w2 - s_79 + 7.9 p_79_plus[0]
   + 1.7 p_79_plus[1] + 5.9 p_79_plus[2] <= -1
 data[79]_constr2: w0 + 7.2 w1 + 3 w2 - 0.1 Lambda - s_79
   + 7.9 p_79_minus[0] + 1.7 p_79_minus[1] + 5.9 p_79_minus[2] <= -1
 data[80]_constr1: - w0 - 7.4 w1 - 2.8 w2 - s_80 + 7.9 p_80_plus[0]
   + 1.5 p_80_plus[1] + 6.1 p_80_plus[2] <= -1
 data[80]_constr2: w0 + 7.4 w1 + 2.8 w2 - 0.1 Lambda - s_80
   + 7.9 p_80_minus[0] + 1.5 p_80_minus[1] + 6.1 p_80_minus[2] <= -1
 data[81]_constr1: - w0 - 7.9 w1 - 3.8 w2 - s_81 + 7.9 p_81_plus[0]
   + p_81_plus[1] + 5.1 p_81_plus[2] <= -1
 data[81]_constr2: w0 + 7.9 w1 + 3.8 w2 - 0.1 Lambda - s_81
   + 7.9 p_81_minus[0] + p_81_minus[1] + 5.1 p_81_minus[2] <= -1
 data[82]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_82 + 7.9 p_82_plus[0]
   + 2.5 p_82_plus[1] + 6.1 p_82_plus[2] <= -1
 data[82]_constr2: w0 + 6.4 w1 + 2.8 w2 - 0.1 Lambda - s_82
   + 7.9 p_82_minus[0] + 2.5 p_82_minus[1] + 6.1 p_82_minus[2] <= -1
 data[83]_constr1: - w0 - 6.3 w1 - 2.8 w2 - s_83 + 7.9 p_83_plus[0]
   + 2.6 p_83_plus[1] + 6.1 p_83_plus[2] <= -1
 data[83]_constr2: w0 + 6.3 w1 + 2.8 w2 - 0.1 Lambda - s_83
   + 7.9 p_83_minus[0] + 2.6 p_83_minus[1] + 6.1 p_83_minus[2] <= -1
 data[84]_constr1: - w0 - 6.1 w1 - 2.6 w2 - s_84 + 7.9 p_84_plus[0]
   + 2.8 p_84_plus[1] + 6.3 p_84_plus[2] <= -1
 data[84]_constr2: w0 + 6.1 w1 + 2.6 w2 - 0.1 Lambda - s_84
   + 7.9 p_84_minus[0] + 2.8 p_84_minus[1] + 6.3 p_84_minus[2] <= -1
 data[85]_constr1: - w0 - 7.7 w1 - 3 w2 - s_85 + 7.9 p_85_plus[0]
   + 1.2 p_85_plus[1] + 5.9 p_85_plus[2] <= -1
 data[85]_constr2: w0 + 7.7 w1 + 3 w2 - 0.1 Lambda - s_85
   + 7.9 p_85_minus[0] + 1.2 p_85_minus[1] + 5.9 p_85_minus[2] <= -1
 data[86]_constr1: - w0 - 6.3 w1 - 3.4 w2 - s_86 + 7.9 p_86_plus[0]
   + 2.6 p_86_plus[1] + 5.5 p_86_plus[2] <= -1
 data[86]_constr2: w0 + 6.3 w1 + 3.4 w2 - 0.1 Lambda - s_86
   + 7.9 p_86_minus[0] + 2.6 p_86_minus[1] + 5.5 p_86_minus[2] <= -1
 data[87]_constr1: - w0 - 6.4 w1 - 3.1 w2 - s_87 + 7.9 p_87_plus[0]
   + 2.5 p_87_plus[1] + 5.8 p_87_plus[2] <= -1
 data[87]_constr2: w0 + 6.4 w1 + 3.1 w2 - 0.1 Lambda - s_87
   + 7.9 p_87_minus[0] + 2.5 p_87_minus[1] + 5.8 p_87_minus[2] <= -1
 data[88]_constr1: - w0 - 6 w1 - 3 w2 - s_88 + 7.9 p_88_plus[0]
   + 2.9 p_88_plus[1] + 5.9 p_88_plus[2] <= -1
 data[88]_constr2: w0 + 6 w1 + 3 w2 - 0.1 Lambda - s_88 + 7.9 p_88_minus[0]
   + 2.9 p_88_minus[1] + 5.9 p_88_minus[2] <= -1
 data[89]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_89 + 7.9 p_89_plus[0]
   + 2 p_89_plus[1] + 5.8 p_89_plus[2] <= -1
 data[89]_constr2: w0 + 6.9 w1 + 3.1 w2 - 0.1 Lambda - s_89
   + 7.9 p_89_minus[0] + 2 p_89_minus[1] + 5.8 p_89_minus[2] <= -1
 data[90]_constr1: - w0 - 6.7 w1 - 3.1 w2 - s_90 + 7.9 p_90_plus[0]
   + 2.2 p_90_plus[1] + 5.8 p_90_plus[2] <= -1
 data[90]_constr2: w0 + 6.7 w1 + 3.1 w2 - 0.1 Lambda - s_90
   + 7.9 p_90_minus[0] + 2.2 p_90_minus[1] + 5.8 p_90_minus[2] <= -1
 data[91]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_91 + 7.9 p_91_plus[0]
   + 2 p_91_plus[1] + 5.8 p_91_plus[2] <= -1
 data[91]_constr2: w0 + 6.9 w1 + 3.1 w2 - 0.1 Lambda - s_91
   + 7.9 p_91_minus[0] + 2 p_91_minus[1] + 5.8 p_91_minus[2] <= -1
 data[92]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_92 + 7.9 p_92_plus[0]
   + 3.1 p_92_plus[1] + 6.2 p_92_plus[2] <= -1
 data[92]_constr2: w0 + 5.8 w1 + 2.7 w2 - 0.1 Lambda - s_92
   + 7.9 p_92_minus[0] + 3.1 p_92_minus[1] + 6.2 p_92_minus[2] <= -1
 data[93]_constr1: - w0 - 6.8 w1 - 3.2 w2 - s_93 + 7.9 p_93_plus[0]
   + 2.1 p_93_plus[1] + 5.7 p_93_plus[2] <= -1
 data[93]_constr2: w0 + 6.8 w1 + 3.2 w2 - 0.1 Lambda - s_93
   + 7.9 p_93_minus[0] + 2.1 p_93_minus[1] + 5.7 p_93_minus[2] <= -1
 data[94]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_94 + 7.9 p_94_plus[0]
   + 2.2 p_94_plus[1] + 5.6 p_94_plus[2] <= -1
 data[94]_constr2: w0 + 6.7 w1 + 3.3 w2 - 0.1 Lambda - s_94
   + 7.9 p_94_minus[0] + 2.2 p_94_minus[1] + 5.6 p_94_minus[2] <= -1
 data[95]_constr1: - w0 - 6.7 w1 - 3 w2 - s_95 + 7.9 p_95_plus[0]
   + 2.2 p_95_plus[1] + 5.9 p_95_plus[2] <= -1
 data[95]_constr2: w0 + 6.7 w1 + 3 w2 - 0.1 Lambda - s_95
   + 7.9 p_95_minus[0] + 2.2 p_95_minus[1] + 5.9 p_95_minus[2] <= -1
 data[96]_constr1: - w0 - 6.3 w1 - 2.5 w2 - s_96 + 7.9 p_96_plus[0]
   + 2.6 p_96_plus[1] + 6.4 p_96_plus[2] <= -1
 data[96]_constr2: w0 + 6.3 w1 + 2.5 w2 - 0.1 Lambda - s_96
   + 7.9 p_96_minus[0] + 2.6 p_96_minus[1] + 6.4 p_96_minus[2] <= -1
 data[97]_constr1: - w0 - 6.5 w1 - 3 w2 - s_97 + 7.9 p_97_plus[0]
   + 2.4 p_97_plus[1] + 5.9 p_97_plus[2] <= -1
 data[97]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_97
   + 7.9 p_97_minus[0] + 2.4 p_97_minus[1] + 5.9 p_97_minus[2] <= -1
 data[98]_constr1: - w0 - 6.2 w1 - 3.4 w2 - s_98 + 7.9 p_98_plus[0]
   + 2.7 p_98_plus[1] + 5.5 p_98_plus[2] <= -1
 data[98]_constr2: w0 + 6.2 w1 + 3.4 w2 - 0.1 Lambda - s_98
   + 7.9 p_98_minus[0] + 2.7 p_98_minus[1] + 5.5 p_98_minus[2] <= -1
 data[99]_constr1: - w0 - 5.9 w1 - 3 w2 - s_99 + 7.9 p_99_plus[0]
   + 3 p_99_plus[1] + 5.9 p_99_plus[2] <= -1
 data[99]_constr2: w0 + 5.9 w1 + 3 w2 - 0.1 Lambda - s_99
   + 7.9 p_99_minus[0] + 3 p_99_minus[1] + 5.9 p_99_minus[2] <= -1
 s_0>=0: s_0 >= 0
 p_[0,0]_plus>=0: p_0_plus[0] >= 0
 p_[0,0]_minus>=0: p_0_minus[0] >= 0
 p_[0,1]_plus>=0: p_0_plus[1] >= 0
 p_[0,1]_minus>=0: p_0_minus[1] >= 0
 p_[0,2]_plus>=0: p_0_plus[2] >= 0
 p_[0,2]_minus>=0: p_0_minus[2] >= 0
 s_1>=0: s_1 >= 0
 p_[1,0]_plus>=0: p_1_plus[0] >= 0
 p_[1,0]_minus>=0: p_1_minus[0] >= 0
 p_[1,1]_plus>=0: p_1_plus[1] >= 0
 p_[1,1]_minus>=0: p_1_minus[1] >= 0
 p_[1,2]_plus>=0: p_1_plus[2] >= 0
 p_[1,2]_minus>=0: p_1_minus[2] >= 0
 s_2>=0: s_2 >= 0
 p_[2,0]_plus>=0: p_2_plus[0] >= 0
 p_[2,0]_minus>=0: p_2_minus[0] >= 0
 p_[2,1]_plus>=0: p_2_plus[1] >= 0
 p_[2,1]_minus>=0: p_2_minus[1] >= 0
 p_[2,2]_plus>=0: p_2_plus[2] >= 0
 p_[2,2]_minus>=0: p_2_minus[2] >= 0
 s_3>=0: s_3 >= 0
 p_[3,0]_plus>=0: p_3_plus[0] >= 0
 p_[3,0]_minus>=0: p_3_minus[0] >= 0
 p_[3,1]_plus>=0: p_3_plus[1] >= 0
 p_[3,1]_minus>=0: p_3_minus[1] >= 0
 p_[3,2]_plus>=0: p_3_plus[2] >= 0
 p_[3,2]_minus>=0: p_3_minus[2] >= 0
 s_4>=0: s_4 >= 0
 p_[4,0]_plus>=0: p_4_plus[0] >= 0
 p_[4,0]_minus>=0: p_4_minus[0] >= 0
 p_[4,1]_plus>=0: p_4_plus[1] >= 0
 p_[4,1]_minus>=0: p_4_minus[1] >= 0
 p_[4,2]_plus>=0: p_4_plus[2] >= 0
 p_[4,2]_minus>=0: p_4_minus[2] >= 0
 s_5>=0: s_5 >= 0
 p_[5,0]_plus>=0: p_5_plus[0] >= 0
 p_[5,0]_minus>=0: p_5_minus[0] >= 0
 p_[5,1]_plus>=0: p_5_plus[1] >= 0
 p_[5,1]_minus>=0: p_5_minus[1] >= 0
 p_[5,2]_plus>=0: p_5_plus[2] >= 0
 p_[5,2]_minus>=0: p_5_minus[2] >= 0
 s_6>=0: s_6 >= 0
 p_[6,0]_plus>=0: p_6_plus[0] >= 0
 p_[6,0]_minus>=0: p_6_minus[0] >= 0
 p_[6,1]_plus>=0: p_6_plus[1] >= 0
 p_[6,1]_minus>=0: p_6_minus[1] >= 0
 p_[6,2]_plus>=0: p_6_plus[2] >= 0
 p_[6,2]_minus>=0: p_6_minus[2] >= 0
 s_7>=0: s_7 >= 0
 p_[7,0]_plus>=0: p_7_plus[0] >= 0
 p_[7,0]_minus>=0: p_7_minus[0] >= 0
 p_[7,1]_plus>=0: p_7_plus[1] >= 0
 p_[7,1]_minus>=0: p_7_minus[1] >= 0
 p_[7,2]_plus>=0: p_7_plus[2] >= 0
 p_[7,2]_minus>=0: p_7_minus[2] >= 0
 s_8>=0: s_8 >= 0
 p_[8,0]_plus>=0: p_8_plus[0] >= 0
 p_[8,0]_minus>=0: p_8_minus[0] >= 0
 p_[8,1]_plus>=0: p_8_plus[1] >= 0
 p_[8,1]_minus>=0: p_8_minus[1] >= 0
 p_[8,2]_plus>=0: p_8_plus[2] >= 0
 p_[8,2]_minus>=0: p_8_minus[2] >= 0
 s_9>=0: s_9 >= 0
 p_[9,0]_plus>=0: p_9_plus[0] >= 0
 p_[9,0]_minus>=0: p_9_minus[0] >= 0
 p_[9,1]_plus>=0: p_9_plus[1] >= 0
 p_[9,1]_minus>=0: p_9_minus[1] >= 0
 p_[9,2]_plus>=0: p_9_plus[2] >= 0
 p_[9,2]_minus>=0: p_9_minus[2] >= 0
 s_10>=0: s_10 >= 0
 p_[10,0]_plus>=0: p_10_plus[0] >= 0
 p_[10,0]_minus>=0: p_10_minus[0] >= 0
 p_[10,1]_plus>=0: p_10_plus[1] >= 0
 p_[10,1]_minus>=0: p_10_minus[1] >= 0
 p_[10,2]_plus>=0: p_10_plus[2] >= 0
 p_[10,2]_minus>=0: p_10_minus[2] >= 0
 s_11>=0: s_11 >= 0
 p_[11,0]_plus>=0: p_11_plus[0] >= 0
 p_[11,0]_minus>=0: p_11_minus[0] >= 0
 p_[11,1]_plus>=0: p_11_plus[1] >= 0
 p_[11,1]_minus>=0: p_11_minus[1] >= 0
 p_[11,2]_plus>=0: p_11_plus[2] >= 0
 p_[11,2]_minus>=0: p_11_minus[2] >= 0
 s_12>=0: s_12 >= 0
 p_[12,0]_plus>=0: p_12_plus[0] >= 0
 p_[12,0]_minus>=0: p_12_minus[0] >= 0
 p_[12,1]_plus>=0: p_12_plus[1] >= 0
 p_[12,1]_minus>=0: p_12_minus[1] >= 0
 p_[12,2]_plus>=0: p_12_plus[2] >= 0
 p_[12,2]_minus>=0: p_12_minus[2] >= 0
 s_13>=0: s_13 >= 0
 p_[13,0]_plus>=0: p_13_plus[0] >= 0
 p_[13,0]_minus>=0: p_13_minus[0] >= 0
 p_[13,1]_plus>=0: p_13_plus[1] >= 0
 p_[13,1]_minus>=0: p_13_minus[1] >= 0
 p_[13,2]_plus>=0: p_13_plus[2] >= 0
 p_[13,2]_minus>=0: p_13_minus[2] >= 0
 s_14>=0: s_14 >= 0
 p_[14,0]_plus>=0: p_14_plus[0] >= 0
 p_[14,0]_minus>=0: p_14_minus[0] >= 0
 p_[14,1]_plus>=0: p_14_plus[1] >= 0
 p_[14,1]_minus>=0: p_14_minus[1] >= 0
 p_[14,2]_plus>=0: p_14_plus[2] >= 0
 p_[14,2]_minus>=0: p_14_minus[2] >= 0
 s_15>=0: s_15 >= 0
 p_[15,0]_plus>=0: p_15_plus[0] >= 0
 p_[15,0]_minus>=0: p_15_minus[0] >= 0
 p_[15,1]_plus>=0: p_15_plus[1] >= 0
 p_[15,1]_minus>=0: p_15_minus[1] >= 0
 p_[15,2]_plus>=0: p_15_plus[2] >= 0
 p_[15,2]_minus>=0: p_15_minus[2] >= 0
 s_16>=0: s_16 >= 0
 p_[16,0]_plus>=0: p_16_plus[0] >= 0
 p_[16,0]_minus>=0: p_16_minus[0] >= 0
 p_[16,1]_plus>=0: p_16_plus[1] >= 0
 p_[16,1]_minus>=0: p_16_minus[1] >= 0
 p_[16,2]_plus>=0: p_16_plus[2] >= 0
 p_[16,2]_minus>=0: p_16_minus[2] >= 0
 s_17>=0: s_17 >= 0
 p_[17,0]_plus>=0: p_17_plus[0] >= 0
 p_[17,0]_minus>=0: p_17_minus[0] >= 0
 p_[17,1]_plus>=0: p_17_plus[1] >= 0
 p_[17,1]_minus>=0: p_17_minus[1] >= 0
 p_[17,2]_plus>=0: p_17_plus[2] >= 0
 p_[17,2]_minus>=0: p_17_minus[2] >= 0
 s_18>=0: s_18 >= 0
 p_[18,0]_plus>=0: p_18_plus[0] >= 0
 p_[18,0]_minus>=0: p_18_minus[0] >= 0
 p_[18,1]_plus>=0: p_18_plus[1] >= 0
 p_[18,1]_minus>=0: p_18_minus[1] >= 0
 p_[18,2]_plus>=0: p_18_plus[2] >= 0
 p_[18,2]_minus>=0: p_18_minus[2] >= 0
 s_19>=0: s_19 >= 0
 p_[19,0]_plus>=0: p_19_plus[0] >= 0
 p_[19,0]_minus>=0: p_19_minus[0] >= 0
 p_[19,1]_plus>=0: p_19_plus[1] >= 0
 p_[19,1]_minus>=0: p_19_minus[1] >= 0
 p_[19,2]_plus>=0: p_19_plus[2] >= 0
 p_[19,2]_minus>=0: p_19_minus[2] >= 0
 s_20>=0: s_20 >= 0
 p_[20,0]_plus>=0: p_20_plus[0] >= 0
 p_[20,0]_minus>=0: p_20_minus[0] >= 0
 p_[20,1]_plus>=0: p_20_plus[1] >= 0
 p_[20,1]_minus>=0: p_20_minus[1] >= 0
 p_[20,2]_plus>=0: p_20_plus[2] >= 0
 p_[20,2]_minus>=0: p_20_minus[2] >= 0
 s_21>=0: s_21 >= 0
 p_[21,0]_plus>=0: p_21_plus[0] >= 0
 p_[21,0]_minus>=0: p_21_minus[0] >= 0
 p_[21,1]_plus>=0: p_21_plus[1] >= 0
 p_[21,1]_minus>=0: p_21_minus[1] >= 0
 p_[21,2]_plus>=0: p_21_plus[2] >= 0
 p_[21,2]_minus>=0: p_21_minus[2] >= 0
 s_22>=0: s_22 >= 0
 p_[22,0]_plus>=0: p_22_plus[0] >= 0
 p_[22,0]_minus>=0: p_22_minus[0] >= 0
 p_[22,1]_plus>=0: p_22_plus[1] >= 0
 p_[22,1]_minus>=0: p_22_minus[1] >= 0
 p_[22,2]_plus>=0: p_22_plus[2] >= 0
 p_[22,2]_minus>=0: p_22_minus[2] >= 0
 s_23>=0: s_23 >= 0
 p_[23,0]_plus>=0: p_23_plus[0] >= 0
 p_[23,0]_minus>=0: p_23_minus[0] >= 0
 p_[23,1]_plus>=0: p_23_plus[1] >= 0
 p_[23,1]_minus>=0: p_23_minus[1] >= 0
 p_[23,2]_plus>=0: p_23_plus[2] >= 0
 p_[23,2]_minus>=0: p_23_minus[2] >= 0
 s_24>=0: s_24 >= 0
 p_[24,0]_plus>=0: p_24_plus[0] >= 0
 p_[24,0]_minus>=0: p_24_minus[0] >= 0
 p_[24,1]_plus>=0: p_24_plus[1] >= 0
 p_[24,1]_minus>=0: p_24_minus[1] >= 0
 p_[24,2]_plus>=0: p_24_plus[2] >= 0
 p_[24,2]_minus>=0: p_24_minus[2] >= 0
 s_25>=0: s_25 >= 0
 p_[25,0]_plus>=0: p_25_plus[0] >= 0
 p_[25,0]_minus>=0: p_25_minus[0] >= 0
 p_[25,1]_plus>=0: p_25_plus[1] >= 0
 p_[25,1]_minus>=0: p_25_minus[1] >= 0
 p_[25,2]_plus>=0: p_25_plus[2] >= 0
 p_[25,2]_minus>=0: p_25_minus[2] >= 0
 s_26>=0: s_26 >= 0
 p_[26,0]_plus>=0: p_26_plus[0] >= 0
 p_[26,0]_minus>=0: p_26_minus[0] >= 0
 p_[26,1]_plus>=0: p_26_plus[1] >= 0
 p_[26,1]_minus>=0: p_26_minus[1] >= 0
 p_[26,2]_plus>=0: p_26_plus[2] >= 0
 p_[26,2]_minus>=0: p_26_minus[2] >= 0
 s_27>=0: s_27 >= 0
 p_[27,0]_plus>=0: p_27_plus[0] >= 0
 p_[27,0]_minus>=0: p_27_minus[0] >= 0
 p_[27,1]_plus>=0: p_27_plus[1] >= 0
 p_[27,1]_minus>=0: p_27_minus[1] >= 0
 p_[27,2]_plus>=0: p_27_plus[2] >= 0
 p_[27,2]_minus>=0: p_27_minus[2] >= 0
 s_28>=0: s_28 >= 0
 p_[28,0]_plus>=0: p_28_plus[0] >= 0
 p_[28,0]_minus>=0: p_28_minus[0] >= 0
 p_[28,1]_plus>=0: p_28_plus[1] >= 0
 p_[28,1]_minus>=0: p_28_minus[1] >= 0
 p_[28,2]_plus>=0: p_28_plus[2] >= 0
 p_[28,2]_minus>=0: p_28_minus[2] >= 0
 s_29>=0: s_29 >= 0
 p_[29,0]_plus>=0: p_29_plus[0] >= 0
 p_[29,0]_minus>=0: p_29_minus[0] >= 0
 p_[29,1]_plus>=0: p_29_plus[1] >= 0
 p_[29,1]_minus>=0: p_29_minus[1] >= 0
 p_[29,2]_plus>=0: p_29_plus[2] >= 0
 p_[29,2]_minus>=0: p_29_minus[2] >= 0
 s_30>=0: s_30 >= 0
 p_[30,0]_plus>=0: p_30_plus[0] >= 0
 p_[30,0]_minus>=0: p_30_minus[0] >= 0
 p_[30,1]_plus>=0: p_30_plus[1] >= 0
 p_[30,1]_minus>=0: p_30_minus[1] >= 0
 p_[30,2]_plus>=0: p_30_plus[2] >= 0
 p_[30,2]_minus>=0: p_30_minus[2] >= 0
 s_31>=0: s_31 >= 0
 p_[31,0]_plus>=0: p_31_plus[0] >= 0
 p_[31,0]_minus>=0: p_31_minus[0] >= 0
 p_[31,1]_plus>=0: p_31_plus[1] >= 0
 p_[31,1]_minus>=0: p_31_minus[1] >= 0
 p_[31,2]_plus>=0: p_31_plus[2] >= 0
 p_[31,2]_minus>=0: p_31_minus[2] >= 0
 s_32>=0: s_32 >= 0
 p_[32,0]_plus>=0: p_32_plus[0] >= 0
 p_[32,0]_minus>=0: p_32_minus[0] >= 0
 p_[32,1]_plus>=0: p_32_plus[1] >= 0
 p_[32,1]_minus>=0: p_32_minus[1] >= 0
 p_[32,2]_plus>=0: p_32_plus[2] >= 0
 p_[32,2]_minus>=0: p_32_minus[2] >= 0
 s_33>=0: s_33 >= 0
 p_[33,0]_plus>=0: p_33_plus[0] >= 0
 p_[33,0]_minus>=0: p_33_minus[0] >= 0
 p_[33,1]_plus>=0: p_33_plus[1] >= 0
 p_[33,1]_minus>=0: p_33_minus[1] >= 0
 p_[33,2]_plus>=0: p_33_plus[2] >= 0
 p_[33,2]_minus>=0: p_33_minus[2] >= 0
 s_34>=0: s_34 >= 0
 p_[34,0]_plus>=0: p_34_plus[0] >= 0
 p_[34,0]_minus>=0: p_34_minus[0] >= 0
 p_[34,1]_plus>=0: p_34_plus[1] >= 0
 p_[34,1]_minus>=0: p_34_minus[1] >= 0
 p_[34,2]_plus>=0: p_34_plus[2] >= 0
 p_[34,2]_minus>=0: p_34_minus[2] >= 0
 s_35>=0: s_35 >= 0
 p_[35,0]_plus>=0: p_35_plus[0] >= 0
 p_[35,0]_minus>=0: p_35_minus[0] >= 0
 p_[35,1]_plus>=0: p_35_plus[1] >= 0
 p_[35,1]_minus>=0: p_35_minus[1] >= 0
 p_[35,2]_plus>=0: p_35_plus[2] >= 0
 p_[35,2]_minus>=0: p_35_minus[2] >= 0
 s_36>=0: s_36 >= 0
 p_[36,0]_plus>=0: p_36_plus[0] >= 0
 p_[36,0]_minus>=0: p_36_minus[0] >= 0
 p_[36,1]_plus>=0: p_36_plus[1] >= 0
 p_[36,1]_minus>=0: p_36_minus[1] >= 0
 p_[36,2]_plus>=0: p_36_plus[2] >= 0
 p_[36,2]_minus>=0: p_36_minus[2] >= 0
 s_37>=0: s_37 >= 0
 p_[37,0]_plus>=0: p_37_plus[0] >= 0
 p_[37,0]_minus>=0: p_37_minus[0] >= 0
 p_[37,1]_plus>=0: p_37_plus[1] >= 0
 p_[37,1]_minus>=0: p_37_minus[1] >= 0
 p_[37,2]_plus>=0: p_37_plus[2] >= 0
 p_[37,2]_minus>=0: p_37_minus[2] >= 0
 s_38>=0: s_38 >= 0
 p_[38,0]_plus>=0: p_38_plus[0] >= 0
 p_[38,0]_minus>=0: p_38_minus[0] >= 0
 p_[38,1]_plus>=0: p_38_plus[1] >= 0
 p_[38,1]_minus>=0: p_38_minus[1] >= 0
 p_[38,2]_plus>=0: p_38_plus[2] >= 0
 p_[38,2]_minus>=0: p_38_minus[2] >= 0
 s_39>=0: s_39 >= 0
 p_[39,0]_plus>=0: p_39_plus[0] >= 0
 p_[39,0]_minus>=0: p_39_minus[0] >= 0
 p_[39,1]_plus>=0: p_39_plus[1] >= 0
 p_[39,1]_minus>=0: p_39_minus[1] >= 0
 p_[39,2]_plus>=0: p_39_plus[2] >= 0
 p_[39,2]_minus>=0: p_39_minus[2] >= 0
 s_40>=0: s_40 >= 0
 p_[40,0]_plus>=0: p_40_plus[0] >= 0
 p_[40,0]_minus>=0: p_40_minus[0] >= 0
 p_[40,1]_plus>=0: p_40_plus[1] >= 0
 p_[40,1]_minus>=0: p_40_minus[1] >= 0
 p_[40,2]_plus>=0: p_40_plus[2] >= 0
 p_[40,2]_minus>=0: p_40_minus[2] >= 0
 s_41>=0: s_41 >= 0
 p_[41,0]_plus>=0: p_41_plus[0] >= 0
 p_[41,0]_minus>=0: p_41_minus[0] >= 0
 p_[41,1]_plus>=0: p_41_plus[1] >= 0
 p_[41,1]_minus>=0: p_41_minus[1] >= 0
 p_[41,2]_plus>=0: p_41_plus[2] >= 0
 p_[41,2]_minus>=0: p_41_minus[2] >= 0
 s_42>=0: s_42 >= 0
 p_[42,0]_plus>=0: p_42_plus[0] >= 0
 p_[42,0]_minus>=0: p_42_minus[0] >= 0
 p_[42,1]_plus>=0: p_42_plus[1] >= 0
 p_[42,1]_minus>=0: p_42_minus[1] >= 0
 p_[42,2]_plus>=0: p_42_plus[2] >= 0
 p_[42,2]_minus>=0: p_42_minus[2] >= 0
 s_43>=0: s_43 >= 0
 p_[43,0]_plus>=0: p_43_plus[0] >= 0
 p_[43,0]_minus>=0: p_43_minus[0] >= 0
 p_[43,1]_plus>=0: p_43_plus[1] >= 0
 p_[43,1]_minus>=0: p_43_minus[1] >= 0
 p_[43,2]_plus>=0: p_43_plus[2] >= 0
 p_[43,2]_minus>=0: p_43_minus[2] >= 0
 s_44>=0: s_44 >= 0
 p_[44,0]_plus>=0: p_44_plus[0] >= 0
 p_[44,0]_minus>=0: p_44_minus[0] >= 0
 p_[44,1]_plus>=0: p_44_plus[1] >= 0
 p_[44,1]_minus>=0: p_44_minus[1] >= 0
 p_[44,2]_plus>=0: p_44_plus[2] >= 0
 p_[44,2]_minus>=0: p_44_minus[2] >= 0
 s_45>=0: s_45 >= 0
 p_[45,0]_plus>=0: p_45_plus[0] >= 0
 p_[45,0]_minus>=0: p_45_minus[0] >= 0
 p_[45,1]_plus>=0: p_45_plus[1] >= 0
 p_[45,1]_minus>=0: p_45_minus[1] >= 0
 p_[45,2]_plus>=0: p_45_plus[2] >= 0
 p_[45,2]_minus>=0: p_45_minus[2] >= 0
 s_46>=0: s_46 >= 0
 p_[46,0]_plus>=0: p_46_plus[0] >= 0
 p_[46,0]_minus>=0: p_46_minus[0] >= 0
 p_[46,1]_plus>=0: p_46_plus[1] >= 0
 p_[46,1]_minus>=0: p_46_minus[1] >= 0
 p_[46,2]_plus>=0: p_46_plus[2] >= 0
 p_[46,2]_minus>=0: p_46_minus[2] >= 0
 s_47>=0: s_47 >= 0
 p_[47,0]_plus>=0: p_47_plus[0] >= 0
 p_[47,0]_minus>=0: p_47_minus[0] >= 0
 p_[47,1]_plus>=0: p_47_plus[1] >= 0
 p_[47,1]_minus>=0: p_47_minus[1] >= 0
 p_[47,2]_plus>=0: p_47_plus[2] >= 0
 p_[47,2]_minus>=0: p_47_minus[2] >= 0
 s_48>=0: s_48 >= 0
 p_[48,0]_plus>=0: p_48_plus[0] >= 0
 p_[48,0]_minus>=0: p_48_minus[0] >= 0
 p_[48,1]_plus>=0: p_48_plus[1] >= 0
 p_[48,1]_minus>=0: p_48_minus[1] >= 0
 p_[48,2]_plus>=0: p_48_plus[2] >= 0
 p_[48,2]_minus>=0: p_48_minus[2] >= 0
 s_49>=0: s_49 >= 0
 p_[49,0]_plus>=0: p_49_plus[0] >= 0
 p_[49,0]_minus>=0: p_49_minus[0] >= 0
 p_[49,1]_plus>=0: p_49_plus[1] >= 0
 p_[49,1]_minus>=0: p_49_minus[1] >= 0
 p_[49,2]_plus>=0: p_49_plus[2] >= 0
 p_[49,2]_minus>=0: p_49_minus[2] >= 0
 s_50>=0: s_50 >= 0
 p_[50,0]_plus>=0: p_50_plus[0] >= 0
 p_[50,0]_minus>=0: p_50_minus[0] >= 0
 p_[50,1]_plus>=0: p_50_plus[1] >= 0
 p_[50,1]_minus>=0: p_50_minus[1] >= 0
 p_[50,2]_plus>=0: p_50_plus[2] >= 0
 p_[50,2]_minus>=0: p_50_minus[2] >= 0
 s_51>=0: s_51 >= 0
 p_[51,0]_plus>=0: p_51_plus[0] >= 0
 p_[51,0]_minus>=0: p_51_minus[0] >= 0
 p_[51,1]_plus>=0: p_51_plus[1] >= 0
 p_[51,1]_minus>=0: p_51_minus[1] >= 0
 p_[51,2]_plus>=0: p_51_plus[2] >= 0
 p_[51,2]_minus>=0: p_51_minus[2] >= 0
 s_52>=0: s_52 >= 0
 p_[52,0]_plus>=0: p_52_plus[0] >= 0
 p_[52,0]_minus>=0: p_52_minus[0] >= 0
 p_[52,1]_plus>=0: p_52_plus[1] >= 0
 p_[52,1]_minus>=0: p_52_minus[1] >= 0
 p_[52,2]_plus>=0: p_52_plus[2] >= 0
 p_[52,2]_minus>=0: p_52_minus[2] >= 0
 s_53>=0: s_53 >= 0
 p_[53,0]_plus>=0: p_53_plus[0] >= 0
 p_[53,0]_minus>=0: p_53_minus[0] >= 0
 p_[53,1]_plus>=0: p_53_plus[1] >= 0
 p_[53,1]_minus>=0: p_53_minus[1] >= 0
 p_[53,2]_plus>=0: p_53_plus[2] >= 0
 p_[53,2]_minus>=0: p_53_minus[2] >= 0
 s_54>=0: s_54 >= 0
 p_[54,0]_plus>=0: p_54_plus[0] >= 0
 p_[54,0]_minus>=0: p_54_minus[0] >= 0
 p_[54,1]_plus>=0: p_54_plus[1] >= 0
 p_[54,1]_minus>=0: p_54_minus[1] >= 0
 p_[54,2]_plus>=0: p_54_plus[2] >= 0
 p_[54,2]_minus>=0: p_54_minus[2] >= 0
 s_55>=0: s_55 >= 0
 p_[55,0]_plus>=0: p_55_plus[0] >= 0
 p_[55,0]_minus>=0: p_55_minus[0] >= 0
 p_[55,1]_plus>=0: p_55_plus[1] >= 0
 p_[55,1]_minus>=0: p_55_minus[1] >= 0
 p_[55,2]_plus>=0: p_55_plus[2] >= 0
 p_[55,2]_minus>=0: p_55_minus[2] >= 0
 s_56>=0: s_56 >= 0
 p_[56,0]_plus>=0: p_56_plus[0] >= 0
 p_[56,0]_minus>=0: p_56_minus[0] >= 0
 p_[56,1]_plus>=0: p_56_plus[1] >= 0
 p_[56,1]_minus>=0: p_56_minus[1] >= 0
 p_[56,2]_plus>=0: p_56_plus[2] >= 0
 p_[56,2]_minus>=0: p_56_minus[2] >= 0
 s_57>=0: s_57 >= 0
 p_[57,0]_plus>=0: p_57_plus[0] >= 0
 p_[57,0]_minus>=0: p_57_minus[0] >= 0
 p_[57,1]_plus>=0: p_57_plus[1] >= 0
 p_[57,1]_minus>=0: p_57_minus[1] >= 0
 p_[57,2]_plus>=0: p_57_plus[2] >= 0
 p_[57,2]_minus>=0: p_57_minus[2] >= 0
 s_58>=0: s_58 >= 0
 p_[58,0]_plus>=0: p_58_plus[0] >= 0
 p_[58,0]_minus>=0: p_58_minus[0] >= 0
 p_[58,1]_plus>=0: p_58_plus[1] >= 0
 p_[58,1]_minus>=0: p_58_minus[1] >= 0
 p_[58,2]_plus>=0: p_58_plus[2] >= 0
 p_[58,2]_minus>=0: p_58_minus[2] >= 0
 s_59>=0: s_59 >= 0
 p_[59,0]_plus>=0: p_59_plus[0] >= 0
 p_[59,0]_minus>=0: p_59_minus[0] >= 0
 p_[59,1]_plus>=0: p_59_plus[1] >= 0
 p_[59,1]_minus>=0: p_59_minus[1] >= 0
 p_[59,2]_plus>=0: p_59_plus[2] >= 0
 p_[59,2]_minus>=0: p_59_minus[2] >= 0
 s_60>=0: s_60 >= 0
 p_[60,0]_plus>=0: p_60_plus[0] >= 0
 p_[60,0]_minus>=0: p_60_minus[0] >= 0
 p_[60,1]_plus>=0: p_60_plus[1] >= 0
 p_[60,1]_minus>=0: p_60_minus[1] >= 0
 p_[60,2]_plus>=0: p_60_plus[2] >= 0
 p_[60,2]_minus>=0: p_60_minus[2] >= 0
 s_61>=0: s_61 >= 0
 p_[61,0]_plus>=0: p_61_plus[0] >= 0
 p_[61,0]_minus>=0: p_61_minus[0] >= 0
 p_[61,1]_plus>=0: p_61_plus[1] >= 0
 p_[61,1]_minus>=0: p_61_minus[1] >= 0
 p_[61,2]_plus>=0: p_61_plus[2] >= 0
 p_[61,2]_minus>=0: p_61_minus[2] >= 0
 s_62>=0: s_62 >= 0
 p_[62,0]_plus>=0: p_62_plus[0] >= 0
 p_[62,0]_minus>=0: p_62_minus[0] >= 0
 p_[62,1]_plus>=0: p_62_plus[1] >= 0
 p_[62,1]_minus>=0: p_62_minus[1] >= 0
 p_[62,2]_plus>=0: p_62_plus[2] >= 0
 p_[62,2]_minus>=0: p_62_minus[2] >= 0
 s_63>=0: s_63 >= 0
 p_[63,0]_plus>=0: p_63_plus[0] >= 0
 p_[63,0]_minus>=0: p_63_minus[0] >= 0
 p_[63,1]_plus>=0: p_63_plus[1] >= 0
 p_[63,1]_minus>=0: p_63_minus[1] >= 0
 p_[63,2]_plus>=0: p_63_plus[2] >= 0
 p_[63,2]_minus>=0: p_63_minus[2] >= 0
 s_64>=0: s_64 >= 0
 p_[64,0]_plus>=0: p_64_plus[0] >= 0
 p_[64,0]_minus>=0: p_64_minus[0] >= 0
 p_[64,1]_plus>=0: p_64_plus[1] >= 0
 p_[64,1]_minus>=0: p_64_minus[1] >= 0
 p_[64,2]_plus>=0: p_64_plus[2] >= 0
 p_[64,2]_minus>=0: p_64_minus[2] >= 0
 s_65>=0: s_65 >= 0
 p_[65,0]_plus>=0: p_65_plus[0] >= 0
 p_[65,0]_minus>=0: p_65_minus[0] >= 0
 p_[65,1]_plus>=0: p_65_plus[1] >= 0
 p_[65,1]_minus>=0: p_65_minus[1] >= 0
 p_[65,2]_plus>=0: p_65_plus[2] >= 0
 p_[65,2]_minus>=0: p_65_minus[2] >= 0
 s_66>=0: s_66 >= 0
 p_[66,0]_plus>=0: p_66_plus[0] >= 0
 p_[66,0]_minus>=0: p_66_minus[0] >= 0
 p_[66,1]_plus>=0: p_66_plus[1] >= 0
 p_[66,1]_minus>=0: p_66_minus[1] >= 0
 p_[66,2]_plus>=0: p_66_plus[2] >= 0
 p_[66,2]_minus>=0: p_66_minus[2] >= 0
 s_67>=0: s_67 >= 0
 p_[67,0]_plus>=0: p_67_plus[0] >= 0
 p_[67,0]_minus>=0: p_67_minus[0] >= 0
 p_[67,1]_plus>=0: p_67_plus[1] >= 0
 p_[67,1]_minus>=0: p_67_minus[1] >= 0
 p_[67,2]_plus>=0: p_67_plus[2] >= 0
 p_[67,2]_minus>=0: p_67_minus[2] >= 0
 s_68>=0: s_68 >= 0
 p_[68,0]_plus>=0: p_68_plus[0] >= 0
 p_[68,0]_minus>=0: p_68_minus[0] >= 0
 p_[68,1]_plus>=0: p_68_plus[1] >= 0
 p_[68,1]_minus>=0: p_68_minus[1] >= 0
 p_[68,2]_plus>=0: p_68_plus[2] >= 0
 p_[68,2]_minus>=0: p_68_minus[2] >= 0
 s_69>=0: s_69 >= 0
 p_[69,0]_plus>=0: p_69_plus[0] >= 0
 p_[69,0]_minus>=0: p_69_minus[0] >= 0
 p_[69,1]_plus>=0: p_69_plus[1] >= 0
 p_[69,1]_minus>=0: p_69_minus[1] >= 0
 p_[69,2]_plus>=0: p_69_plus[2] >= 0
 p_[69,2]_minus>=0: p_69_minus[2] >= 0
 s_70>=0: s_70 >= 0
 p_[70,0]_plus>=0: p_70_plus[0] >= 0
 p_[70,0]_minus>=0: p_70_minus[0] >= 0
 p_[70,1]_plus>=0: p_70_plus[1] >= 0
 p_[70,1]_minus>=0: p_70_minus[1] >= 0
 p_[70,2]_plus>=0: p_70_plus[2] >= 0
 p_[70,2]_minus>=0: p_70_minus[2] >= 0
 s_71>=0: s_71 >= 0
 p_[71,0]_plus>=0: p_71_plus[0] >= 0
 p_[71,0]_minus>=0: p_71_minus[0] >= 0
 p_[71,1]_plus>=0: p_71_plus[1] >= 0
 p_[71,1]_minus>=0: p_71_minus[1] >= 0
 p_[71,2]_plus>=0: p_71_plus[2] >= 0
 p_[71,2]_minus>=0: p_71_minus[2] >= 0
 s_72>=0: s_72 >= 0
 p_[72,0]_plus>=0: p_72_plus[0] >= 0
 p_[72,0]_minus>=0: p_72_minus[0] >= 0
 p_[72,1]_plus>=0: p_72_plus[1] >= 0
 p_[72,1]_minus>=0: p_72_minus[1] >= 0
 p_[72,2]_plus>=0: p_72_plus[2] >= 0
 p_[72,2]_minus>=0: p_72_minus[2] >= 0
 s_73>=0: s_73 >= 0
 p_[73,0]_plus>=0: p_73_plus[0] >= 0
 p_[73,0]_minus>=0: p_73_minus[0] >= 0
 p_[73,1]_plus>=0: p_73_plus[1] >= 0
 p_[73,1]_minus>=0: p_73_minus[1] >= 0
 p_[73,2]_plus>=0: p_73_plus[2] >= 0
 p_[73,2]_minus>=0: p_73_minus[2] >= 0
 s_74>=0: s_74 >= 0
 p_[74,0]_plus>=0: p_74_plus[0] >= 0
 p_[74,0]_minus>=0: p_74_minus[0] >= 0
 p_[74,1]_plus>=0: p_74_plus[1] >= 0
 p_[74,1]_minus>=0: p_74_minus[1] >= 0
 p_[74,2]_plus>=0: p_74_plus[2] >= 0
 p_[74,2]_minus>=0: p_74_minus[2] >= 0
 s_75>=0: s_75 >= 0
 p_[75,0]_plus>=0: p_75_plus[0] >= 0
 p_[75,0]_minus>=0: p_75_minus[0] >= 0
 p_[75,1]_plus>=0: p_75_plus[1] >= 0
 p_[75,1]_minus>=0: p_75_minus[1] >= 0
 p_[75,2]_plus>=0: p_75_plus[2] >= 0
 p_[75,2]_minus>=0: p_75_minus[2] >= 0
 s_76>=0: s_76 >= 0
 p_[76,0]_plus>=0: p_76_plus[0] >= 0
 p_[76,0]_minus>=0: p_76_minus[0] >= 0
 p_[76,1]_plus>=0: p_76_plus[1] >= 0
 p_[76,1]_minus>=0: p_76_minus[1] >= 0
 p_[76,2]_plus>=0: p_76_plus[2] >= 0
 p_[76,2]_minus>=0: p_76_minus[2] >= 0
 s_77>=0: s_77 >= 0
 p_[77,0]_plus>=0: p_77_plus[0] >= 0
 p_[77,0]_minus>=0: p_77_minus[0] >= 0
 p_[77,1]_plus>=0: p_77_plus[1] >= 0
 p_[77,1]_minus>=0: p_77_minus[1] >= 0
 p_[77,2]_plus>=0: p_77_plus[2] >= 0
 p_[77,2]_minus>=0: p_77_minus[2] >= 0
 s_78>=0: s_78 >= 0
 p_[78,0]_plus>=0: p_78_plus[0] >= 0
 p_[78,0]_minus>=0: p_78_minus[0] >= 0
 p_[78,1]_plus>=0: p_78_plus[1] >= 0
 p_[78,1]_minus>=0: p_78_minus[1] >= 0
 p_[78,2]_plus>=0: p_78_plus[2] >= 0
 p_[78,2]_minus>=0: p_78_minus[2] >= 0
 s_79>=0: s_79 >= 0
 p_[79,0]_plus>=0: p_79_plus[0] >= 0
 p_[79,0]_minus>=0: p_79_minus[0] >= 0
 p_[79,1]_plus>=0: p_79_plus[1] >= 0
 p_[79,1]_minus>=0: p_79_minus[1] >= 0
 p_[79,2]_plus>=0: p_79_plus[2] >= 0
 p_[79,2]_minus>=0: p_79_minus[2] >= 0
 s_80>=0: s_80 >= 0
 p_[80,0]_plus>=0: p_80_plus[0] >= 0
 p_[80,0]_minus>=0: p_80_minus[0] >= 0
 p_[80,1]_plus>=0: p_80_plus[1] >= 0
 p_[80,1]_minus>=0: p_80_minus[1] >= 0
 p_[80,2]_plus>=0: p_80_plus[2] >= 0
 p_[80,2]_minus>=0: p_80_minus[2] >= 0
 s_81>=0: s_81 >= 0
 p_[81,0]_plus>=0: p_81_plus[0] >= 0
 p_[81,0]_minus>=0: p_81_minus[0] >= 0
 p_[81,1]_plus>=0: p_81_plus[1] >= 0
 p_[81,1]_minus>=0: p_81_minus[1] >= 0
 p_[81,2]_plus>=0: p_81_plus[2] >= 0
 p_[81,2]_minus>=0: p_81_minus[2] >= 0
 s_82>=0: s_82 >= 0
 p_[82,0]_plus>=0: p_82_plus[0] >= 0
 p_[82,0]_minus>=0: p_82_minus[0] >= 0
 p_[82,1]_plus>=0: p_82_plus[1] >= 0
 p_[82,1]_minus>=0: p_82_minus[1] >= 0
 p_[82,2]_plus>=0: p_82_plus[2] >= 0
 p_[82,2]_minus>=0: p_82_minus[2] >= 0
 s_83>=0: s_83 >= 0
 p_[83,0]_plus>=0: p_83_plus[0] >= 0
 p_[83,0]_minus>=0: p_83_minus[0] >= 0
 p_[83,1]_plus>=0: p_83_plus[1] >= 0
 p_[83,1]_minus>=0: p_83_minus[1] >= 0
 p_[83,2]_plus>=0: p_83_plus[2] >= 0
 p_[83,2]_minus>=0: p_83_minus[2] >= 0
 s_84>=0: s_84 >= 0
 p_[84,0]_plus>=0: p_84_plus[0] >= 0
 p_[84,0]_minus>=0: p_84_minus[0] >= 0
 p_[84,1]_plus>=0: p_84_plus[1] >= 0
 p_[84,1]_minus>=0: p_84_minus[1] >= 0
 p_[84,2]_plus>=0: p_84_plus[2] >= 0
 p_[84,2]_minus>=0: p_84_minus[2] >= 0
 s_85>=0: s_85 >= 0
 p_[85,0]_plus>=0: p_85_plus[0] >= 0
 p_[85,0]_minus>=0: p_85_minus[0] >= 0
 p_[85,1]_plus>=0: p_85_plus[1] >= 0
 p_[85,1]_minus>=0: p_85_minus[1] >= 0
 p_[85,2]_plus>=0: p_85_plus[2] >= 0
 p_[85,2]_minus>=0: p_85_minus[2] >= 0
 s_86>=0: s_86 >= 0
 p_[86,0]_plus>=0: p_86_plus[0] >= 0
 p_[86,0]_minus>=0: p_86_minus[0] >= 0
 p_[86,1]_plus>=0: p_86_plus[1] >= 0
 p_[86,1]_minus>=0: p_86_minus[1] >= 0
 p_[86,2]_plus>=0: p_86_plus[2] >= 0
 p_[86,2]_minus>=0: p_86_minus[2] >= 0
 s_87>=0: s_87 >= 0
 p_[87,0]_plus>=0: p_87_plus[0] >= 0
 p_[87,0]_minus>=0: p_87_minus[0] >= 0
 p_[87,1]_plus>=0: p_87_plus[1] >= 0
 p_[87,1]_minus>=0: p_87_minus[1] >= 0
 p_[87,2]_plus>=0: p_87_plus[2] >= 0
 p_[87,2]_minus>=0: p_87_minus[2] >= 0
 s_88>=0: s_88 >= 0
 p_[88,0]_plus>=0: p_88_plus[0] >= 0
 p_[88,0]_minus>=0: p_88_minus[0] >= 0
 p_[88,1]_plus>=0: p_88_plus[1] >= 0
 p_[88,1]_minus>=0: p_88_minus[1] >= 0
 p_[88,2]_plus>=0: p_88_plus[2] >= 0
 p_[88,2]_minus>=0: p_88_minus[2] >= 0
 s_89>=0: s_89 >= 0
 p_[89,0]_plus>=0: p_89_plus[0] >= 0
 p_[89,0]_minus>=0: p_89_minus[0] >= 0
 p_[89,1]_plus>=0: p_89_plus[1] >= 0
 p_[89,1]_minus>=0: p_89_minus[1] >= 0
 p_[89,2]_plus>=0: p_89_plus[2] >= 0
 p_[89,2]_minus>=0: p_89_minus[2] >= 0
 s_90>=0: s_90 >= 0
 p_[90,0]_plus>=0: p_90_plus[0] >= 0
 p_[90,0]_minus>=0: p_90_minus[0] >= 0
 p_[90,1]_plus>=0: p_90_plus[1] >= 0
 p_[90,1]_minus>=0: p_90_minus[1] >= 0
 p_[90,2]_plus>=0: p_90_plus[2] >= 0
 p_[90,2]_minus>=0: p_90_minus[2] >= 0
 s_91>=0: s_91 >= 0
 p_[91,0]_plus>=0: p_91_plus[0] >= 0
 p_[91,0]_minus>=0: p_91_minus[0] >= 0
 p_[91,1]_plus>=0: p_91_plus[1] >= 0
 p_[91,1]_minus>=0: p_91_minus[1] >= 0
 p_[91,2]_plus>=0: p_91_plus[2] >= 0
 p_[91,2]_minus>=0: p_91_minus[2] >= 0
 s_92>=0: s_92 >= 0
 p_[92,0]_plus>=0: p_92_plus[0] >= 0
 p_[92,0]_minus>=0: p_92_minus[0] >= 0
 p_[92,1]_plus>=0: p_92_plus[1] >= 0
 p_[92,1]_minus>=0: p_92_minus[1] >= 0
 p_[92,2]_plus>=0: p_92_plus[2] >= 0
 p_[92,2]_minus>=0: p_92_minus[2] >= 0
 s_93>=0: s_93 >= 0
 p_[93,0]_plus>=0: p_93_plus[0] >= 0
 p_[93,0]_minus>=0: p_93_minus[0] >= 0
 p_[93,1]_plus>=0: p_93_plus[1] >= 0
 p_[93,1]_minus>=0: p_93_minus[1] >= 0
 p_[93,2]_plus>=0: p_93_plus[2] >= 0
 p_[93,2]_minus>=0: p_93_minus[2] >= 0
 s_94>=0: s_94 >= 0
 p_[94,0]_plus>=0: p_94_plus[0] >= 0
 p_[94,0]_minus>=0: p_94_minus[0] >= 0
 p_[94,1]_plus>=0: p_94_plus[1] >= 0
 p_[94,1]_minus>=0: p_94_minus[1] >= 0
 p_[94,2]_plus>=0: p_94_plus[2] >= 0
 p_[94,2]_minus>=0: p_94_minus[2] >= 0
 s_95>=0: s_95 >= 0
 p_[95,0]_plus>=0: p_95_plus[0] >= 0
 p_[95,0]_minus>=0: p_95_minus[0] >= 0
 p_[95,1]_plus>=0: p_95_plus[1] >= 0
 p_[95,1]_minus>=0: p_95_minus[1] >= 0
 p_[95,2]_plus>=0: p_95_plus[2] >= 0
 p_[95,2]_minus>=0: p_95_minus[2] >= 0
 s_96>=0: s_96 >= 0
 p_[96,0]_plus>=0: p_96_plus[0] >= 0
 p_[96,0]_minus>=0: p_96_minus[0] >= 0
 p_[96,1]_plus>=0: p_96_plus[1] >= 0
 p_[96,1]_minus>=0: p_96_minus[1] >= 0
 p_[96,2]_plus>=0: p_96_plus[2] >= 0
 p_[96,2]_minus>=0: p_96_minus[2] >= 0
 s_97>=0: s_97 >= 0
 p_[97,0]_plus>=0: p_97_plus[0] >= 0
 p_[97,0]_minus>=0: p_97_minus[0] >= 0
 p_[97,1]_plus>=0: p_97_plus[1] >= 0
 p_[97,1]_minus>=0: p_97_minus[1] >= 0
 p_[97,2]_plus>=0: p_97_plus[2] >= 0
 p_[97,2]_minus>=0: p_97_minus[2] >= 0
 s_98>=0: s_98 >= 0
 p_[98,0]_plus>=0: p_98_plus[0] >= 0
 p_[98,0]_minus>=0: p_98_minus[0] >= 0
 p_[98,1]_plus>=0: p_98_plus[1] >= 0
 p_[98,1]_minus>=0: p_98_minus[1] >= 0
 p_[98,2]_plus>=0: p_98_plus[2] >= 0
 p_[98,2]_minus>=0: p_98_minus[2] >= 0
 s_99>=0: s_99 >= 0
 p_[99,0]_plus>=0: p_99_plus[0] >= 0
 p_[99,0]_minus>=0: p_99_minus[0] >= 0
 p_[99,1]_plus>=0: p_99_plus[1] >= 0
 p_[99,1]_minus>=0: p_99_minus[1] >= 0
 p_[99,2]_plus>=0: p_99_plus[2] >= 0
 p_[99,2]_minus>=0: p_99_minus[2] >= 0
 qc0: [ w0 ^2 - 2 w0 * p_0_plus[0] + w1 ^2 - 2 w1 * p_0_plus[1] + w2 ^2
   - 2 w2 * p_0_plus[2] - Lambda ^2 + p_0_plus[0] ^2 + p_0_plus[1] ^2
   + p_0_plus[2] ^2 ] <= 0
 qc1: [ w0 ^2 + 2 w0 * p_0_minus[0] + w1 ^2 + 2 w1 * p_0_minus[1] + w2 ^2
   + 2 w2 * p_0_minus[2] - Lambda ^2 + p_0_minus[0] ^2 + p_0_minus[1] ^2
   + p_0_minus[2] ^2 ] <= 0
 qc2: [ w0 ^2 - 2 w0 * p_1_plus[0] + w1 ^2 - 2 w1 * p_1_plus[1] + w2 ^2
   - 2 w2 * p_1_plus[2] - Lambda ^2 + p_1_plus[0] ^2 + p_1_plus[1] ^2
   + p_1_plus[2] ^2 ] <= 0
 qc3: [ w0 ^2 + 2 w0 * p_1_minus[0] + w1 ^2 + 2 w1 * p_1_minus[1] + w2 ^2
   + 2 w2 * p_1_minus[2] - Lambda ^2 + p_1_minus[0] ^2 + p_1_minus[1] ^2
   + p_1_minus[2] ^2 ] <= 0
 qc4: [ w0 ^2 - 2 w0 * p_2_plus[0] + w1 ^2 - 2 w1 * p_2_plus[1] + w2 ^2
   - 2 w2 * p_2_plus[2] - Lambda ^2 + p_2_plus[0] ^2 + p_2_plus[1] ^2
   + p_2_plus[2] ^2 ] <= 0
 qc5: [ w0 ^2 + 2 w0 * p_2_minus[0] + w1 ^2 + 2 w1 * p_2_minus[1] + w2 ^2
   + 2 w2 * p_2_minus[2] - Lambda ^2 + p_2_minus[0] ^2 + p_2_minus[1] ^2
   + p_2_minus[2] ^2 ] <= 0
 qc6: [ w0 ^2 - 2 w0 * p_3_plus[0] + w1 ^2 - 2 w1 * p_3_plus[1] + w2 ^2
   - 2 w2 * p_3_plus[2] - Lambda ^2 + p_3_plus[0] ^2 + p_3_plus[1] ^2
   + p_3_plus[2] ^2 ] <= 0
 qc7: [ w0 ^2 + 2 w0 * p_3_minus[0] + w1 ^2 + 2 w1 * p_3_minus[1] + w2 ^2
   + 2 w2 * p_3_minus[2] - Lambda ^2 + p_3_minus[0] ^2 + p_3_minus[1] ^2
   + p_3_minus[2] ^2 ] <= 0
 qc8: [ w0 ^2 - 2 w0 * p_4_plus[0] + w1 ^2 - 2 w1 * p_4_plus[1] + w2 ^2
   - 2 w2 * p_4_plus[2] - Lambda ^2 + p_4_plus[0] ^2 + p_4_plus[1] ^2
   + p_4_plus[2] ^2 ] <= 0
 qc9: [ w0 ^2 + 2 w0 * p_4_minus[0] + w1 ^2 + 2 w1 * p_4_minus[1] + w2 ^2
   + 2 w2 * p_4_minus[2] - Lambda ^2 + p_4_minus[0] ^2 + p_4_minus[1] ^2
   + p_4_minus[2] ^2 ] <= 0
 qc10: [ w0 ^2 - 2 w0 * p_5_plus[0] + w1 ^2 - 2 w1 * p_5_plus[1] + w2 ^2
   - 2 w2 * p_5_plus[2] - Lambda ^2 + p_5_plus[0] ^2 + p_5_plus[1] ^2
   + p_5_plus[2] ^2 ] <= 0
 qc11: [ w0 ^2 + 2 w0 * p_5_minus[0] + w1 ^2 + 2 w1 * p_5_minus[1] + w2 ^2
   + 2 w2 * p_5_minus[2] - Lambda ^2 + p_5_minus[0] ^2 + p_5_minus[1] ^2
   + p_5_minus[2] ^2 ] <= 0
 qc12: [ w0 ^2 - 2 w0 * p_6_plus[0] + w1 ^2 - 2 w1 * p_6_plus[1] + w2 ^2
   - 2 w2 * p_6_plus[2] - Lambda ^2 + p_6_plus[0] ^2 + p_6_plus[1] ^2
   + p_6_plus[2] ^2 ] <= 0
 qc13: [ w0 ^2 + 2 w0 * p_6_minus[0] + w1 ^2 + 2 w1 * p_6_minus[1] + w2 ^2
   + 2 w2 * p_6_minus[2] - Lambda ^2 + p_6_minus[0] ^2 + p_6_minus[1] ^2
   + p_6_minus[2] ^2 ] <= 0
 qc14: [ w0 ^2 - 2 w0 * p_7_plus[0] + w1 ^2 - 2 w1 * p_7_plus[1] + w2 ^2
   - 2 w2 * p_7_plus[2] - Lambda ^2 + p_7_plus[0] ^2 + p_7_plus[1] ^2
   + p_7_plus[2] ^2 ] <= 0
 qc15: [ w0 ^2 + 2 w0 * p_7_minus[0] + w1 ^2 + 2 w1 * p_7_minus[1] + w2 ^2
   + 2 w2 * p_7_minus[2] - Lambda ^2 + p_7_minus[0] ^2 + p_7_minus[1] ^2
   + p_7_minus[2] ^2 ] <= 0
 qc16: [ w0 ^2 - 2 w0 * p_8_plus[0] + w1 ^2 - 2 w1 * p_8_plus[1] + w2 ^2
   - 2 w2 * p_8_plus[2] - Lambda ^2 + p_8_plus[0] ^2 + p_8_plus[1] ^2
   + p_8_plus[2] ^2 ] <= 0
 qc17: [ w0 ^2 + 2 w0 * p_8_minus[0] + w1 ^2 + 2 w1 * p_8_minus[1] + w2 ^2
   + 2 w2 * p_8_minus[2] - Lambda ^2 + p_8_minus[0] ^2 + p_8_minus[1] ^2
   + p_8_minus[2] ^2 ] <= 0
 qc18: [ w0 ^2 - 2 w0 * p_9_plus[0] + w1 ^2 - 2 w1 * p_9_plus[1] + w2 ^2
   - 2 w2 * p_9_plus[2] - Lambda ^2 + p_9_plus[0] ^2 + p_9_plus[1] ^2
   + p_9_plus[2] ^2 ] <= 0
 qc19: [ w0 ^2 + 2 w0 * p_9_minus[0] + w1 ^2 + 2 w1 * p_9_minus[1] + w2 ^2
   + 2 w2 * p_9_minus[2] - Lambda ^2 + p_9_minus[0] ^2 + p_9_minus[1] ^2
   + p_9_minus[2] ^2 ] <= 0
 qc20: [ w0 ^2 - 2 w0 * p_10_plus[0] + w1 ^2 - 2 w1 * p_10_plus[1] + w2 ^2
   - 2 w2 * p_10_plus[2] - Lambda ^2 + p_10_plus[0] ^2 + p_10_plus[1] ^2
   + p_10_plus[2] ^2 ] <= 0
 qc21: [ w0 ^2 + 2 w0 * p_10_minus[0] + w1 ^2 + 2 w1 * p_10_minus[1]
   + w2 ^2 + 2 w2 * p_10_minus[2] - Lambda ^2 + p_10_minus[0] ^2
   + p_10_minus[1] ^2 + p_10_minus[2] ^2 ] <= 0
 qc22: [ w0 ^2 - 2 w0 * p_11_plus[0] + w1 ^2 - 2 w1 * p_11_plus[1] + w2 ^2
   - 2 w2 * p_11_plus[2] - Lambda ^2 + p_11_plus[0] ^2 + p_11_plus[1] ^2
   + p_11_plus[2] ^2 ] <= 0
 qc23: [ w0 ^2 + 2 w0 * p_11_minus[0] + w1 ^2 + 2 w1 * p_11_minus[1]
   + w2 ^2 + 2 w2 * p_11_minus[2] - Lambda ^2 + p_11_minus[0] ^2
   + p_11_minus[1] ^2 + p_11_minus[2] ^2 ] <= 0
 qc24: [ w0 ^2 - 2 w0 * p_12_plus[0] + w1 ^2 - 2 w1 * p_12_plus[1] + w2 ^2
   - 2 w2 * p_12_plus[2] - Lambda ^2 + p_12_plus[0] ^2 + p_12_plus[1] ^2
   + p_12_plus[2] ^2 ] <= 0
 qc25: [ w0 ^2 + 2 w0 * p_12_minus[0] + w1 ^2 + 2 w1 * p_12_minus[1]
   + w2 ^2 + 2 w2 * p_12_minus[2] - Lambda ^2 + p_12_minus[0] ^2
   + p_12_minus[1] ^2 + p_12_minus[2] ^2 ] <= 0
 qc26: [ w0 ^2 - 2 w0 * p_13_plus[0] + w1 ^2 - 2 w1 * p_13_plus[1] + w2 ^2
   - 2 w2 * p_13_plus[2] - Lambda ^2 + p_13_plus[0] ^2 + p_13_plus[1] ^2
   + p_13_plus[2] ^2 ] <= 0
 qc27: [ w0 ^2 + 2 w0 * p_13_minus[0] + w1 ^2 + 2 w1 * p_13_minus[1]
   + w2 ^2 + 2 w2 * p_13_minus[2] - Lambda ^2 + p_13_minus[0] ^2
   + p_13_minus[1] ^2 + p_13_minus[2] ^2 ] <= 0
 qc28: [ w0 ^2 - 2 w0 * p_14_plus[0] + w1 ^2 - 2 w1 * p_14_plus[1] + w2 ^2
   - 2 w2 * p_14_plus[2] - Lambda ^2 + p_14_plus[0] ^2 + p_14_plus[1] ^2
   + p_14_plus[2] ^2 ] <= 0
 qc29: [ w0 ^2 + 2 w0 * p_14_minus[0] + w1 ^2 + 2 w1 * p_14_minus[1]
   + w2 ^2 + 2 w2 * p_14_minus[2] - Lambda ^2 + p_14_minus[0] ^2
   + p_14_minus[1] ^2 + p_14_minus[2] ^2 ] <= 0
 qc30: [ w0 ^2 - 2 w0 * p_15_plus[0] + w1 ^2 - 2 w1 * p_15_plus[1] + w2 ^2
   - 2 w2 * p_15_plus[2] - Lambda ^2 + p_15_plus[0] ^2 + p_15_plus[1] ^2
   + p_15_plus[2] ^2 ] <= 0
 qc31: [ w0 ^2 + 2 w0 * p_15_minus[0] + w1 ^2 + 2 w1 * p_15_minus[1]
   + w2 ^2 + 2 w2 * p_15_minus[2] - Lambda ^2 + p_15_minus[0] ^2
   + p_15_minus[1] ^2 + p_15_minus[2] ^2 ] <= 0
 qc32: [ w0 ^2 - 2 w0 * p_16_plus[0] + w1 ^2 - 2 w1 * p_16_plus[1] + w2 ^2
   - 2 w2 * p_16_plus[2] - Lambda ^2 + p_16_plus[0] ^2 + p_16_plus[1] ^2
   + p_16_plus[2] ^2 ] <= 0
 qc33: [ w0 ^2 + 2 w0 * p_16_minus[0] + w1 ^2 + 2 w1 * p_16_minus[1]
   + w2 ^2 + 2 w2 * p_16_minus[2] - Lambda ^2 + p_16_minus[0] ^2
   + p_16_minus[1] ^2 + p_16_minus[2] ^2 ] <= 0
 qc34: [ w0 ^2 - 2 w0 * p_17_plus[0] + w1 ^2 - 2 w1 * p_17_plus[1] + w2 ^2
   - 2 w2 * p_17_plus[2] - Lambda ^2 + p_17_plus[0] ^2 + p_17_plus[1] ^2
   + p_17_plus[2] ^2 ] <= 0
 qc35: [ w0 ^2 + 2 w0 * p_17_minus[0] + w1 ^2 + 2 w1 * p_17_minus[1]
   + w2 ^2 + 2 w2 * p_17_minus[2] - Lambda ^2 + p_17_minus[0] ^2
   + p_17_minus[1] ^2 + p_17_minus[2] ^2 ] <= 0
 qc36: [ w0 ^2 - 2 w0 * p_18_plus[0] + w1 ^2 - 2 w1 * p_18_plus[1] + w2 ^2
   - 2 w2 * p_18_plus[2] - Lambda ^2 + p_18_plus[0] ^2 + p_18_plus[1] ^2
   + p_18_plus[2] ^2 ] <= 0
 qc37: [ w0 ^2 + 2 w0 * p_18_minus[0] + w1 ^2 + 2 w1 * p_18_minus[1]
   + w2 ^2 + 2 w2 * p_18_minus[2] - Lambda ^2 + p_18_minus[0] ^2
   + p_18_minus[1] ^2 + p_18_minus[2] ^2 ] <= 0
 qc38: [ w0 ^2 - 2 w0 * p_19_plus[0] + w1 ^2 - 2 w1 * p_19_plus[1] + w2 ^2
   - 2 w2 * p_19_plus[2] - Lambda ^2 + p_19_plus[0] ^2 + p_19_plus[1] ^2
   + p_19_plus[2] ^2 ] <= 0
 qc39: [ w0 ^2 + 2 w0 * p_19_minus[0] + w1 ^2 + 2 w1 * p_19_minus[1]
   + w2 ^2 + 2 w2 * p_19_minus[2] - Lambda ^2 + p_19_minus[0] ^2
   + p_19_minus[1] ^2 + p_19_minus[2] ^2 ] <= 0
 qc40: [ w0 ^2 - 2 w0 * p_20_plus[0] + w1 ^2 - 2 w1 * p_20_plus[1] + w2 ^2
   - 2 w2 * p_20_plus[2] - Lambda ^2 + p_20_plus[0] ^2 + p_20_plus[1] ^2
   + p_20_plus[2] ^2 ] <= 0
 qc41: [ w0 ^2 + 2 w0 * p_20_minus[0] + w1 ^2 + 2 w1 * p_20_minus[1]
   + w2 ^2 + 2 w2 * p_20_minus[2] - Lambda ^2 + p_20_minus[0] ^2
   + p_20_minus[1] ^2 + p_20_minus[2] ^2 ] <= 0
 qc42: [ w0 ^2 - 2 w0 * p_21_plus[0] + w1 ^2 - 2 w1 * p_21_plus[1] + w2 ^2
   - 2 w2 * p_21_plus[2] - Lambda ^2 + p_21_plus[0] ^2 + p_21_plus[1] ^2
   + p_21_plus[2] ^2 ] <= 0
 qc43: [ w0 ^2 + 2 w0 * p_21_minus[0] + w1 ^2 + 2 w1 * p_21_minus[1]
   + w2 ^2 + 2 w2 * p_21_minus[2] - Lambda ^2 + p_21_minus[0] ^2
   + p_21_minus[1] ^2 + p_21_minus[2] ^2 ] <= 0
 qc44: [ w0 ^2 - 2 w0 * p_22_plus[0] + w1 ^2 - 2 w1 * p_22_plus[1] + w2 ^2
   - 2 w2 * p_22_plus[2] - Lambda ^2 + p_22_plus[0] ^2 + p_22_plus[1] ^2
   + p_22_plus[2] ^2 ] <= 0
 qc45: [ w0 ^2 + 2 w0 * p_22_minus[0] + w1 ^2 + 2 w1 * p_22_minus[1]
   + w2 ^2 + 2 w2 * p_22_minus[2] - Lambda ^2 + p_22_minus[0] ^2
   + p_22_minus[1] ^2 + p_22_minus[2] ^2 ] <= 0
 qc46: [ w0 ^2 - 2 w0 * p_23_plus[0] + w1 ^2 - 2 w1 * p_23_plus[1] + w2 ^2
   - 2 w2 * p_23_plus[2] - Lambda ^2 + p_23_plus[0] ^2 + p_23_plus[1] ^2
   + p_23_plus[2] ^2 ] <= 0
 qc47: [ w0 ^2 + 2 w0 * p_23_minus[0] + w1 ^2 + 2 w1 * p_23_minus[1]
   + w2 ^2 + 2 w2 * p_23_minus[2] - Lambda ^2 + p_23_minus[0] ^2
   + p_23_minus[1] ^2 + p_23_minus[2] ^2 ] <= 0
 qc48: [ w0 ^2 - 2 w0 * p_24_plus[0] + w1 ^2 - 2 w1 * p_24_plus[1] + w2 ^2
   - 2 w2 * p_24_plus[2] - Lambda ^2 + p_24_plus[0] ^2 + p_24_plus[1] ^2
   + p_24_plus[2] ^2 ] <= 0
 qc49: [ w0 ^2 + 2 w0 * p_24_minus[0] + w1 ^2 + 2 w1 * p_24_minus[1]
   + w2 ^2 + 2 w2 * p_24_minus[2] - Lambda ^2 + p_24_minus[0] ^2
   + p_24_minus[1] ^2 + p_24_minus[2] ^2 ] <= 0
 qc50: [ w0 ^2 - 2 w0 * p_25_plus[0] + w1 ^2 - 2 w1 * p_25_plus[1] + w2 ^2
   - 2 w2 * p_25_plus[2] - Lambda ^2 + p_25_plus[0] ^2 + p_25_plus[1] ^2
   + p_25_plus[2] ^2 ] <= 0
 qc51: [ w0 ^2 + 2 w0 * p_25_minus[0] + w1 ^2 + 2 w1 * p_25_minus[1]
   + w2 ^2 + 2 w2 * p_25_minus[2] - Lambda ^2 + p_25_minus[0] ^2
   + p_25_minus[1] ^2 + p_25_minus[2] ^2 ] <= 0
 qc52: [ w0 ^2 - 2 w0 * p_26_plus[0] + w1 ^2 - 2 w1 * p_26_plus[1] + w2 ^2
   - 2 w2 * p_26_plus[2] - Lambda ^2 + p_26_plus[0] ^2 + p_26_plus[1] ^2
   + p_26_plus[2] ^2 ] <= 0
 qc53: [ w0 ^2 + 2 w0 * p_26_minus[0] + w1 ^2 + 2 w1 * p_26_minus[1]
   + w2 ^2 + 2 w2 * p_26_minus[2] - Lambda ^2 + p_26_minus[0] ^2
   + p_26_minus[1] ^2 + p_26_minus[2] ^2 ] <= 0
 qc54: [ w0 ^2 - 2 w0 * p_27_plus[0] + w1 ^2 - 2 w1 * p_27_plus[1] + w2 ^2
   - 2 w2 * p_27_plus[2] - Lambda ^2 + p_27_plus[0] ^2 + p_27_plus[1] ^2
   + p_27_plus[2] ^2 ] <= 0
 qc55: [ w0 ^2 + 2 w0 * p_27_minus[0] + w1 ^2 + 2 w1 * p_27_minus[1]
   + w2 ^2 + 2 w2 * p_27_minus[2] - Lambda ^2 + p_27_minus[0] ^2
   + p_27_minus[1] ^2 + p_27_minus[2] ^2 ] <= 0
 qc56: [ w0 ^2 - 2 w0 * p_28_plus[0] + w1 ^2 - 2 w1 * p_28_plus[1] + w2 ^2
   - 2 w2 * p_28_plus[2] - Lambda ^2 + p_28_plus[0] ^2 + p_28_plus[1] ^2
   + p_28_plus[2] ^2 ] <= 0
 qc57: [ w0 ^2 + 2 w0 * p_28_minus[0] + w1 ^2 + 2 w1 * p_28_minus[1]
   + w2 ^2 + 2 w2 * p_28_minus[2] - Lambda ^2 + p_28_minus[0] ^2
   + p_28_minus[1] ^2 + p_28_minus[2] ^2 ] <= 0
 qc58: [ w0 ^2 - 2 w0 * p_29_plus[0] + w1 ^2 - 2 w1 * p_29_plus[1] + w2 ^2
   - 2 w2 * p_29_plus[2] - Lambda ^2 + p_29_plus[0] ^2 + p_29_plus[1] ^2
   + p_29_plus[2] ^2 ] <= 0
 qc59: [ w0 ^2 + 2 w0 * p_29_minus[0] + w1 ^2 + 2 w1 * p_29_minus[1]
   + w2 ^2 + 2 w2 * p_29_minus[2] - Lambda ^2 + p_29_minus[0] ^2
   + p_29_minus[1] ^2 + p_29_minus[2] ^2 ] <= 0
 qc60: [ w0 ^2 - 2 w0 * p_30_plus[0] + w1 ^2 - 2 w1 * p_30_plus[1] + w2 ^2
   - 2 w2 * p_30_plus[2] - Lambda ^2 + p_30_plus[0] ^2 + p_30_plus[1] ^2
   + p_30_plus[2] ^2 ] <= 0
 qc61: [ w0 ^2 + 2 w0 * p_30_minus[0] + w1 ^2 + 2 w1 * p_30_minus[1]
   + w2 ^2 + 2 w2 * p_30_minus[2] - Lambda ^2 + p_30_minus[0] ^2
   + p_30_minus[1] ^2 + p_30_minus[2] ^2 ] <= 0
 qc62: [ w0 ^2 - 2 w0 * p_31_plus[0] + w1 ^2 - 2 w1 * p_31_plus[1] + w2 ^2
   - 2 w2 * p_31_plus[2] - Lambda ^2 + p_31_plus[0] ^2 + p_31_plus[1] ^2
   + p_31_plus[2] ^2 ] <= 0
 qc63: [ w0 ^2 + 2 w0 * p_31_minus[0] + w1 ^2 + 2 w1 * p_31_minus[1]
   + w2 ^2 + 2 w2 * p_31_minus[2] - Lambda ^2 + p_31_minus[0] ^2
   + p_31_minus[1] ^2 + p_31_minus[2] ^2 ] <= 0
 qc64: [ w0 ^2 - 2 w0 * p_32_plus[0] + w1 ^2 - 2 w1 * p_32_plus[1] + w2 ^2
   - 2 w2 * p_32_plus[2] - Lambda ^2 + p_32_plus[0] ^2 + p_32_plus[1] ^2
   + p_32_plus[2] ^2 ] <= 0
 qc65: [ w0 ^2 + 2 w0 * p_32_minus[0] + w1 ^2 + 2 w1 * p_32_minus[1]
   + w2 ^2 + 2 w2 * p_32_minus[2] - Lambda ^2 + p_32_minus[0] ^2
   + p_32_minus[1] ^2 + p_32_minus[2] ^2 ] <= 0
 qc66: [ w0 ^2 - 2 w0 * p_33_plus[0] + w1 ^2 - 2 w1 * p_33_plus[1] + w2 ^2
   - 2 w2 * p_33_plus[2] - Lambda ^2 + p_33_plus[0] ^2 + p_33_plus[1] ^2
   + p_33_plus[2] ^2 ] <= 0
 qc67: [ w0 ^2 + 2 w0 * p_33_minus[0] + w1 ^2 + 2 w1 * p_33_minus[1]
   + w2 ^2 + 2 w2 * p_33_minus[2] - Lambda ^2 + p_33_minus[0] ^2
   + p_33_minus[1] ^2 + p_33_minus[2] ^2 ] <= 0
 qc68: [ w0 ^2 - 2 w0 * p_34_plus[0] + w1 ^2 - 2 w1 * p_34_plus[1] + w2 ^2
   - 2 w2 * p_34_plus[2] - Lambda ^2 + p_34_plus[0] ^2 + p_34_plus[1] ^2
   + p_34_plus[2] ^2 ] <= 0
 qc69: [ w0 ^2 + 2 w0 * p_34_minus[0] + w1 ^2 + 2 w1 * p_34_minus[1]
   + w2 ^2 + 2 w2 * p_34_minus[2] - Lambda ^2 + p_34_minus[0] ^2
   + p_34_minus[1] ^2 + p_34_minus[2] ^2 ] <= 0
 qc70: [ w0 ^2 - 2 w0 * p_35_plus[0] + w1 ^2 - 2 w1 * p_35_plus[1] + w2 ^2
   - 2 w2 * p_35_plus[2] - Lambda ^2 + p_35_plus[0] ^2 + p_35_plus[1] ^2
   + p_35_plus[2] ^2 ] <= 0
 qc71: [ w0 ^2 + 2 w0 * p_35_minus[0] + w1 ^2 + 2 w1 * p_35_minus[1]
   + w2 ^2 + 2 w2 * p_35_minus[2] - Lambda ^2 + p_35_minus[0] ^2
   + p_35_minus[1] ^2 + p_35_minus[2] ^2 ] <= 0
 qc72: [ w0 ^2 - 2 w0 * p_36_plus[0] + w1 ^2 - 2 w1 * p_36_plus[1] + w2 ^2
   - 2 w2 * p_36_plus[2] - Lambda ^2 + p_36_plus[0] ^2 + p_36_plus[1] ^2
   + p_36_plus[2] ^2 ] <= 0
 qc73: [ w0 ^2 + 2 w0 * p_36_minus[0] + w1 ^2 + 2 w1 * p_36_minus[1]
   + w2 ^2 + 2 w2 * p_36_minus[2] - Lambda ^2 + p_36_minus[0] ^2
   + p_36_minus[1] ^2 + p_36_minus[2] ^2 ] <= 0
 qc74: [ w0 ^2 - 2 w0 * p_37_plus[0] + w1 ^2 - 2 w1 * p_37_plus[1] + w2 ^2
   - 2 w2 * p_37_plus[2] - Lambda ^2 + p_37_plus[0] ^2 + p_37_plus[1] ^2
   + p_37_plus[2] ^2 ] <= 0
 qc75: [ w0 ^2 + 2 w0 * p_37_minus[0] + w1 ^2 + 2 w1 * p_37_minus[1]
   + w2 ^2 + 2 w2 * p_37_minus[2] - Lambda ^2 + p_37_minus[0] ^2
   + p_37_minus[1] ^2 + p_37_minus[2] ^2 ] <= 0
 qc76: [ w0 ^2 - 2 w0 * p_38_plus[0] + w1 ^2 - 2 w1 * p_38_plus[1] + w2 ^2
   - 2 w2 * p_38_plus[2] - Lambda ^2 + p_38_plus[0] ^2 + p_38_plus[1] ^2
   + p_38_plus[2] ^2 ] <= 0
 qc77: [ w0 ^2 + 2 w0 * p_38_minus[0] + w1 ^2 + 2 w1 * p_38_minus[1]
   + w2 ^2 + 2 w2 * p_38_minus[2] - Lambda ^2 + p_38_minus[0] ^2
   + p_38_minus[1] ^2 + p_38_minus[2] ^2 ] <= 0
 qc78: [ w0 ^2 - 2 w0 * p_39_plus[0] + w1 ^2 - 2 w1 * p_39_plus[1] + w2 ^2
   - 2 w2 * p_39_plus[2] - Lambda ^2 + p_39_plus[0] ^2 + p_39_plus[1] ^2
   + p_39_plus[2] ^2 ] <= 0
 qc79: [ w0 ^2 + 2 w0 * p_39_minus[0] + w1 ^2 + 2 w1 * p_39_minus[1]
   + w2 ^2 + 2 w2 * p_39_minus[2] - Lambda ^2 + p_39_minus[0] ^2
   + p_39_minus[1] ^2 + p_39_minus[2] ^2 ] <= 0
 qc80: [ w0 ^2 - 2 w0 * p_40_plus[0] + w1 ^2 - 2 w1 * p_40_plus[1] + w2 ^2
   - 2 w2 * p_40_plus[2] - Lambda ^2 + p_40_plus[0] ^2 + p_40_plus[1] ^2
   + p_40_plus[2] ^2 ] <= 0
 qc81: [ w0 ^2 + 2 w0 * p_40_minus[0] + w1 ^2 + 2 w1 * p_40_minus[1]
   + w2 ^2 + 2 w2 * p_40_minus[2] - Lambda ^2 + p_40_minus[0] ^2
   + p_40_minus[1] ^2 + p_40_minus[2] ^2 ] <= 0
 qc82: [ w0 ^2 - 2 w0 * p_41_plus[0] + w1 ^2 - 2 w1 * p_41_plus[1] + w2 ^2
   - 2 w2 * p_41_plus[2] - Lambda ^2 + p_41_plus[0] ^2 + p_41_plus[1] ^2
   + p_41_plus[2] ^2 ] <= 0
 qc83: [ w0 ^2 + 2 w0 * p_41_minus[0] + w1 ^2 + 2 w1 * p_41_minus[1]
   + w2 ^2 + 2 w2 * p_41_minus[2] - Lambda ^2 + p_41_minus[0] ^2
   + p_41_minus[1] ^2 + p_41_minus[2] ^2 ] <= 0
 qc84: [ w0 ^2 - 2 w0 * p_42_plus[0] + w1 ^2 - 2 w1 * p_42_plus[1] + w2 ^2
   - 2 w2 * p_42_plus[2] - Lambda ^2 + p_42_plus[0] ^2 + p_42_plus[1] ^2
   + p_42_plus[2] ^2 ] <= 0
 qc85: [ w0 ^2 + 2 w0 * p_42_minus[0] + w1 ^2 + 2 w1 * p_42_minus[1]
   + w2 ^2 + 2 w2 * p_42_minus[2] - Lambda ^2 + p_42_minus[0] ^2
   + p_42_minus[1] ^2 + p_42_minus[2] ^2 ] <= 0
 qc86: [ w0 ^2 - 2 w0 * p_43_plus[0] + w1 ^2 - 2 w1 * p_43_plus[1] + w2 ^2
   - 2 w2 * p_43_plus[2] - Lambda ^2 + p_43_plus[0] ^2 + p_43_plus[1] ^2
   + p_43_plus[2] ^2 ] <= 0
 qc87: [ w0 ^2 + 2 w0 * p_43_minus[0] + w1 ^2 + 2 w1 * p_43_minus[1]
   + w2 ^2 + 2 w2 * p_43_minus[2] - Lambda ^2 + p_43_minus[0] ^2
   + p_43_minus[1] ^2 + p_43_minus[2] ^2 ] <= 0
 qc88: [ w0 ^2 - 2 w0 * p_44_plus[0] + w1 ^2 - 2 w1 * p_44_plus[1] + w2 ^2
   - 2 w2 * p_44_plus[2] - Lambda ^2 + p_44_plus[0] ^2 + p_44_plus[1] ^2
   + p_44_plus[2] ^2 ] <= 0
 qc89: [ w0 ^2 + 2 w0 * p_44_minus[0] + w1 ^2 + 2 w1 * p_44_minus[1]
   + w2 ^2 + 2 w2 * p_44_minus[2] - Lambda ^2 + p_44_minus[0] ^2
   + p_44_minus[1] ^2 + p_44_minus[2] ^2 ] <= 0
 qc90: [ w0 ^2 - 2 w0 * p_45_plus[0] + w1 ^2 - 2 w1 * p_45_plus[1] + w2 ^2
   - 2 w2 * p_45_plus[2] - Lambda ^2 + p_45_plus[0] ^2 + p_45_plus[1] ^2
   + p_45_plus[2] ^2 ] <= 0
 qc91: [ w0 ^2 + 2 w0 * p_45_minus[0] + w1 ^2 + 2 w1 * p_45_minus[1]
   + w2 ^2 + 2 w2 * p_45_minus[2] - Lambda ^2 + p_45_minus[0] ^2
   + p_45_minus[1] ^2 + p_45_minus[2] ^2 ] <= 0
 qc92: [ w0 ^2 - 2 w0 * p_46_plus[0] + w1 ^2 - 2 w1 * p_46_plus[1] + w2 ^2
   - 2 w2 * p_46_plus[2] - Lambda ^2 + p_46_plus[0] ^2 + p_46_plus[1] ^2
   + p_46_plus[2] ^2 ] <= 0
 qc93: [ w0 ^2 + 2 w0 * p_46_minus[0] + w1 ^2 + 2 w1 * p_46_minus[1]
   + w2 ^2 + 2 w2 * p_46_minus[2] - Lambda ^2 + p_46_minus[0] ^2
   + p_46_minus[1] ^2 + p_46_minus[2] ^2 ] <= 0
 qc94: [ w0 ^2 - 2 w0 * p_47_plus[0] + w1 ^2 - 2 w1 * p_47_plus[1] + w2 ^2
   - 2 w2 * p_47_plus[2] - Lambda ^2 + p_47_plus[0] ^2 + p_47_plus[1] ^2
   + p_47_plus[2] ^2 ] <= 0
 qc95: [ w0 ^2 + 2 w0 * p_47_minus[0] + w1 ^2 + 2 w1 * p_47_minus[1]
   + w2 ^2 + 2 w2 * p_47_minus[2] - Lambda ^2 + p_47_minus[0] ^2
   + p_47_minus[1] ^2 + p_47_minus[2] ^2 ] <= 0
 qc96: [ w0 ^2 - 2 w0 * p_48_plus[0] + w1 ^2 - 2 w1 * p_48_plus[1] + w2 ^2
   - 2 w2 * p_48_plus[2] - Lambda ^2 + p_48_plus[0] ^2 + p_48_plus[1] ^2
   + p_48_plus[2] ^2 ] <= 0
 qc97: [ w0 ^2 + 2 w0 * p_48_minus[0] + w1 ^2 + 2 w1 * p_48_minus[1]
   + w2 ^2 + 2 w2 * p_48_minus[2] - Lambda ^2 + p_48_minus[0] ^2
   + p_48_minus[1] ^2 + p_48_minus[2] ^2 ] <= 0
 qc98: [ w0 ^2 - 2 w0 * p_49_plus[0] + w1 ^2 - 2 w1 * p_49_plus[1] + w2 ^2
   - 2 w2 * p_49_plus[2] - Lambda ^2 + p_49_plus[0] ^2 + p_49_plus[1] ^2
   + p_49_plus[2] ^2 ] <= 0
 qc99: [ w0 ^2 + 2 w0 * p_49_minus[0] + w1 ^2 + 2 w1 * p_49_minus[1]
   + w2 ^2 + 2 w2 * p_49_minus[2] - Lambda ^2 + p_49_minus[0] ^2
   + p_49_minus[1] ^2 + p_49_minus[2] ^2 ] <= 0
 qc100: [ w0 ^2 + 2 w0 * p_50_plus[0] + w1 ^2 + 2 w1 * p_50_plus[1] + w2 ^2
   + 2 w2 * p_50_plus[2] - Lambda ^2 + p_50_plus[0] ^2 + p_50_plus[1] ^2
   + p_50_plus[2] ^2 ] <= 0
 qc101: [ w0 ^2 - 2 w0 * p_50_minus[0] + w1 ^2 - 2 w1 * p_50_minus[1]
   + w2 ^2 - 2 w2 * p_50_minus[2] - Lambda ^2 + p_50_minus[0] ^2
   + p_50_minus[1] ^2 + p_50_minus[2] ^2 ] <= 0
 qc102: [ w0 ^2 + 2 w0 * p_51_plus[0] + w1 ^2 + 2 w1 * p_51_plus[1] + w2 ^2
   + 2 w2 * p_51_plus[2] - Lambda ^2 + p_51_plus[0] ^2 + p_51_plus[1] ^2
   + p_51_plus[2] ^2 ] <= 0
 qc103: [ w0 ^2 - 2 w0 * p_51_minus[0] + w1 ^2 - 2 w1 * p_51_minus[1]
   + w2 ^2 - 2 w2 * p_51_minus[2] - Lambda ^2 + p_51_minus[0] ^2
   + p_51_minus[1] ^2 + p_51_minus[2] ^2 ] <= 0
 qc104: [ w0 ^2 + 2 w0 * p_52_plus[0] + w1 ^2 + 2 w1 * p_52_plus[1] + w2 ^2
   + 2 w2 * p_52_plus[2] - Lambda ^2 + p_52_plus[0] ^2 + p_52_plus[1] ^2
   + p_52_plus[2] ^2 ] <= 0
 qc105: [ w0 ^2 - 2 w0 * p_52_minus[0] + w1 ^2 - 2 w1 * p_52_minus[1]
   + w2 ^2 - 2 w2 * p_52_minus[2] - Lambda ^2 + p_52_minus[0] ^2
   + p_52_minus[1] ^2 + p_52_minus[2] ^2 ] <= 0
 qc106: [ w0 ^2 + 2 w0 * p_53_plus[0] + w1 ^2 + 2 w1 * p_53_plus[1] + w2 ^2
   + 2 w2 * p_53_plus[2] - Lambda ^2 + p_53_plus[0] ^2 + p_53_plus[1] ^2
   + p_53_plus[2] ^2 ] <= 0
 qc107: [ w0 ^2 - 2 w0 * p_53_minus[0] + w1 ^2 - 2 w1 * p_53_minus[1]
   + w2 ^2 - 2 w2 * p_53_minus[2] - Lambda ^2 + p_53_minus[0] ^2
   + p_53_minus[1] ^2 + p_53_minus[2] ^2 ] <= 0
 qc108: [ w0 ^2 + 2 w0 * p_54_plus[0] + w1 ^2 + 2 w1 * p_54_plus[1] + w2 ^2
   + 2 w2 * p_54_plus[2] - Lambda ^2 + p_54_plus[0] ^2 + p_54_plus[1] ^2
   + p_54_plus[2] ^2 ] <= 0
 qc109: [ w0 ^2 - 2 w0 * p_54_minus[0] + w1 ^2 - 2 w1 * p_54_minus[1]
   + w2 ^2 - 2 w2 * p_54_minus[2] - Lambda ^2 + p_54_minus[0] ^2
   + p_54_minus[1] ^2 + p_54_minus[2] ^2 ] <= 0
 qc110: [ w0 ^2 + 2 w0 * p_55_plus[0] + w1 ^2 + 2 w1 * p_55_plus[1] + w2 ^2
   + 2 w2 * p_55_plus[2] - Lambda ^2 + p_55_plus[0] ^2 + p_55_plus[1] ^2
   + p_55_plus[2] ^2 ] <= 0
 qc111: [ w0 ^2 - 2 w0 * p_55_minus[0] + w1 ^2 - 2 w1 * p_55_minus[1]
   + w2 ^2 - 2 w2 * p_55_minus[2] - Lambda ^2 + p_55_minus[0] ^2
   + p_55_minus[1] ^2 + p_55_minus[2] ^2 ] <= 0
 qc112: [ w0 ^2 + 2 w0 * p_56_plus[0] + w1 ^2 + 2 w1 * p_56_plus[1] + w2 ^2
   + 2 w2 * p_56_plus[2] - Lambda ^2 + p_56_plus[0] ^2 + p_56_plus[1] ^2
   + p_56_plus[2] ^2 ] <= 0
 qc113: [ w0 ^2 - 2 w0 * p_56_minus[0] + w1 ^2 - 2 w1 * p_56_minus[1]
   + w2 ^2 - 2 w2 * p_56_minus[2] - Lambda ^2 + p_56_minus[0] ^2
   + p_56_minus[1] ^2 + p_56_minus[2] ^2 ] <= 0
 qc114: [ w0 ^2 + 2 w0 * p_57_plus[0] + w1 ^2 + 2 w1 * p_57_plus[1] + w2 ^2
   + 2 w2 * p_57_plus[2] - Lambda ^2 + p_57_plus[0] ^2 + p_57_plus[1] ^2
   + p_57_plus[2] ^2 ] <= 0
 qc115: [ w0 ^2 - 2 w0 * p_57_minus[0] + w1 ^2 - 2 w1 * p_57_minus[1]
   + w2 ^2 - 2 w2 * p_57_minus[2] - Lambda ^2 + p_57_minus[0] ^2
   + p_57_minus[1] ^2 + p_57_minus[2] ^2 ] <= 0
 qc116: [ w0 ^2 + 2 w0 * p_58_plus[0] + w1 ^2 + 2 w1 * p_58_plus[1] + w2 ^2
   + 2 w2 * p_58_plus[2] - Lambda ^2 + p_58_plus[0] ^2 + p_58_plus[1] ^2
   + p_58_plus[2] ^2 ] <= 0
 qc117: [ w0 ^2 - 2 w0 * p_58_minus[0] + w1 ^2 - 2 w1 * p_58_minus[1]
   + w2 ^2 - 2 w2 * p_58_minus[2] - Lambda ^2 + p_58_minus[0] ^2
   + p_58_minus[1] ^2 + p_58_minus[2] ^2 ] <= 0
 qc118: [ w0 ^2 + 2 w0 * p_59_plus[0] + w1 ^2 + 2 w1 * p_59_plus[1] + w2 ^2
   + 2 w2 * p_59_plus[2] - Lambda ^2 + p_59_plus[0] ^2 + p_59_plus[1] ^2
   + p_59_plus[2] ^2 ] <= 0
 qc119: [ w0 ^2 - 2 w0 * p_59_minus[0] + w1 ^2 - 2 w1 * p_59_minus[1]
   + w2 ^2 - 2 w2 * p_59_minus[2] - Lambda ^2 + p_59_minus[0] ^2
   + p_59_minus[1] ^2 + p_59_minus[2] ^2 ] <= 0
 qc120: [ w0 ^2 + 2 w0 * p_60_plus[0] + w1 ^2 + 2 w1 * p_60_plus[1] + w2 ^2
   + 2 w2 * p_60_plus[2] - Lambda ^2 + p_60_plus[0] ^2 + p_60_plus[1] ^2
   + p_60_plus[2] ^2 ] <= 0
 qc121: [ w0 ^2 - 2 w0 * p_60_minus[0] + w1 ^2 - 2 w1 * p_60_minus[1]
   + w2 ^2 - 2 w2 * p_60_minus[2] - Lambda ^2 + p_60_minus[0] ^2
   + p_60_minus[1] ^2 + p_60_minus[2] ^2 ] <= 0
 qc122: [ w0 ^2 + 2 w0 * p_61_plus[0] + w1 ^2 + 2 w1 * p_61_plus[1] + w2 ^2
   + 2 w2 * p_61_plus[2] - Lambda ^2 + p_61_plus[0] ^2 + p_61_plus[1] ^2
   + p_61_plus[2] ^2 ] <= 0
 qc123: [ w0 ^2 - 2 w0 * p_61_minus[0] + w1 ^2 - 2 w1 * p_61_minus[1]
   + w2 ^2 - 2 w2 * p_61_minus[2] - Lambda ^2 + p_61_minus[0] ^2
   + p_61_minus[1] ^2 + p_61_minus[2] ^2 ] <= 0
 qc124: [ w0 ^2 + 2 w0 * p_62_plus[0] + w1 ^2 + 2 w1 * p_62_plus[1] + w2 ^2
   + 2 w2 * p_62_plus[2] - Lambda ^2 + p_62_plus[0] ^2 + p_62_plus[1] ^2
   + p_62_plus[2] ^2 ] <= 0
 qc125: [ w0 ^2 - 2 w0 * p_62_minus[0] + w1 ^2 - 2 w1 * p_62_minus[1]
   + w2 ^2 - 2 w2 * p_62_minus[2] - Lambda ^2 + p_62_minus[0] ^2
   + p_62_minus[1] ^2 + p_62_minus[2] ^2 ] <= 0
 qc126: [ w0 ^2 + 2 w0 * p_63_plus[0] + w1 ^2 + 2 w1 * p_63_plus[1] + w2 ^2
   + 2 w2 * p_63_plus[2] - Lambda ^2 + p_63_plus[0] ^2 + p_63_plus[1] ^2
   + p_63_plus[2] ^2 ] <= 0
 qc127: [ w0 ^2 - 2 w0 * p_63_minus[0] + w1 ^2 - 2 w1 * p_63_minus[1]
   + w2 ^2 - 2 w2 * p_63_minus[2] - Lambda ^2 + p_63_minus[0] ^2
   + p_63_minus[1] ^2 + p_63_minus[2] ^2 ] <= 0
 qc128: [ w0 ^2 + 2 w0 * p_64_plus[0] + w1 ^2 + 2 w1 * p_64_plus[1] + w2 ^2
   + 2 w2 * p_64_plus[2] - Lambda ^2 + p_64_plus[0] ^2 + p_64_plus[1] ^2
   + p_64_plus[2] ^2 ] <= 0
 qc129: [ w0 ^2 - 2 w0 * p_64_minus[0] + w1 ^2 - 2 w1 * p_64_minus[1]
   + w2 ^2 - 2 w2 * p_64_minus[2] - Lambda ^2 + p_64_minus[0] ^2
   + p_64_minus[1] ^2 + p_64_minus[2] ^2 ] <= 0
 qc130: [ w0 ^2 + 2 w0 * p_65_plus[0] + w1 ^2 + 2 w1 * p_65_plus[1] + w2 ^2
   + 2 w2 * p_65_plus[2] - Lambda ^2 + p_65_plus[0] ^2 + p_65_plus[1] ^2
   + p_65_plus[2] ^2 ] <= 0
 qc131: [ w0 ^2 - 2 w0 * p_65_minus[0] + w1 ^2 - 2 w1 * p_65_minus[1]
   + w2 ^2 - 2 w2 * p_65_minus[2] - Lambda ^2 + p_65_minus[0] ^2
   + p_65_minus[1] ^2 + p_65_minus[2] ^2 ] <= 0
 qc132: [ w0 ^2 + 2 w0 * p_66_plus[0] + w1 ^2 + 2 w1 * p_66_plus[1] + w2 ^2
   + 2 w2 * p_66_plus[2] - Lambda ^2 + p_66_plus[0] ^2 + p_66_plus[1] ^2
   + p_66_plus[2] ^2 ] <= 0
 qc133: [ w0 ^2 - 2 w0 * p_66_minus[0] + w1 ^2 - 2 w1 * p_66_minus[1]
   + w2 ^2 - 2 w2 * p_66_minus[2] - Lambda ^2 + p_66_minus[0] ^2
   + p_66_minus[1] ^2 + p_66_minus[2] ^2 ] <= 0
 qc134: [ w0 ^2 + 2 w0 * p_67_plus[0] + w1 ^2 + 2 w1 * p_67_plus[1] + w2 ^2
   + 2 w2 * p_67_plus[2] - Lambda ^2 + p_67_plus[0] ^2 + p_67_plus[1] ^2
   + p_67_plus[2] ^2 ] <= 0
 qc135: [ w0 ^2 - 2 w0 * p_67_minus[0] + w1 ^2 - 2 w1 * p_67_minus[1]
   + w2 ^2 - 2 w2 * p_67_minus[2] - Lambda ^2 + p_67_minus[0] ^2
   + p_67_minus[1] ^2 + p_67_minus[2] ^2 ] <= 0
 qc136: [ w0 ^2 + 2 w0 * p_68_plus[0] + w1 ^2 + 2 w1 * p_68_plus[1] + w2 ^2
   + 2 w2 * p_68_plus[2] - Lambda ^2 + p_68_plus[0] ^2 + p_68_plus[1] ^2
   + p_68_plus[2] ^2 ] <= 0
 qc137: [ w0 ^2 - 2 w0 * p_68_minus[0] + w1 ^2 - 2 w1 * p_68_minus[1]
   + w2 ^2 - 2 w2 * p_68_minus[2] - Lambda ^2 + p_68_minus[0] ^2
   + p_68_minus[1] ^2 + p_68_minus[2] ^2 ] <= 0
 qc138: [ w0 ^2 + 2 w0 * p_69_plus[0] + w1 ^2 + 2 w1 * p_69_plus[1] + w2 ^2
   + 2 w2 * p_69_plus[2] - Lambda ^2 + p_69_plus[0] ^2 + p_69_plus[1] ^2
   + p_69_plus[2] ^2 ] <= 0
 qc139: [ w0 ^2 - 2 w0 * p_69_minus[0] + w1 ^2 - 2 w1 * p_69_minus[1]
   + w2 ^2 - 2 w2 * p_69_minus[2] - Lambda ^2 + p_69_minus[0] ^2
   + p_69_minus[1] ^2 + p_69_minus[2] ^2 ] <= 0
 qc140: [ w0 ^2 + 2 w0 * p_70_plus[0] + w1 ^2 + 2 w1 * p_70_plus[1] + w2 ^2
   + 2 w2 * p_70_plus[2] - Lambda ^2 + p_70_plus[0] ^2 + p_70_plus[1] ^2
   + p_70_plus[2] ^2 ] <= 0
 qc141: [ w0 ^2 - 2 w0 * p_70_minus[0] + w1 ^2 - 2 w1 * p_70_minus[1]
   + w2 ^2 - 2 w2 * p_70_minus[2] - Lambda ^2 + p_70_minus[0] ^2
   + p_70_minus[1] ^2 + p_70_minus[2] ^2 ] <= 0
 qc142: [ w0 ^2 + 2 w0 * p_71_plus[0] + w1 ^2 + 2 w1 * p_71_plus[1] + w2 ^2
   + 2 w2 * p_71_plus[2] - Lambda ^2 + p_71_plus[0] ^2 + p_71_plus[1] ^2
   + p_71_plus[2] ^2 ] <= 0
 qc143: [ w0 ^2 - 2 w0 * p_71_minus[0] + w1 ^2 - 2 w1 * p_71_minus[1]
   + w2 ^2 - 2 w2 * p_71_minus[2] - Lambda ^2 + p_71_minus[0] ^2
   + p_71_minus[1] ^2 + p_71_minus[2] ^2 ] <= 0
 qc144: [ w0 ^2 + 2 w0 * p_72_plus[0] + w1 ^2 + 2 w1 * p_72_plus[1] + w2 ^2
   + 2 w2 * p_72_plus[2] - Lambda ^2 + p_72_plus[0] ^2 + p_72_plus[1] ^2
   + p_72_plus[2] ^2 ] <= 0
 qc145: [ w0 ^2 - 2 w0 * p_72_minus[0] + w1 ^2 - 2 w1 * p_72_minus[1]
   + w2 ^2 - 2 w2 * p_72_minus[2] - Lambda ^2 + p_72_minus[0] ^2
   + p_72_minus[1] ^2 + p_72_minus[2] ^2 ] <= 0
 qc146: [ w0 ^2 + 2 w0 * p_73_plus[0] + w1 ^2 + 2 w1 * p_73_plus[1] + w2 ^2
   + 2 w2 * p_73_plus[2] - Lambda ^2 + p_73_plus[0] ^2 + p_73_plus[1] ^2
   + p_73_plus[2] ^2 ] <= 0
 qc147: [ w0 ^2 - 2 w0 * p_73_minus[0] + w1 ^2 - 2 w1 * p_73_minus[1]
   + w2 ^2 - 2 w2 * p_73_minus[2] - Lambda ^2 + p_73_minus[0] ^2
   + p_73_minus[1] ^2 + p_73_minus[2] ^2 ] <= 0
 qc148: [ w0 ^2 + 2 w0 * p_74_plus[0] + w1 ^2 + 2 w1 * p_74_plus[1] + w2 ^2
   + 2 w2 * p_74_plus[2] - Lambda ^2 + p_74_plus[0] ^2 + p_74_plus[1] ^2
   + p_74_plus[2] ^2 ] <= 0
 qc149: [ w0 ^2 - 2 w0 * p_74_minus[0] + w1 ^2 - 2 w1 * p_74_minus[1]
   + w2 ^2 - 2 w2 * p_74_minus[2] - Lambda ^2 + p_74_minus[0] ^2
   + p_74_minus[1] ^2 + p_74_minus[2] ^2 ] <= 0
 qc150: [ w0 ^2 + 2 w0 * p_75_plus[0] + w1 ^2 + 2 w1 * p_75_plus[1] + w2 ^2
   + 2 w2 * p_75_plus[2] - Lambda ^2 + p_75_plus[0] ^2 + p_75_plus[1] ^2
   + p_75_plus[2] ^2 ] <= 0
 qc151: [ w0 ^2 - 2 w0 * p_75_minus[0] + w1 ^2 - 2 w1 * p_75_minus[1]
   + w2 ^2 - 2 w2 * p_75_minus[2] - Lambda ^2 + p_75_minus[0] ^2
   + p_75_minus[1] ^2 + p_75_minus[2] ^2 ] <= 0
 qc152: [ w0 ^2 + 2 w0 * p_76_plus[0] + w1 ^2 + 2 w1 * p_76_plus[1] + w2 ^2
   + 2 w2 * p_76_plus[2] - Lambda ^2 + p_76_plus[0] ^2 + p_76_plus[1] ^2
   + p_76_plus[2] ^2 ] <= 0
 qc153: [ w0 ^2 - 2 w0 * p_76_minus[0] + w1 ^2 - 2 w1 * p_76_minus[1]
   + w2 ^2 - 2 w2 * p_76_minus[2] - Lambda ^2 + p_76_minus[0] ^2
   + p_76_minus[1] ^2 + p_76_minus[2] ^2 ] <= 0
 qc154: [ w0 ^2 + 2 w0 * p_77_plus[0] + w1 ^2 + 2 w1 * p_77_plus[1] + w2 ^2
   + 2 w2 * p_77_plus[2] - Lambda ^2 + p_77_plus[0] ^2 + p_77_plus[1] ^2
   + p_77_plus[2] ^2 ] <= 0
 qc155: [ w0 ^2 - 2 w0 * p_77_minus[0] + w1 ^2 - 2 w1 * p_77_minus[1]
   + w2 ^2 - 2 w2 * p_77_minus[2] - Lambda ^2 + p_77_minus[0] ^2
   + p_77_minus[1] ^2 + p_77_minus[2] ^2 ] <= 0
 qc156: [ w0 ^2 + 2 w0 * p_78_plus[0] + w1 ^2 + 2 w1 * p_78_plus[1] + w2 ^2
   + 2 w2 * p_78_plus[2] - Lambda ^2 + p_78_plus[0] ^2 + p_78_plus[1] ^2
   + p_78_plus[2] ^2 ] <= 0
 qc157: [ w0 ^2 - 2 w0 * p_78_minus[0] + w1 ^2 - 2 w1 * p_78_minus[1]
   + w2 ^2 - 2 w2 * p_78_minus[2] - Lambda ^2 + p_78_minus[0] ^2
   + p_78_minus[1] ^2 + p_78_minus[2] ^2 ] <= 0
 qc158: [ w0 ^2 + 2 w0 * p_79_plus[0] + w1 ^2 + 2 w1 * p_79_plus[1] + w2 ^2
   + 2 w2 * p_79_plus[2] - Lambda ^2 + p_79_plus[0] ^2 + p_79_plus[1] ^2
   + p_79_plus[2] ^2 ] <= 0
 qc159: [ w0 ^2 - 2 w0 * p_79_minus[0] + w1 ^2 - 2 w1 * p_79_minus[1]
   + w2 ^2 - 2 w2 * p_79_minus[2] - Lambda ^2 + p_79_minus[0] ^2
   + p_79_minus[1] ^2 + p_79_minus[2] ^2 ] <= 0
 qc160: [ w0 ^2 + 2 w0 * p_80_plus[0] + w1 ^2 + 2 w1 * p_80_plus[1] + w2 ^2
   + 2 w2 * p_80_plus[2] - Lambda ^2 + p_80_plus[0] ^2 + p_80_plus[1] ^2
   + p_80_plus[2] ^2 ] <= 0
 qc161: [ w0 ^2 - 2 w0 * p_80_minus[0] + w1 ^2 - 2 w1 * p_80_minus[1]
   + w2 ^2 - 2 w2 * p_80_minus[2] - Lambda ^2 + p_80_minus[0] ^2
   + p_80_minus[1] ^2 + p_80_minus[2] ^2 ] <= 0
 qc162: [ w0 ^2 + 2 w0 * p_81_plus[0] + w1 ^2 + 2 w1 * p_81_plus[1] + w2 ^2
   + 2 w2 * p_81_plus[2] - Lambda ^2 + p_81_plus[0] ^2 + p_81_plus[1] ^2
   + p_81_plus[2] ^2 ] <= 0
 qc163: [ w0 ^2 - 2 w0 * p_81_minus[0] + w1 ^2 - 2 w1 * p_81_minus[1]
   + w2 ^2 - 2 w2 * p_81_minus[2] - Lambda ^2 + p_81_minus[0] ^2
   + p_81_minus[1] ^2 + p_81_minus[2] ^2 ] <= 0
 qc164: [ w0 ^2 + 2 w0 * p_82_plus[0] + w1 ^2 + 2 w1 * p_82_plus[1] + w2 ^2
   + 2 w2 * p_82_plus[2] - Lambda ^2 + p_82_plus[0] ^2 + p_82_plus[1] ^2
   + p_82_plus[2] ^2 ] <= 0
 qc165: [ w0 ^2 - 2 w0 * p_82_minus[0] + w1 ^2 - 2 w1 * p_82_minus[1]
   + w2 ^2 - 2 w2 * p_82_minus[2] - Lambda ^2 + p_82_minus[0] ^2
   + p_82_minus[1] ^2 + p_82_minus[2] ^2 ] <= 0
 qc166: [ w0 ^2 + 2 w0 * p_83_plus[0] + w1 ^2 + 2 w1 * p_83_plus[1] + w2 ^2
   + 2 w2 * p_83_plus[2] - Lambda ^2 + p_83_plus[0] ^2 + p_83_plus[1] ^2
   + p_83_plus[2] ^2 ] <= 0
 qc167: [ w0 ^2 - 2 w0 * p_83_minus[0] + w1 ^2 - 2 w1 * p_83_minus[1]
   + w2 ^2 - 2 w2 * p_83_minus[2] - Lambda ^2 + p_83_minus[0] ^2
   + p_83_minus[1] ^2 + p_83_minus[2] ^2 ] <= 0
 qc168: [ w0 ^2 + 2 w0 * p_84_plus[0] + w1 ^2 + 2 w1 * p_84_plus[1] + w2 ^2
   + 2 w2 * p_84_plus[2] - Lambda ^2 + p_84_plus[0] ^2 + p_84_plus[1] ^2
   + p_84_plus[2] ^2 ] <= 0
 qc169: [ w0 ^2 - 2 w0 * p_84_minus[0] + w1 ^2 - 2 w1 * p_84_minus[1]
   + w2 ^2 - 2 w2 * p_84_minus[2] - Lambda ^2 + p_84_minus[0] ^2
   + p_84_minus[1] ^2 + p_84_minus[2] ^2 ] <= 0
 qc170: [ w0 ^2 + 2 w0 * p_85_plus[0] + w1 ^2 + 2 w1 * p_85_plus[1] + w2 ^2
   + 2 w2 * p_85_plus[2] - Lambda ^2 + p_85_plus[0] ^2 + p_85_plus[1] ^2
   + p_85_plus[2] ^2 ] <= 0
 qc171: [ w0 ^2 - 2 w0 * p_85_minus[0] + w1 ^2 - 2 w1 * p_85_minus[1]
   + w2 ^2 - 2 w2 * p_85_minus[2] - Lambda ^2 + p_85_minus[0] ^2
   + p_85_minus[1] ^2 + p_85_minus[2] ^2 ] <= 0
 qc172: [ w0 ^2 + 2 w0 * p_86_plus[0] + w1 ^2 + 2 w1 * p_86_plus[1] + w2 ^2
   + 2 w2 * p_86_plus[2] - Lambda ^2 + p_86_plus[0] ^2 + p_86_plus[1] ^2
   + p_86_plus[2] ^2 ] <= 0
 qc173: [ w0 ^2 - 2 w0 * p_86_minus[0] + w1 ^2 - 2 w1 * p_86_minus[1]
   + w2 ^2 - 2 w2 * p_86_minus[2] - Lambda ^2 + p_86_minus[0] ^2
   + p_86_minus[1] ^2 + p_86_minus[2] ^2 ] <= 0
 qc174: [ w0 ^2 + 2 w0 * p_87_plus[0] + w1 ^2 + 2 w1 * p_87_plus[1] + w2 ^2
   + 2 w2 * p_87_plus[2] - Lambda ^2 + p_87_plus[0] ^2 + p_87_plus[1] ^2
   + p_87_plus[2] ^2 ] <= 0
 qc175: [ w0 ^2 - 2 w0 * p_87_minus[0] + w1 ^2 - 2 w1 * p_87_minus[1]
   + w2 ^2 - 2 w2 * p_87_minus[2] - Lambda ^2 + p_87_minus[0] ^2
   + p_87_minus[1] ^2 + p_87_minus[2] ^2 ] <= 0
 qc176: [ w0 ^2 + 2 w0 * p_88_plus[0] + w1 ^2 + 2 w1 * p_88_plus[1] + w2 ^2
   + 2 w2 * p_88_plus[2] - Lambda ^2 + p_88_plus[0] ^2 + p_88_plus[1] ^2
   + p_88_plus[2] ^2 ] <= 0
 qc177: [ w0 ^2 - 2 w0 * p_88_minus[0] + w1 ^2 - 2 w1 * p_88_minus[1]
   + w2 ^2 - 2 w2 * p_88_minus[2] - Lambda ^2 + p_88_minus[0] ^2
   + p_88_minus[1] ^2 + p_88_minus[2] ^2 ] <= 0
 qc178: [ w0 ^2 + 2 w0 * p_89_plus[0] + w1 ^2 + 2 w1 * p_89_plus[1] + w2 ^2
   + 2 w2 * p_89_plus[2] - Lambda ^2 + p_89_plus[0] ^2 + p_89_plus[1] ^2
   + p_89_plus[2] ^2 ] <= 0
 qc179: [ w0 ^2 - 2 w0 * p_89_minus[0] + w1 ^2 - 2 w1 * p_89_minus[1]
   + w2 ^2 - 2 w2 * p_89_minus[2] - Lambda ^2 + p_89_minus[0] ^2
   + p_89_minus[1] ^2 + p_89_minus[2] ^2 ] <= 0
 qc180: [ w0 ^2 + 2 w0 * p_90_plus[0] + w1 ^2 + 2 w1 * p_90_plus[1] + w2 ^2
   + 2 w2 * p_90_plus[2] - Lambda ^2 + p_90_plus[0] ^2 + p_90_plus[1] ^2
   + p_90_plus[2] ^2 ] <= 0
 qc181: [ w0 ^2 - 2 w0 * p_90_minus[0] + w1 ^2 - 2 w1 * p_90_minus[1]
   + w2 ^2 - 2 w2 * p_90_minus[2] - Lambda ^2 + p_90_minus[0] ^2
   + p_90_minus[1] ^2 + p_90_minus[2] ^2 ] <= 0
 qc182: [ w0 ^2 + 2 w0 * p_91_plus[0] + w1 ^2 + 2 w1 * p_91_plus[1] + w2 ^2
   + 2 w2 * p_91_plus[2] - Lambda ^2 + p_91_plus[0] ^2 + p_91_plus[1] ^2
   + p_91_plus[2] ^2 ] <= 0
 qc183: [ w0 ^2 - 2 w0 * p_91_minus[0] + w1 ^2 - 2 w1 * p_91_minus[1]
   + w2 ^2 - 2 w2 * p_91_minus[2] - Lambda ^2 + p_91_minus[0] ^2
   + p_91_minus[1] ^2 + p_91_minus[2] ^2 ] <= 0
 qc184: [ w0 ^2 + 2 w0 * p_92_plus[0] + w1 ^2 + 2 w1 * p_92_plus[1] + w2 ^2
   + 2 w2 * p_92_plus[2] - Lambda ^2 + p_92_plus[0] ^2 + p_92_plus[1] ^2
   + p_92_plus[2] ^2 ] <= 0
 qc185: [ w0 ^2 - 2 w0 * p_92_minus[0] + w1 ^2 - 2 w1 * p_92_minus[1]
   + w2 ^2 - 2 w2 * p_92_minus[2] - Lambda ^2 + p_92_minus[0] ^2
   + p_92_minus[1] ^2 + p_92_minus[2] ^2 ] <= 0
 qc186: [ w0 ^2 + 2 w0 * p_93_plus[0] + w1 ^2 + 2 w1 * p_93_plus[1] + w2 ^2
   + 2 w2 * p_93_plus[2] - Lambda ^2 + p_93_plus[0] ^2 + p_93_plus[1] ^2
   + p_93_plus[2] ^2 ] <= 0
 qc187: [ w0 ^2 - 2 w0 * p_93_minus[0] + w1 ^2 - 2 w1 * p_93_minus[1]
   + w2 ^2 - 2 w2 * p_93_minus[2] - Lambda ^2 + p_93_minus[0] ^2
   + p_93_minus[1] ^2 + p_93_minus[2] ^2 ] <= 0
 qc188: [ w0 ^2 + 2 w0 * p_94_plus[0] + w1 ^2 + 2 w1 * p_94_plus[1] + w2 ^2
   + 2 w2 * p_94_plus[2] - Lambda ^2 + p_94_plus[0] ^2 + p_94_plus[1] ^2
   + p_94_plus[2] ^2 ] <= 0
 qc189: [ w0 ^2 - 2 w0 * p_94_minus[0] + w1 ^2 - 2 w1 * p_94_minus[1]
   + w2 ^2 - 2 w2 * p_94_minus[2] - Lambda ^2 + p_94_minus[0] ^2
   + p_94_minus[1] ^2 + p_94_minus[2] ^2 ] <= 0
 qc190: [ w0 ^2 + 2 w0 * p_95_plus[0] + w1 ^2 + 2 w1 * p_95_plus[1] + w2 ^2
   + 2 w2 * p_95_plus[2] - Lambda ^2 + p_95_plus[0] ^2 + p_95_plus[1] ^2
   + p_95_plus[2] ^2 ] <= 0
 qc191: [ w0 ^2 - 2 w0 * p_95_minus[0] + w1 ^2 - 2 w1 * p_95_minus[1]
   + w2 ^2 - 2 w2 * p_95_minus[2] - Lambda ^2 + p_95_minus[0] ^2
   + p_95_minus[1] ^2 + p_95_minus[2] ^2 ] <= 0
 qc192: [ w0 ^2 + 2 w0 * p_96_plus[0] + w1 ^2 + 2 w1 * p_96_plus[1] + w2 ^2
   + 2 w2 * p_96_plus[2] - Lambda ^2 + p_96_plus[0] ^2 + p_96_plus[1] ^2
   + p_96_plus[2] ^2 ] <= 0
 qc193: [ w0 ^2 - 2 w0 * p_96_minus[0] + w1 ^2 - 2 w1 * p_96_minus[1]
   + w2 ^2 - 2 w2 * p_96_minus[2] - Lambda ^2 + p_96_minus[0] ^2
   + p_96_minus[1] ^2 + p_96_minus[2] ^2 ] <= 0
 qc194: [ w0 ^2 + 2 w0 * p_97_plus[0] + w1 ^2 + 2 w1 * p_97_plus[1] + w2 ^2
   + 2 w2 * p_97_plus[2] - Lambda ^2 + p_97_plus[0] ^2 + p_97_plus[1] ^2
   + p_97_plus[2] ^2 ] <= 0
 qc195: [ w0 ^2 - 2 w0 * p_97_minus[0] + w1 ^2 - 2 w1 * p_97_minus[1]
   + w2 ^2 - 2 w2 * p_97_minus[2] - Lambda ^2 + p_97_minus[0] ^2
   + p_97_minus[1] ^2 + p_97_minus[2] ^2 ] <= 0
 qc196: [ w0 ^2 + 2 w0 * p_98_plus[0] + w1 ^2 + 2 w1 * p_98_plus[1] + w2 ^2
   + 2 w2 * p_98_plus[2] - Lambda ^2 + p_98_plus[0] ^2 + p_98_plus[1] ^2
   + p_98_plus[2] ^2 ] <= 0
 qc197: [ w0 ^2 - 2 w0 * p_98_minus[0] + w1 ^2 - 2 w1 * p_98_minus[1]
   + w2 ^2 - 2 w2 * p_98_minus[2] - Lambda ^2 + p_98_minus[0] ^2
   + p_98_minus[1] ^2 + p_98_minus[2] ^2 ] <= 0
 qc198: [ w0 ^2 + 2 w0 * p_99_plus[0] + w1 ^2 + 2 w1 * p_99_plus[1] + w2 ^2
   + 2 w2 * p_99_plus[2] - Lambda ^2 + p_99_plus[0] ^2 + p_99_plus[1] ^2
   + p_99_plus[2] ^2 ] <= 0
 qc199: [ w0 ^2 - 2 w0 * p_99_minus[0] + w1 ^2 - 2 w1 * p_99_minus[1]
   + w2 ^2 - 2 w2 * p_99_minus[2] - Lambda ^2 + p_99_minus[0] ^2
   + p_99_minus[1] ^2 + p_99_minus[2] ^2 ] <= 0
Bounds
 w0 free
 w1 free
 w2 free
End
