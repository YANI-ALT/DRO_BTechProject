\ Model RegularisedSVM_withSupport
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.1 Lambda + 0.01 s_0 + 0.01 s_1 + 0.01 s_2 + 0.01 s_3 + 0.01 s_4
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
 data[0]_constr1: w0 + 5.1 w1 + 3.5 w2 - s_0 <= -1
 data[1]_constr1: w0 + 4.9 w1 + 3 w2 - s_1 <= -1
 data[2]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_2 <= -1
 data[3]_constr1: w0 + 4.6 w1 + 3.1 w2 - s_3 <= -1
 data[4]_constr1: w0 + 5 w1 + 3.6 w2 - s_4 <= -1
 data[5]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_5 <= -1
 data[6]_constr1: w0 + 4.6 w1 + 3.4 w2 - s_6 <= -1
 data[7]_constr1: w0 + 5 w1 + 3.4 w2 - s_7 <= -1
 data[8]_constr1: w0 + 4.4 w1 + 2.9 w2 - s_8 <= -1
 data[9]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_9 <= -1
 data[10]_constr1: w0 + 5.4 w1 + 3.7 w2 - s_10 <= -1
 data[11]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_11 <= -1
 data[12]_constr1: w0 + 4.8 w1 + 3 w2 - s_12 <= -1
 data[13]_constr1: w0 + 4.3 w1 + 3 w2 - s_13 <= -1
 data[14]_constr1: w0 + 5.8 w1 + 4 w2 - s_14 <= -1
 data[15]_constr1: w0 + 5.7 w1 + 4.4 w2 - s_15 <= -1
 data[16]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_16 <= -1
 data[17]_constr1: w0 + 5.1 w1 + 3.5 w2 - s_17 <= -1
 data[18]_constr1: w0 + 5.7 w1 + 3.8 w2 - s_18 <= -1
 data[19]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_19 <= -1
 data[20]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_20 <= -1
 data[21]_constr1: w0 + 5.1 w1 + 3.7 w2 - s_21 <= -1
 data[22]_constr1: w0 + 4.6 w1 + 3.6 w2 - s_22 <= -1
 data[23]_constr1: w0 + 5.1 w1 + 3.3 w2 - s_23 <= -1
 data[24]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_24 <= -1
 data[25]_constr1: w0 + 5 w1 + 3 w2 - s_25 <= -1
 data[26]_constr1: w0 + 5 w1 + 3.4 w2 - s_26 <= -1
 data[27]_constr1: w0 + 5.2 w1 + 3.5 w2 - s_27 <= -1
 data[28]_constr1: w0 + 5.2 w1 + 3.4 w2 - s_28 <= -1
 data[29]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_29 <= -1
 data[30]_constr1: w0 + 4.8 w1 + 3.1 w2 - s_30 <= -1
 data[31]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_31 <= -1
 data[32]_constr1: w0 + 5.2 w1 + 4.1 w2 - s_32 <= -1
 data[33]_constr1: w0 + 5.5 w1 + 4.2 w2 - s_33 <= -1
 data[34]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_34 <= -1
 data[35]_constr1: w0 + 5 w1 + 3.2 w2 - s_35 <= -1
 data[36]_constr1: w0 + 5.5 w1 + 3.5 w2 - s_36 <= -1
 data[37]_constr1: w0 + 4.9 w1 + 3.6 w2 - s_37 <= -1
 data[38]_constr1: w0 + 4.4 w1 + 3 w2 - s_38 <= -1
 data[39]_constr1: w0 + 5.1 w1 + 3.4 w2 - s_39 <= -1
 data[40]_constr1: w0 + 5 w1 + 3.5 w2 - s_40 <= -1
 data[41]_constr1: w0 + 4.5 w1 + 2.3 w2 - s_41 <= -1
 data[42]_constr1: w0 + 4.4 w1 + 3.2 w2 - s_42 <= -1
 data[43]_constr1: w0 + 5 w1 + 3.5 w2 - s_43 <= -1
 data[44]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_44 <= -1
 data[45]_constr1: w0 + 4.8 w1 + 3 w2 - s_45 <= -1
 data[46]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_46 <= -1
 data[47]_constr1: w0 + 4.6 w1 + 3.2 w2 - s_47 <= -1
 data[48]_constr1: w0 + 5.3 w1 + 3.7 w2 - s_48 <= -1
 data[49]_constr1: w0 + 5 w1 + 3.3 w2 - s_49 <= -1
 data[50]_constr1: - w0 - 6.3 w1 - 3.3 w2 - s_50 <= -1
 data[51]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_51 <= -1
 data[52]_constr1: - w0 - 7.1 w1 - 3 w2 - s_52 <= -1
 data[53]_constr1: - w0 - 6.3 w1 - 2.9 w2 - s_53 <= -1
 data[54]_constr1: - w0 - 6.5 w1 - 3 w2 - s_54 <= -1
 data[55]_constr1: - w0 - 7.6 w1 - 3 w2 - s_55 <= -1
 data[56]_constr1: - w0 - 4.9 w1 - 2.5 w2 - s_56 <= -1
 data[57]_constr1: - w0 - 7.3 w1 - 2.9 w2 - s_57 <= -1
 data[58]_constr1: - w0 - 6.7 w1 - 2.5 w2 - s_58 <= -1
 data[59]_constr1: - w0 - 7.2 w1 - 3.6 w2 - s_59 <= -1
 data[60]_constr1: - w0 - 6.5 w1 - 3.2 w2 - s_60 <= -1
 data[61]_constr1: - w0 - 6.4 w1 - 2.7 w2 - s_61 <= -1
 data[62]_constr1: - w0 - 6.8 w1 - 3 w2 - s_62 <= -1
 data[63]_constr1: - w0 - 5.7 w1 - 2.5 w2 - s_63 <= -1
 data[64]_constr1: - w0 - 5.8 w1 - 2.8 w2 - s_64 <= -1
 data[65]_constr1: - w0 - 6.4 w1 - 3.2 w2 - s_65 <= -1
 data[66]_constr1: - w0 - 6.5 w1 - 3 w2 - s_66 <= -1
 data[67]_constr1: - w0 - 7.7 w1 - 3.8 w2 - s_67 <= -1
 data[68]_constr1: - w0 - 7.7 w1 - 2.6 w2 - s_68 <= -1
 data[69]_constr1: - w0 - 6 w1 - 2.2 w2 - s_69 <= -1
 data[70]_constr1: - w0 - 6.9 w1 - 3.2 w2 - s_70 <= -1
 data[71]_constr1: - w0 - 5.6 w1 - 2.8 w2 - s_71 <= -1
 data[72]_constr1: - w0 - 7.7 w1 - 2.8 w2 - s_72 <= -1
 data[73]_constr1: - w0 - 6.3 w1 - 2.7 w2 - s_73 <= -1
 data[74]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_74 <= -1
 data[75]_constr1: - w0 - 7.2 w1 - 3.2 w2 - s_75 <= -1
 data[76]_constr1: - w0 - 6.2 w1 - 2.8 w2 - s_76 <= -1
 data[77]_constr1: - w0 - 6.1 w1 - 3 w2 - s_77 <= -1
 data[78]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_78 <= -1
 data[79]_constr1: - w0 - 7.2 w1 - 3 w2 - s_79 <= -1
 data[80]_constr1: - w0 - 7.4 w1 - 2.8 w2 - s_80 <= -1
 data[81]_constr1: - w0 - 7.9 w1 - 3.8 w2 - s_81 <= -1
 data[82]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_82 <= -1
 data[83]_constr1: - w0 - 6.3 w1 - 2.8 w2 - s_83 <= -1
 data[84]_constr1: - w0 - 6.1 w1 - 2.6 w2 - s_84 <= -1
 data[85]_constr1: - w0 - 7.7 w1 - 3 w2 - s_85 <= -1
 data[86]_constr1: - w0 - 6.3 w1 - 3.4 w2 - s_86 <= -1
 data[87]_constr1: - w0 - 6.4 w1 - 3.1 w2 - s_87 <= -1
 data[88]_constr1: - w0 - 6 w1 - 3 w2 - s_88 <= -1
 data[89]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_89 <= -1
 data[90]_constr1: - w0 - 6.7 w1 - 3.1 w2 - s_90 <= -1
 data[91]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_91 <= -1
 data[92]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_92 <= -1
 data[93]_constr1: - w0 - 6.8 w1 - 3.2 w2 - s_93 <= -1
 data[94]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_94 <= -1
 data[95]_constr1: - w0 - 6.7 w1 - 3 w2 - s_95 <= -1
 data[96]_constr1: - w0 - 6.3 w1 - 2.5 w2 - s_96 <= -1
 data[97]_constr1: - w0 - 6.5 w1 - 3 w2 - s_97 <= -1
 data[98]_constr1: - w0 - 6.2 w1 - 3.4 w2 - s_98 <= -1
 data[99]_constr1: - w0 - 5.9 w1 - 3 w2 - s_99 <= -1
 s_0>=0: s_0 >= 0
 p_[0,0]_plus>=0: p_0_plus[0] >= 0
 p_[0,1]_plus>=0: p_0_plus[1] >= 0
 p_[0,2]_plus>=0: p_0_plus[2] >= 0
 s_1>=0: s_1 >= 0
 p_[1,0]_plus>=0: p_1_plus[0] >= 0
 p_[1,1]_plus>=0: p_1_plus[1] >= 0
 p_[1,2]_plus>=0: p_1_plus[2] >= 0
 s_2>=0: s_2 >= 0
 p_[2,0]_plus>=0: p_2_plus[0] >= 0
 p_[2,1]_plus>=0: p_2_plus[1] >= 0
 p_[2,2]_plus>=0: p_2_plus[2] >= 0
 s_3>=0: s_3 >= 0
 p_[3,0]_plus>=0: p_3_plus[0] >= 0
 p_[3,1]_plus>=0: p_3_plus[1] >= 0
 p_[3,2]_plus>=0: p_3_plus[2] >= 0
 s_4>=0: s_4 >= 0
 p_[4,0]_plus>=0: p_4_plus[0] >= 0
 p_[4,1]_plus>=0: p_4_plus[1] >= 0
 p_[4,2]_plus>=0: p_4_plus[2] >= 0
 s_5>=0: s_5 >= 0
 p_[5,0]_plus>=0: p_5_plus[0] >= 0
 p_[5,1]_plus>=0: p_5_plus[1] >= 0
 p_[5,2]_plus>=0: p_5_plus[2] >= 0
 s_6>=0: s_6 >= 0
 p_[6,0]_plus>=0: p_6_plus[0] >= 0
 p_[6,1]_plus>=0: p_6_plus[1] >= 0
 p_[6,2]_plus>=0: p_6_plus[2] >= 0
 s_7>=0: s_7 >= 0
 p_[7,0]_plus>=0: p_7_plus[0] >= 0
 p_[7,1]_plus>=0: p_7_plus[1] >= 0
 p_[7,2]_plus>=0: p_7_plus[2] >= 0
 s_8>=0: s_8 >= 0
 p_[8,0]_plus>=0: p_8_plus[0] >= 0
 p_[8,1]_plus>=0: p_8_plus[1] >= 0
 p_[8,2]_plus>=0: p_8_plus[2] >= 0
 s_9>=0: s_9 >= 0
 p_[9,0]_plus>=0: p_9_plus[0] >= 0
 p_[9,1]_plus>=0: p_9_plus[1] >= 0
 p_[9,2]_plus>=0: p_9_plus[2] >= 0
 s_10>=0: s_10 >= 0
 p_[10,0]_plus>=0: p_10_plus[0] >= 0
 p_[10,1]_plus>=0: p_10_plus[1] >= 0
 p_[10,2]_plus>=0: p_10_plus[2] >= 0
 s_11>=0: s_11 >= 0
 p_[11,0]_plus>=0: p_11_plus[0] >= 0
 p_[11,1]_plus>=0: p_11_plus[1] >= 0
 p_[11,2]_plus>=0: p_11_plus[2] >= 0
 s_12>=0: s_12 >= 0
 p_[12,0]_plus>=0: p_12_plus[0] >= 0
 p_[12,1]_plus>=0: p_12_plus[1] >= 0
 p_[12,2]_plus>=0: p_12_plus[2] >= 0
 s_13>=0: s_13 >= 0
 p_[13,0]_plus>=0: p_13_plus[0] >= 0
 p_[13,1]_plus>=0: p_13_plus[1] >= 0
 p_[13,2]_plus>=0: p_13_plus[2] >= 0
 s_14>=0: s_14 >= 0
 p_[14,0]_plus>=0: p_14_plus[0] >= 0
 p_[14,1]_plus>=0: p_14_plus[1] >= 0
 p_[14,2]_plus>=0: p_14_plus[2] >= 0
 s_15>=0: s_15 >= 0
 p_[15,0]_plus>=0: p_15_plus[0] >= 0
 p_[15,1]_plus>=0: p_15_plus[1] >= 0
 p_[15,2]_plus>=0: p_15_plus[2] >= 0
 s_16>=0: s_16 >= 0
 p_[16,0]_plus>=0: p_16_plus[0] >= 0
 p_[16,1]_plus>=0: p_16_plus[1] >= 0
 p_[16,2]_plus>=0: p_16_plus[2] >= 0
 s_17>=0: s_17 >= 0
 p_[17,0]_plus>=0: p_17_plus[0] >= 0
 p_[17,1]_plus>=0: p_17_plus[1] >= 0
 p_[17,2]_plus>=0: p_17_plus[2] >= 0
 s_18>=0: s_18 >= 0
 p_[18,0]_plus>=0: p_18_plus[0] >= 0
 p_[18,1]_plus>=0: p_18_plus[1] >= 0
 p_[18,2]_plus>=0: p_18_plus[2] >= 0
 s_19>=0: s_19 >= 0
 p_[19,0]_plus>=0: p_19_plus[0] >= 0
 p_[19,1]_plus>=0: p_19_plus[1] >= 0
 p_[19,2]_plus>=0: p_19_plus[2] >= 0
 s_20>=0: s_20 >= 0
 p_[20,0]_plus>=0: p_20_plus[0] >= 0
 p_[20,1]_plus>=0: p_20_plus[1] >= 0
 p_[20,2]_plus>=0: p_20_plus[2] >= 0
 s_21>=0: s_21 >= 0
 p_[21,0]_plus>=0: p_21_plus[0] >= 0
 p_[21,1]_plus>=0: p_21_plus[1] >= 0
 p_[21,2]_plus>=0: p_21_plus[2] >= 0
 s_22>=0: s_22 >= 0
 p_[22,0]_plus>=0: p_22_plus[0] >= 0
 p_[22,1]_plus>=0: p_22_plus[1] >= 0
 p_[22,2]_plus>=0: p_22_plus[2] >= 0
 s_23>=0: s_23 >= 0
 p_[23,0]_plus>=0: p_23_plus[0] >= 0
 p_[23,1]_plus>=0: p_23_plus[1] >= 0
 p_[23,2]_plus>=0: p_23_plus[2] >= 0
 s_24>=0: s_24 >= 0
 p_[24,0]_plus>=0: p_24_plus[0] >= 0
 p_[24,1]_plus>=0: p_24_plus[1] >= 0
 p_[24,2]_plus>=0: p_24_plus[2] >= 0
 s_25>=0: s_25 >= 0
 p_[25,0]_plus>=0: p_25_plus[0] >= 0
 p_[25,1]_plus>=0: p_25_plus[1] >= 0
 p_[25,2]_plus>=0: p_25_plus[2] >= 0
 s_26>=0: s_26 >= 0
 p_[26,0]_plus>=0: p_26_plus[0] >= 0
 p_[26,1]_plus>=0: p_26_plus[1] >= 0
 p_[26,2]_plus>=0: p_26_plus[2] >= 0
 s_27>=0: s_27 >= 0
 p_[27,0]_plus>=0: p_27_plus[0] >= 0
 p_[27,1]_plus>=0: p_27_plus[1] >= 0
 p_[27,2]_plus>=0: p_27_plus[2] >= 0
 s_28>=0: s_28 >= 0
 p_[28,0]_plus>=0: p_28_plus[0] >= 0
 p_[28,1]_plus>=0: p_28_plus[1] >= 0
 p_[28,2]_plus>=0: p_28_plus[2] >= 0
 s_29>=0: s_29 >= 0
 p_[29,0]_plus>=0: p_29_plus[0] >= 0
 p_[29,1]_plus>=0: p_29_plus[1] >= 0
 p_[29,2]_plus>=0: p_29_plus[2] >= 0
 s_30>=0: s_30 >= 0
 p_[30,0]_plus>=0: p_30_plus[0] >= 0
 p_[30,1]_plus>=0: p_30_plus[1] >= 0
 p_[30,2]_plus>=0: p_30_plus[2] >= 0
 s_31>=0: s_31 >= 0
 p_[31,0]_plus>=0: p_31_plus[0] >= 0
 p_[31,1]_plus>=0: p_31_plus[1] >= 0
 p_[31,2]_plus>=0: p_31_plus[2] >= 0
 s_32>=0: s_32 >= 0
 p_[32,0]_plus>=0: p_32_plus[0] >= 0
 p_[32,1]_plus>=0: p_32_plus[1] >= 0
 p_[32,2]_plus>=0: p_32_plus[2] >= 0
 s_33>=0: s_33 >= 0
 p_[33,0]_plus>=0: p_33_plus[0] >= 0
 p_[33,1]_plus>=0: p_33_plus[1] >= 0
 p_[33,2]_plus>=0: p_33_plus[2] >= 0
 s_34>=0: s_34 >= 0
 p_[34,0]_plus>=0: p_34_plus[0] >= 0
 p_[34,1]_plus>=0: p_34_plus[1] >= 0
 p_[34,2]_plus>=0: p_34_plus[2] >= 0
 s_35>=0: s_35 >= 0
 p_[35,0]_plus>=0: p_35_plus[0] >= 0
 p_[35,1]_plus>=0: p_35_plus[1] >= 0
 p_[35,2]_plus>=0: p_35_plus[2] >= 0
 s_36>=0: s_36 >= 0
 p_[36,0]_plus>=0: p_36_plus[0] >= 0
 p_[36,1]_plus>=0: p_36_plus[1] >= 0
 p_[36,2]_plus>=0: p_36_plus[2] >= 0
 s_37>=0: s_37 >= 0
 p_[37,0]_plus>=0: p_37_plus[0] >= 0
 p_[37,1]_plus>=0: p_37_plus[1] >= 0
 p_[37,2]_plus>=0: p_37_plus[2] >= 0
 s_38>=0: s_38 >= 0
 p_[38,0]_plus>=0: p_38_plus[0] >= 0
 p_[38,1]_plus>=0: p_38_plus[1] >= 0
 p_[38,2]_plus>=0: p_38_plus[2] >= 0
 s_39>=0: s_39 >= 0
 p_[39,0]_plus>=0: p_39_plus[0] >= 0
 p_[39,1]_plus>=0: p_39_plus[1] >= 0
 p_[39,2]_plus>=0: p_39_plus[2] >= 0
 s_40>=0: s_40 >= 0
 p_[40,0]_plus>=0: p_40_plus[0] >= 0
 p_[40,1]_plus>=0: p_40_plus[1] >= 0
 p_[40,2]_plus>=0: p_40_plus[2] >= 0
 s_41>=0: s_41 >= 0
 p_[41,0]_plus>=0: p_41_plus[0] >= 0
 p_[41,1]_plus>=0: p_41_plus[1] >= 0
 p_[41,2]_plus>=0: p_41_plus[2] >= 0
 s_42>=0: s_42 >= 0
 p_[42,0]_plus>=0: p_42_plus[0] >= 0
 p_[42,1]_plus>=0: p_42_plus[1] >= 0
 p_[42,2]_plus>=0: p_42_plus[2] >= 0
 s_43>=0: s_43 >= 0
 p_[43,0]_plus>=0: p_43_plus[0] >= 0
 p_[43,1]_plus>=0: p_43_plus[1] >= 0
 p_[43,2]_plus>=0: p_43_plus[2] >= 0
 s_44>=0: s_44 >= 0
 p_[44,0]_plus>=0: p_44_plus[0] >= 0
 p_[44,1]_plus>=0: p_44_plus[1] >= 0
 p_[44,2]_plus>=0: p_44_plus[2] >= 0
 s_45>=0: s_45 >= 0
 p_[45,0]_plus>=0: p_45_plus[0] >= 0
 p_[45,1]_plus>=0: p_45_plus[1] >= 0
 p_[45,2]_plus>=0: p_45_plus[2] >= 0
 s_46>=0: s_46 >= 0
 p_[46,0]_plus>=0: p_46_plus[0] >= 0
 p_[46,1]_plus>=0: p_46_plus[1] >= 0
 p_[46,2]_plus>=0: p_46_plus[2] >= 0
 s_47>=0: s_47 >= 0
 p_[47,0]_plus>=0: p_47_plus[0] >= 0
 p_[47,1]_plus>=0: p_47_plus[1] >= 0
 p_[47,2]_plus>=0: p_47_plus[2] >= 0
 s_48>=0: s_48 >= 0
 p_[48,0]_plus>=0: p_48_plus[0] >= 0
 p_[48,1]_plus>=0: p_48_plus[1] >= 0
 p_[48,2]_plus>=0: p_48_plus[2] >= 0
 s_49>=0: s_49 >= 0
 p_[49,0]_plus>=0: p_49_plus[0] >= 0
 p_[49,1]_plus>=0: p_49_plus[1] >= 0
 p_[49,2]_plus>=0: p_49_plus[2] >= 0
 s_50>=0: s_50 >= 0
 p_[50,0]_plus>=0: p_50_plus[0] >= 0
 p_[50,1]_plus>=0: p_50_plus[1] >= 0
 p_[50,2]_plus>=0: p_50_plus[2] >= 0
 s_51>=0: s_51 >= 0
 p_[51,0]_plus>=0: p_51_plus[0] >= 0
 p_[51,1]_plus>=0: p_51_plus[1] >= 0
 p_[51,2]_plus>=0: p_51_plus[2] >= 0
 s_52>=0: s_52 >= 0
 p_[52,0]_plus>=0: p_52_plus[0] >= 0
 p_[52,1]_plus>=0: p_52_plus[1] >= 0
 p_[52,2]_plus>=0: p_52_plus[2] >= 0
 s_53>=0: s_53 >= 0
 p_[53,0]_plus>=0: p_53_plus[0] >= 0
 p_[53,1]_plus>=0: p_53_plus[1] >= 0
 p_[53,2]_plus>=0: p_53_plus[2] >= 0
 s_54>=0: s_54 >= 0
 p_[54,0]_plus>=0: p_54_plus[0] >= 0
 p_[54,1]_plus>=0: p_54_plus[1] >= 0
 p_[54,2]_plus>=0: p_54_plus[2] >= 0
 s_55>=0: s_55 >= 0
 p_[55,0]_plus>=0: p_55_plus[0] >= 0
 p_[55,1]_plus>=0: p_55_plus[1] >= 0
 p_[55,2]_plus>=0: p_55_plus[2] >= 0
 s_56>=0: s_56 >= 0
 p_[56,0]_plus>=0: p_56_plus[0] >= 0
 p_[56,1]_plus>=0: p_56_plus[1] >= 0
 p_[56,2]_plus>=0: p_56_plus[2] >= 0
 s_57>=0: s_57 >= 0
 p_[57,0]_plus>=0: p_57_plus[0] >= 0
 p_[57,1]_plus>=0: p_57_plus[1] >= 0
 p_[57,2]_plus>=0: p_57_plus[2] >= 0
 s_58>=0: s_58 >= 0
 p_[58,0]_plus>=0: p_58_plus[0] >= 0
 p_[58,1]_plus>=0: p_58_plus[1] >= 0
 p_[58,2]_plus>=0: p_58_plus[2] >= 0
 s_59>=0: s_59 >= 0
 p_[59,0]_plus>=0: p_59_plus[0] >= 0
 p_[59,1]_plus>=0: p_59_plus[1] >= 0
 p_[59,2]_plus>=0: p_59_plus[2] >= 0
 s_60>=0: s_60 >= 0
 p_[60,0]_plus>=0: p_60_plus[0] >= 0
 p_[60,1]_plus>=0: p_60_plus[1] >= 0
 p_[60,2]_plus>=0: p_60_plus[2] >= 0
 s_61>=0: s_61 >= 0
 p_[61,0]_plus>=0: p_61_plus[0] >= 0
 p_[61,1]_plus>=0: p_61_plus[1] >= 0
 p_[61,2]_plus>=0: p_61_plus[2] >= 0
 s_62>=0: s_62 >= 0
 p_[62,0]_plus>=0: p_62_plus[0] >= 0
 p_[62,1]_plus>=0: p_62_plus[1] >= 0
 p_[62,2]_plus>=0: p_62_plus[2] >= 0
 s_63>=0: s_63 >= 0
 p_[63,0]_plus>=0: p_63_plus[0] >= 0
 p_[63,1]_plus>=0: p_63_plus[1] >= 0
 p_[63,2]_plus>=0: p_63_plus[2] >= 0
 s_64>=0: s_64 >= 0
 p_[64,0]_plus>=0: p_64_plus[0] >= 0
 p_[64,1]_plus>=0: p_64_plus[1] >= 0
 p_[64,2]_plus>=0: p_64_plus[2] >= 0
 s_65>=0: s_65 >= 0
 p_[65,0]_plus>=0: p_65_plus[0] >= 0
 p_[65,1]_plus>=0: p_65_plus[1] >= 0
 p_[65,2]_plus>=0: p_65_plus[2] >= 0
 s_66>=0: s_66 >= 0
 p_[66,0]_plus>=0: p_66_plus[0] >= 0
 p_[66,1]_plus>=0: p_66_plus[1] >= 0
 p_[66,2]_plus>=0: p_66_plus[2] >= 0
 s_67>=0: s_67 >= 0
 p_[67,0]_plus>=0: p_67_plus[0] >= 0
 p_[67,1]_plus>=0: p_67_plus[1] >= 0
 p_[67,2]_plus>=0: p_67_plus[2] >= 0
 s_68>=0: s_68 >= 0
 p_[68,0]_plus>=0: p_68_plus[0] >= 0
 p_[68,1]_plus>=0: p_68_plus[1] >= 0
 p_[68,2]_plus>=0: p_68_plus[2] >= 0
 s_69>=0: s_69 >= 0
 p_[69,0]_plus>=0: p_69_plus[0] >= 0
 p_[69,1]_plus>=0: p_69_plus[1] >= 0
 p_[69,2]_plus>=0: p_69_plus[2] >= 0
 s_70>=0: s_70 >= 0
 p_[70,0]_plus>=0: p_70_plus[0] >= 0
 p_[70,1]_plus>=0: p_70_plus[1] >= 0
 p_[70,2]_plus>=0: p_70_plus[2] >= 0
 s_71>=0: s_71 >= 0
 p_[71,0]_plus>=0: p_71_plus[0] >= 0
 p_[71,1]_plus>=0: p_71_plus[1] >= 0
 p_[71,2]_plus>=0: p_71_plus[2] >= 0
 s_72>=0: s_72 >= 0
 p_[72,0]_plus>=0: p_72_plus[0] >= 0
 p_[72,1]_plus>=0: p_72_plus[1] >= 0
 p_[72,2]_plus>=0: p_72_plus[2] >= 0
 s_73>=0: s_73 >= 0
 p_[73,0]_plus>=0: p_73_plus[0] >= 0
 p_[73,1]_plus>=0: p_73_plus[1] >= 0
 p_[73,2]_plus>=0: p_73_plus[2] >= 0
 s_74>=0: s_74 >= 0
 p_[74,0]_plus>=0: p_74_plus[0] >= 0
 p_[74,1]_plus>=0: p_74_plus[1] >= 0
 p_[74,2]_plus>=0: p_74_plus[2] >= 0
 s_75>=0: s_75 >= 0
 p_[75,0]_plus>=0: p_75_plus[0] >= 0
 p_[75,1]_plus>=0: p_75_plus[1] >= 0
 p_[75,2]_plus>=0: p_75_plus[2] >= 0
 s_76>=0: s_76 >= 0
 p_[76,0]_plus>=0: p_76_plus[0] >= 0
 p_[76,1]_plus>=0: p_76_plus[1] >= 0
 p_[76,2]_plus>=0: p_76_plus[2] >= 0
 s_77>=0: s_77 >= 0
 p_[77,0]_plus>=0: p_77_plus[0] >= 0
 p_[77,1]_plus>=0: p_77_plus[1] >= 0
 p_[77,2]_plus>=0: p_77_plus[2] >= 0
 s_78>=0: s_78 >= 0
 p_[78,0]_plus>=0: p_78_plus[0] >= 0
 p_[78,1]_plus>=0: p_78_plus[1] >= 0
 p_[78,2]_plus>=0: p_78_plus[2] >= 0
 s_79>=0: s_79 >= 0
 p_[79,0]_plus>=0: p_79_plus[0] >= 0
 p_[79,1]_plus>=0: p_79_plus[1] >= 0
 p_[79,2]_plus>=0: p_79_plus[2] >= 0
 s_80>=0: s_80 >= 0
 p_[80,0]_plus>=0: p_80_plus[0] >= 0
 p_[80,1]_plus>=0: p_80_plus[1] >= 0
 p_[80,2]_plus>=0: p_80_plus[2] >= 0
 s_81>=0: s_81 >= 0
 p_[81,0]_plus>=0: p_81_plus[0] >= 0
 p_[81,1]_plus>=0: p_81_plus[1] >= 0
 p_[81,2]_plus>=0: p_81_plus[2] >= 0
 s_82>=0: s_82 >= 0
 p_[82,0]_plus>=0: p_82_plus[0] >= 0
 p_[82,1]_plus>=0: p_82_plus[1] >= 0
 p_[82,2]_plus>=0: p_82_plus[2] >= 0
 s_83>=0: s_83 >= 0
 p_[83,0]_plus>=0: p_83_plus[0] >= 0
 p_[83,1]_plus>=0: p_83_plus[1] >= 0
 p_[83,2]_plus>=0: p_83_plus[2] >= 0
 s_84>=0: s_84 >= 0
 p_[84,0]_plus>=0: p_84_plus[0] >= 0
 p_[84,1]_plus>=0: p_84_plus[1] >= 0
 p_[84,2]_plus>=0: p_84_plus[2] >= 0
 s_85>=0: s_85 >= 0
 p_[85,0]_plus>=0: p_85_plus[0] >= 0
 p_[85,1]_plus>=0: p_85_plus[1] >= 0
 p_[85,2]_plus>=0: p_85_plus[2] >= 0
 s_86>=0: s_86 >= 0
 p_[86,0]_plus>=0: p_86_plus[0] >= 0
 p_[86,1]_plus>=0: p_86_plus[1] >= 0
 p_[86,2]_plus>=0: p_86_plus[2] >= 0
 s_87>=0: s_87 >= 0
 p_[87,0]_plus>=0: p_87_plus[0] >= 0
 p_[87,1]_plus>=0: p_87_plus[1] >= 0
 p_[87,2]_plus>=0: p_87_plus[2] >= 0
 s_88>=0: s_88 >= 0
 p_[88,0]_plus>=0: p_88_plus[0] >= 0
 p_[88,1]_plus>=0: p_88_plus[1] >= 0
 p_[88,2]_plus>=0: p_88_plus[2] >= 0
 s_89>=0: s_89 >= 0
 p_[89,0]_plus>=0: p_89_plus[0] >= 0
 p_[89,1]_plus>=0: p_89_plus[1] >= 0
 p_[89,2]_plus>=0: p_89_plus[2] >= 0
 s_90>=0: s_90 >= 0
 p_[90,0]_plus>=0: p_90_plus[0] >= 0
 p_[90,1]_plus>=0: p_90_plus[1] >= 0
 p_[90,2]_plus>=0: p_90_plus[2] >= 0
 s_91>=0: s_91 >= 0
 p_[91,0]_plus>=0: p_91_plus[0] >= 0
 p_[91,1]_plus>=0: p_91_plus[1] >= 0
 p_[91,2]_plus>=0: p_91_plus[2] >= 0
 s_92>=0: s_92 >= 0
 p_[92,0]_plus>=0: p_92_plus[0] >= 0
 p_[92,1]_plus>=0: p_92_plus[1] >= 0
 p_[92,2]_plus>=0: p_92_plus[2] >= 0
 s_93>=0: s_93 >= 0
 p_[93,0]_plus>=0: p_93_plus[0] >= 0
 p_[93,1]_plus>=0: p_93_plus[1] >= 0
 p_[93,2]_plus>=0: p_93_plus[2] >= 0
 s_94>=0: s_94 >= 0
 p_[94,0]_plus>=0: p_94_plus[0] >= 0
 p_[94,1]_plus>=0: p_94_plus[1] >= 0
 p_[94,2]_plus>=0: p_94_plus[2] >= 0
 s_95>=0: s_95 >= 0
 p_[95,0]_plus>=0: p_95_plus[0] >= 0
 p_[95,1]_plus>=0: p_95_plus[1] >= 0
 p_[95,2]_plus>=0: p_95_plus[2] >= 0
 s_96>=0: s_96 >= 0
 p_[96,0]_plus>=0: p_96_plus[0] >= 0
 p_[96,1]_plus>=0: p_96_plus[1] >= 0
 p_[96,2]_plus>=0: p_96_plus[2] >= 0
 s_97>=0: s_97 >= 0
 p_[97,0]_plus>=0: p_97_plus[0] >= 0
 p_[97,1]_plus>=0: p_97_plus[1] >= 0
 p_[97,2]_plus>=0: p_97_plus[2] >= 0
 s_98>=0: s_98 >= 0
 p_[98,0]_plus>=0: p_98_plus[0] >= 0
 p_[98,1]_plus>=0: p_98_plus[1] >= 0
 p_[98,2]_plus>=0: p_98_plus[2] >= 0
 s_99>=0: s_99 >= 0
 p_[99,0]_plus>=0: p_99_plus[0] >= 0
 p_[99,1]_plus>=0: p_99_plus[1] >= 0
 p_[99,2]_plus>=0: p_99_plus[2] >= 0
 qc0: [ w0 ^2 - 2 w0 * p_0_plus[0] + w1 ^2 - 2 w1 * p_0_plus[1] + w2 ^2
   - 2 w2 * p_0_plus[2] - Lambda ^2 + p_0_plus[0] ^2 + p_0_plus[1] ^2
   + p_0_plus[2] ^2 ] <= 0
 qc1: [ w0 ^2 - 2 w0 * p_1_plus[0] + w1 ^2 - 2 w1 * p_1_plus[1] + w2 ^2
   - 2 w2 * p_1_plus[2] - Lambda ^2 + p_1_plus[0] ^2 + p_1_plus[1] ^2
   + p_1_plus[2] ^2 ] <= 0
 qc2: [ w0 ^2 - 2 w0 * p_2_plus[0] + w1 ^2 - 2 w1 * p_2_plus[1] + w2 ^2
   - 2 w2 * p_2_plus[2] - Lambda ^2 + p_2_plus[0] ^2 + p_2_plus[1] ^2
   + p_2_plus[2] ^2 ] <= 0
 qc3: [ w0 ^2 - 2 w0 * p_3_plus[0] + w1 ^2 - 2 w1 * p_3_plus[1] + w2 ^2
   - 2 w2 * p_3_plus[2] - Lambda ^2 + p_3_plus[0] ^2 + p_3_plus[1] ^2
   + p_3_plus[2] ^2 ] <= 0
 qc4: [ w0 ^2 - 2 w0 * p_4_plus[0] + w1 ^2 - 2 w1 * p_4_plus[1] + w2 ^2
   - 2 w2 * p_4_plus[2] - Lambda ^2 + p_4_plus[0] ^2 + p_4_plus[1] ^2
   + p_4_plus[2] ^2 ] <= 0
 qc5: [ w0 ^2 - 2 w0 * p_5_plus[0] + w1 ^2 - 2 w1 * p_5_plus[1] + w2 ^2
   - 2 w2 * p_5_plus[2] - Lambda ^2 + p_5_plus[0] ^2 + p_5_plus[1] ^2
   + p_5_plus[2] ^2 ] <= 0
 qc6: [ w0 ^2 - 2 w0 * p_6_plus[0] + w1 ^2 - 2 w1 * p_6_plus[1] + w2 ^2
   - 2 w2 * p_6_plus[2] - Lambda ^2 + p_6_plus[0] ^2 + p_6_plus[1] ^2
   + p_6_plus[2] ^2 ] <= 0
 qc7: [ w0 ^2 - 2 w0 * p_7_plus[0] + w1 ^2 - 2 w1 * p_7_plus[1] + w2 ^2
   - 2 w2 * p_7_plus[2] - Lambda ^2 + p_7_plus[0] ^2 + p_7_plus[1] ^2
   + p_7_plus[2] ^2 ] <= 0
 qc8: [ w0 ^2 - 2 w0 * p_8_plus[0] + w1 ^2 - 2 w1 * p_8_plus[1] + w2 ^2
   - 2 w2 * p_8_plus[2] - Lambda ^2 + p_8_plus[0] ^2 + p_8_plus[1] ^2
   + p_8_plus[2] ^2 ] <= 0
 qc9: [ w0 ^2 - 2 w0 * p_9_plus[0] + w1 ^2 - 2 w1 * p_9_plus[1] + w2 ^2
   - 2 w2 * p_9_plus[2] - Lambda ^2 + p_9_plus[0] ^2 + p_9_plus[1] ^2
   + p_9_plus[2] ^2 ] <= 0
 qc10: [ w0 ^2 - 2 w0 * p_10_plus[0] + w1 ^2 - 2 w1 * p_10_plus[1] + w2 ^2
   - 2 w2 * p_10_plus[2] - Lambda ^2 + p_10_plus[0] ^2 + p_10_plus[1] ^2
   + p_10_plus[2] ^2 ] <= 0
 qc11: [ w0 ^2 - 2 w0 * p_11_plus[0] + w1 ^2 - 2 w1 * p_11_plus[1] + w2 ^2
   - 2 w2 * p_11_plus[2] - Lambda ^2 + p_11_plus[0] ^2 + p_11_plus[1] ^2
   + p_11_plus[2] ^2 ] <= 0
 qc12: [ w0 ^2 - 2 w0 * p_12_plus[0] + w1 ^2 - 2 w1 * p_12_plus[1] + w2 ^2
   - 2 w2 * p_12_plus[2] - Lambda ^2 + p_12_plus[0] ^2 + p_12_plus[1] ^2
   + p_12_plus[2] ^2 ] <= 0
 qc13: [ w0 ^2 - 2 w0 * p_13_plus[0] + w1 ^2 - 2 w1 * p_13_plus[1] + w2 ^2
   - 2 w2 * p_13_plus[2] - Lambda ^2 + p_13_plus[0] ^2 + p_13_plus[1] ^2
   + p_13_plus[2] ^2 ] <= 0
 qc14: [ w0 ^2 - 2 w0 * p_14_plus[0] + w1 ^2 - 2 w1 * p_14_plus[1] + w2 ^2
   - 2 w2 * p_14_plus[2] - Lambda ^2 + p_14_plus[0] ^2 + p_14_plus[1] ^2
   + p_14_plus[2] ^2 ] <= 0
 qc15: [ w0 ^2 - 2 w0 * p_15_plus[0] + w1 ^2 - 2 w1 * p_15_plus[1] + w2 ^2
   - 2 w2 * p_15_plus[2] - Lambda ^2 + p_15_plus[0] ^2 + p_15_plus[1] ^2
   + p_15_plus[2] ^2 ] <= 0
 qc16: [ w0 ^2 - 2 w0 * p_16_plus[0] + w1 ^2 - 2 w1 * p_16_plus[1] + w2 ^2
   - 2 w2 * p_16_plus[2] - Lambda ^2 + p_16_plus[0] ^2 + p_16_plus[1] ^2
   + p_16_plus[2] ^2 ] <= 0
 qc17: [ w0 ^2 - 2 w0 * p_17_plus[0] + w1 ^2 - 2 w1 * p_17_plus[1] + w2 ^2
   - 2 w2 * p_17_plus[2] - Lambda ^2 + p_17_plus[0] ^2 + p_17_plus[1] ^2
   + p_17_plus[2] ^2 ] <= 0
 qc18: [ w0 ^2 - 2 w0 * p_18_plus[0] + w1 ^2 - 2 w1 * p_18_plus[1] + w2 ^2
   - 2 w2 * p_18_plus[2] - Lambda ^2 + p_18_plus[0] ^2 + p_18_plus[1] ^2
   + p_18_plus[2] ^2 ] <= 0
 qc19: [ w0 ^2 - 2 w0 * p_19_plus[0] + w1 ^2 - 2 w1 * p_19_plus[1] + w2 ^2
   - 2 w2 * p_19_plus[2] - Lambda ^2 + p_19_plus[0] ^2 + p_19_plus[1] ^2
   + p_19_plus[2] ^2 ] <= 0
 qc20: [ w0 ^2 - 2 w0 * p_20_plus[0] + w1 ^2 - 2 w1 * p_20_plus[1] + w2 ^2
   - 2 w2 * p_20_plus[2] - Lambda ^2 + p_20_plus[0] ^2 + p_20_plus[1] ^2
   + p_20_plus[2] ^2 ] <= 0
 qc21: [ w0 ^2 - 2 w0 * p_21_plus[0] + w1 ^2 - 2 w1 * p_21_plus[1] + w2 ^2
   - 2 w2 * p_21_plus[2] - Lambda ^2 + p_21_plus[0] ^2 + p_21_plus[1] ^2
   + p_21_plus[2] ^2 ] <= 0
 qc22: [ w0 ^2 - 2 w0 * p_22_plus[0] + w1 ^2 - 2 w1 * p_22_plus[1] + w2 ^2
   - 2 w2 * p_22_plus[2] - Lambda ^2 + p_22_plus[0] ^2 + p_22_plus[1] ^2
   + p_22_plus[2] ^2 ] <= 0
 qc23: [ w0 ^2 - 2 w0 * p_23_plus[0] + w1 ^2 - 2 w1 * p_23_plus[1] + w2 ^2
   - 2 w2 * p_23_plus[2] - Lambda ^2 + p_23_plus[0] ^2 + p_23_plus[1] ^2
   + p_23_plus[2] ^2 ] <= 0
 qc24: [ w0 ^2 - 2 w0 * p_24_plus[0] + w1 ^2 - 2 w1 * p_24_plus[1] + w2 ^2
   - 2 w2 * p_24_plus[2] - Lambda ^2 + p_24_plus[0] ^2 + p_24_plus[1] ^2
   + p_24_plus[2] ^2 ] <= 0
 qc25: [ w0 ^2 - 2 w0 * p_25_plus[0] + w1 ^2 - 2 w1 * p_25_plus[1] + w2 ^2
   - 2 w2 * p_25_plus[2] - Lambda ^2 + p_25_plus[0] ^2 + p_25_plus[1] ^2
   + p_25_plus[2] ^2 ] <= 0
 qc26: [ w0 ^2 - 2 w0 * p_26_plus[0] + w1 ^2 - 2 w1 * p_26_plus[1] + w2 ^2
   - 2 w2 * p_26_plus[2] - Lambda ^2 + p_26_plus[0] ^2 + p_26_plus[1] ^2
   + p_26_plus[2] ^2 ] <= 0
 qc27: [ w0 ^2 - 2 w0 * p_27_plus[0] + w1 ^2 - 2 w1 * p_27_plus[1] + w2 ^2
   - 2 w2 * p_27_plus[2] - Lambda ^2 + p_27_plus[0] ^2 + p_27_plus[1] ^2
   + p_27_plus[2] ^2 ] <= 0
 qc28: [ w0 ^2 - 2 w0 * p_28_plus[0] + w1 ^2 - 2 w1 * p_28_plus[1] + w2 ^2
   - 2 w2 * p_28_plus[2] - Lambda ^2 + p_28_plus[0] ^2 + p_28_plus[1] ^2
   + p_28_plus[2] ^2 ] <= 0
 qc29: [ w0 ^2 - 2 w0 * p_29_plus[0] + w1 ^2 - 2 w1 * p_29_plus[1] + w2 ^2
   - 2 w2 * p_29_plus[2] - Lambda ^2 + p_29_plus[0] ^2 + p_29_plus[1] ^2
   + p_29_plus[2] ^2 ] <= 0
 qc30: [ w0 ^2 - 2 w0 * p_30_plus[0] + w1 ^2 - 2 w1 * p_30_plus[1] + w2 ^2
   - 2 w2 * p_30_plus[2] - Lambda ^2 + p_30_plus[0] ^2 + p_30_plus[1] ^2
   + p_30_plus[2] ^2 ] <= 0
 qc31: [ w0 ^2 - 2 w0 * p_31_plus[0] + w1 ^2 - 2 w1 * p_31_plus[1] + w2 ^2
   - 2 w2 * p_31_plus[2] - Lambda ^2 + p_31_plus[0] ^2 + p_31_plus[1] ^2
   + p_31_plus[2] ^2 ] <= 0
 qc32: [ w0 ^2 - 2 w0 * p_32_plus[0] + w1 ^2 - 2 w1 * p_32_plus[1] + w2 ^2
   - 2 w2 * p_32_plus[2] - Lambda ^2 + p_32_plus[0] ^2 + p_32_plus[1] ^2
   + p_32_plus[2] ^2 ] <= 0
 qc33: [ w0 ^2 - 2 w0 * p_33_plus[0] + w1 ^2 - 2 w1 * p_33_plus[1] + w2 ^2
   - 2 w2 * p_33_plus[2] - Lambda ^2 + p_33_plus[0] ^2 + p_33_plus[1] ^2
   + p_33_plus[2] ^2 ] <= 0
 qc34: [ w0 ^2 - 2 w0 * p_34_plus[0] + w1 ^2 - 2 w1 * p_34_plus[1] + w2 ^2
   - 2 w2 * p_34_plus[2] - Lambda ^2 + p_34_plus[0] ^2 + p_34_plus[1] ^2
   + p_34_plus[2] ^2 ] <= 0
 qc35: [ w0 ^2 - 2 w0 * p_35_plus[0] + w1 ^2 - 2 w1 * p_35_plus[1] + w2 ^2
   - 2 w2 * p_35_plus[2] - Lambda ^2 + p_35_plus[0] ^2 + p_35_plus[1] ^2
   + p_35_plus[2] ^2 ] <= 0
 qc36: [ w0 ^2 - 2 w0 * p_36_plus[0] + w1 ^2 - 2 w1 * p_36_plus[1] + w2 ^2
   - 2 w2 * p_36_plus[2] - Lambda ^2 + p_36_plus[0] ^2 + p_36_plus[1] ^2
   + p_36_plus[2] ^2 ] <= 0
 qc37: [ w0 ^2 - 2 w0 * p_37_plus[0] + w1 ^2 - 2 w1 * p_37_plus[1] + w2 ^2
   - 2 w2 * p_37_plus[2] - Lambda ^2 + p_37_plus[0] ^2 + p_37_plus[1] ^2
   + p_37_plus[2] ^2 ] <= 0
 qc38: [ w0 ^2 - 2 w0 * p_38_plus[0] + w1 ^2 - 2 w1 * p_38_plus[1] + w2 ^2
   - 2 w2 * p_38_plus[2] - Lambda ^2 + p_38_plus[0] ^2 + p_38_plus[1] ^2
   + p_38_plus[2] ^2 ] <= 0
 qc39: [ w0 ^2 - 2 w0 * p_39_plus[0] + w1 ^2 - 2 w1 * p_39_plus[1] + w2 ^2
   - 2 w2 * p_39_plus[2] - Lambda ^2 + p_39_plus[0] ^2 + p_39_plus[1] ^2
   + p_39_plus[2] ^2 ] <= 0
 qc40: [ w0 ^2 - 2 w0 * p_40_plus[0] + w1 ^2 - 2 w1 * p_40_plus[1] + w2 ^2
   - 2 w2 * p_40_plus[2] - Lambda ^2 + p_40_plus[0] ^2 + p_40_plus[1] ^2
   + p_40_plus[2] ^2 ] <= 0
 qc41: [ w0 ^2 - 2 w0 * p_41_plus[0] + w1 ^2 - 2 w1 * p_41_plus[1] + w2 ^2
   - 2 w2 * p_41_plus[2] - Lambda ^2 + p_41_plus[0] ^2 + p_41_plus[1] ^2
   + p_41_plus[2] ^2 ] <= 0
 qc42: [ w0 ^2 - 2 w0 * p_42_plus[0] + w1 ^2 - 2 w1 * p_42_plus[1] + w2 ^2
   - 2 w2 * p_42_plus[2] - Lambda ^2 + p_42_plus[0] ^2 + p_42_plus[1] ^2
   + p_42_plus[2] ^2 ] <= 0
 qc43: [ w0 ^2 - 2 w0 * p_43_plus[0] + w1 ^2 - 2 w1 * p_43_plus[1] + w2 ^2
   - 2 w2 * p_43_plus[2] - Lambda ^2 + p_43_plus[0] ^2 + p_43_plus[1] ^2
   + p_43_plus[2] ^2 ] <= 0
 qc44: [ w0 ^2 - 2 w0 * p_44_plus[0] + w1 ^2 - 2 w1 * p_44_plus[1] + w2 ^2
   - 2 w2 * p_44_plus[2] - Lambda ^2 + p_44_plus[0] ^2 + p_44_plus[1] ^2
   + p_44_plus[2] ^2 ] <= 0
 qc45: [ w0 ^2 - 2 w0 * p_45_plus[0] + w1 ^2 - 2 w1 * p_45_plus[1] + w2 ^2
   - 2 w2 * p_45_plus[2] - Lambda ^2 + p_45_plus[0] ^2 + p_45_plus[1] ^2
   + p_45_plus[2] ^2 ] <= 0
 qc46: [ w0 ^2 - 2 w0 * p_46_plus[0] + w1 ^2 - 2 w1 * p_46_plus[1] + w2 ^2
   - 2 w2 * p_46_plus[2] - Lambda ^2 + p_46_plus[0] ^2 + p_46_plus[1] ^2
   + p_46_plus[2] ^2 ] <= 0
 qc47: [ w0 ^2 - 2 w0 * p_47_plus[0] + w1 ^2 - 2 w1 * p_47_plus[1] + w2 ^2
   - 2 w2 * p_47_plus[2] - Lambda ^2 + p_47_plus[0] ^2 + p_47_plus[1] ^2
   + p_47_plus[2] ^2 ] <= 0
 qc48: [ w0 ^2 - 2 w0 * p_48_plus[0] + w1 ^2 - 2 w1 * p_48_plus[1] + w2 ^2
   - 2 w2 * p_48_plus[2] - Lambda ^2 + p_48_plus[0] ^2 + p_48_plus[1] ^2
   + p_48_plus[2] ^2 ] <= 0
 qc49: [ w0 ^2 - 2 w0 * p_49_plus[0] + w1 ^2 - 2 w1 * p_49_plus[1] + w2 ^2
   - 2 w2 * p_49_plus[2] - Lambda ^2 + p_49_plus[0] ^2 + p_49_plus[1] ^2
   + p_49_plus[2] ^2 ] <= 0
 qc50: [ w0 ^2 + 2 w0 * p_50_plus[0] + w1 ^2 + 2 w1 * p_50_plus[1] + w2 ^2
   + 2 w2 * p_50_plus[2] - Lambda ^2 + p_50_plus[0] ^2 + p_50_plus[1] ^2
   + p_50_plus[2] ^2 ] <= 0
 qc51: [ w0 ^2 + 2 w0 * p_51_plus[0] + w1 ^2 + 2 w1 * p_51_plus[1] + w2 ^2
   + 2 w2 * p_51_plus[2] - Lambda ^2 + p_51_plus[0] ^2 + p_51_plus[1] ^2
   + p_51_plus[2] ^2 ] <= 0
 qc52: [ w0 ^2 + 2 w0 * p_52_plus[0] + w1 ^2 + 2 w1 * p_52_plus[1] + w2 ^2
   + 2 w2 * p_52_plus[2] - Lambda ^2 + p_52_plus[0] ^2 + p_52_plus[1] ^2
   + p_52_plus[2] ^2 ] <= 0
 qc53: [ w0 ^2 + 2 w0 * p_53_plus[0] + w1 ^2 + 2 w1 * p_53_plus[1] + w2 ^2
   + 2 w2 * p_53_plus[2] - Lambda ^2 + p_53_plus[0] ^2 + p_53_plus[1] ^2
   + p_53_plus[2] ^2 ] <= 0
 qc54: [ w0 ^2 + 2 w0 * p_54_plus[0] + w1 ^2 + 2 w1 * p_54_plus[1] + w2 ^2
   + 2 w2 * p_54_plus[2] - Lambda ^2 + p_54_plus[0] ^2 + p_54_plus[1] ^2
   + p_54_plus[2] ^2 ] <= 0
 qc55: [ w0 ^2 + 2 w0 * p_55_plus[0] + w1 ^2 + 2 w1 * p_55_plus[1] + w2 ^2
   + 2 w2 * p_55_plus[2] - Lambda ^2 + p_55_plus[0] ^2 + p_55_plus[1] ^2
   + p_55_plus[2] ^2 ] <= 0
 qc56: [ w0 ^2 + 2 w0 * p_56_plus[0] + w1 ^2 + 2 w1 * p_56_plus[1] + w2 ^2
   + 2 w2 * p_56_plus[2] - Lambda ^2 + p_56_plus[0] ^2 + p_56_plus[1] ^2
   + p_56_plus[2] ^2 ] <= 0
 qc57: [ w0 ^2 + 2 w0 * p_57_plus[0] + w1 ^2 + 2 w1 * p_57_plus[1] + w2 ^2
   + 2 w2 * p_57_plus[2] - Lambda ^2 + p_57_plus[0] ^2 + p_57_plus[1] ^2
   + p_57_plus[2] ^2 ] <= 0
 qc58: [ w0 ^2 + 2 w0 * p_58_plus[0] + w1 ^2 + 2 w1 * p_58_plus[1] + w2 ^2
   + 2 w2 * p_58_plus[2] - Lambda ^2 + p_58_plus[0] ^2 + p_58_plus[1] ^2
   + p_58_plus[2] ^2 ] <= 0
 qc59: [ w0 ^2 + 2 w0 * p_59_plus[0] + w1 ^2 + 2 w1 * p_59_plus[1] + w2 ^2
   + 2 w2 * p_59_plus[2] - Lambda ^2 + p_59_plus[0] ^2 + p_59_plus[1] ^2
   + p_59_plus[2] ^2 ] <= 0
 qc60: [ w0 ^2 + 2 w0 * p_60_plus[0] + w1 ^2 + 2 w1 * p_60_plus[1] + w2 ^2
   + 2 w2 * p_60_plus[2] - Lambda ^2 + p_60_plus[0] ^2 + p_60_plus[1] ^2
   + p_60_plus[2] ^2 ] <= 0
 qc61: [ w0 ^2 + 2 w0 * p_61_plus[0] + w1 ^2 + 2 w1 * p_61_plus[1] + w2 ^2
   + 2 w2 * p_61_plus[2] - Lambda ^2 + p_61_plus[0] ^2 + p_61_plus[1] ^2
   + p_61_plus[2] ^2 ] <= 0
 qc62: [ w0 ^2 + 2 w0 * p_62_plus[0] + w1 ^2 + 2 w1 * p_62_plus[1] + w2 ^2
   + 2 w2 * p_62_plus[2] - Lambda ^2 + p_62_plus[0] ^2 + p_62_plus[1] ^2
   + p_62_plus[2] ^2 ] <= 0
 qc63: [ w0 ^2 + 2 w0 * p_63_plus[0] + w1 ^2 + 2 w1 * p_63_plus[1] + w2 ^2
   + 2 w2 * p_63_plus[2] - Lambda ^2 + p_63_plus[0] ^2 + p_63_plus[1] ^2
   + p_63_plus[2] ^2 ] <= 0
 qc64: [ w0 ^2 + 2 w0 * p_64_plus[0] + w1 ^2 + 2 w1 * p_64_plus[1] + w2 ^2
   + 2 w2 * p_64_plus[2] - Lambda ^2 + p_64_plus[0] ^2 + p_64_plus[1] ^2
   + p_64_plus[2] ^2 ] <= 0
 qc65: [ w0 ^2 + 2 w0 * p_65_plus[0] + w1 ^2 + 2 w1 * p_65_plus[1] + w2 ^2
   + 2 w2 * p_65_plus[2] - Lambda ^2 + p_65_plus[0] ^2 + p_65_plus[1] ^2
   + p_65_plus[2] ^2 ] <= 0
 qc66: [ w0 ^2 + 2 w0 * p_66_plus[0] + w1 ^2 + 2 w1 * p_66_plus[1] + w2 ^2
   + 2 w2 * p_66_plus[2] - Lambda ^2 + p_66_plus[0] ^2 + p_66_plus[1] ^2
   + p_66_plus[2] ^2 ] <= 0
 qc67: [ w0 ^2 + 2 w0 * p_67_plus[0] + w1 ^2 + 2 w1 * p_67_plus[1] + w2 ^2
   + 2 w2 * p_67_plus[2] - Lambda ^2 + p_67_plus[0] ^2 + p_67_plus[1] ^2
   + p_67_plus[2] ^2 ] <= 0
 qc68: [ w0 ^2 + 2 w0 * p_68_plus[0] + w1 ^2 + 2 w1 * p_68_plus[1] + w2 ^2
   + 2 w2 * p_68_plus[2] - Lambda ^2 + p_68_plus[0] ^2 + p_68_plus[1] ^2
   + p_68_plus[2] ^2 ] <= 0
 qc69: [ w0 ^2 + 2 w0 * p_69_plus[0] + w1 ^2 + 2 w1 * p_69_plus[1] + w2 ^2
   + 2 w2 * p_69_plus[2] - Lambda ^2 + p_69_plus[0] ^2 + p_69_plus[1] ^2
   + p_69_plus[2] ^2 ] <= 0
 qc70: [ w0 ^2 + 2 w0 * p_70_plus[0] + w1 ^2 + 2 w1 * p_70_plus[1] + w2 ^2
   + 2 w2 * p_70_plus[2] - Lambda ^2 + p_70_plus[0] ^2 + p_70_plus[1] ^2
   + p_70_plus[2] ^2 ] <= 0
 qc71: [ w0 ^2 + 2 w0 * p_71_plus[0] + w1 ^2 + 2 w1 * p_71_plus[1] + w2 ^2
   + 2 w2 * p_71_plus[2] - Lambda ^2 + p_71_plus[0] ^2 + p_71_plus[1] ^2
   + p_71_plus[2] ^2 ] <= 0
 qc72: [ w0 ^2 + 2 w0 * p_72_plus[0] + w1 ^2 + 2 w1 * p_72_plus[1] + w2 ^2
   + 2 w2 * p_72_plus[2] - Lambda ^2 + p_72_plus[0] ^2 + p_72_plus[1] ^2
   + p_72_plus[2] ^2 ] <= 0
 qc73: [ w0 ^2 + 2 w0 * p_73_plus[0] + w1 ^2 + 2 w1 * p_73_plus[1] + w2 ^2
   + 2 w2 * p_73_plus[2] - Lambda ^2 + p_73_plus[0] ^2 + p_73_plus[1] ^2
   + p_73_plus[2] ^2 ] <= 0
 qc74: [ w0 ^2 + 2 w0 * p_74_plus[0] + w1 ^2 + 2 w1 * p_74_plus[1] + w2 ^2
   + 2 w2 * p_74_plus[2] - Lambda ^2 + p_74_plus[0] ^2 + p_74_plus[1] ^2
   + p_74_plus[2] ^2 ] <= 0
 qc75: [ w0 ^2 + 2 w0 * p_75_plus[0] + w1 ^2 + 2 w1 * p_75_plus[1] + w2 ^2
   + 2 w2 * p_75_plus[2] - Lambda ^2 + p_75_plus[0] ^2 + p_75_plus[1] ^2
   + p_75_plus[2] ^2 ] <= 0
 qc76: [ w0 ^2 + 2 w0 * p_76_plus[0] + w1 ^2 + 2 w1 * p_76_plus[1] + w2 ^2
   + 2 w2 * p_76_plus[2] - Lambda ^2 + p_76_plus[0] ^2 + p_76_plus[1] ^2
   + p_76_plus[2] ^2 ] <= 0
 qc77: [ w0 ^2 + 2 w0 * p_77_plus[0] + w1 ^2 + 2 w1 * p_77_plus[1] + w2 ^2
   + 2 w2 * p_77_plus[2] - Lambda ^2 + p_77_plus[0] ^2 + p_77_plus[1] ^2
   + p_77_plus[2] ^2 ] <= 0
 qc78: [ w0 ^2 + 2 w0 * p_78_plus[0] + w1 ^2 + 2 w1 * p_78_plus[1] + w2 ^2
   + 2 w2 * p_78_plus[2] - Lambda ^2 + p_78_plus[0] ^2 + p_78_plus[1] ^2
   + p_78_plus[2] ^2 ] <= 0
 qc79: [ w0 ^2 + 2 w0 * p_79_plus[0] + w1 ^2 + 2 w1 * p_79_plus[1] + w2 ^2
   + 2 w2 * p_79_plus[2] - Lambda ^2 + p_79_plus[0] ^2 + p_79_plus[1] ^2
   + p_79_plus[2] ^2 ] <= 0
 qc80: [ w0 ^2 + 2 w0 * p_80_plus[0] + w1 ^2 + 2 w1 * p_80_plus[1] + w2 ^2
   + 2 w2 * p_80_plus[2] - Lambda ^2 + p_80_plus[0] ^2 + p_80_plus[1] ^2
   + p_80_plus[2] ^2 ] <= 0
 qc81: [ w0 ^2 + 2 w0 * p_81_plus[0] + w1 ^2 + 2 w1 * p_81_plus[1] + w2 ^2
   + 2 w2 * p_81_plus[2] - Lambda ^2 + p_81_plus[0] ^2 + p_81_plus[1] ^2
   + p_81_plus[2] ^2 ] <= 0
 qc82: [ w0 ^2 + 2 w0 * p_82_plus[0] + w1 ^2 + 2 w1 * p_82_plus[1] + w2 ^2
   + 2 w2 * p_82_plus[2] - Lambda ^2 + p_82_plus[0] ^2 + p_82_plus[1] ^2
   + p_82_plus[2] ^2 ] <= 0
 qc83: [ w0 ^2 + 2 w0 * p_83_plus[0] + w1 ^2 + 2 w1 * p_83_plus[1] + w2 ^2
   + 2 w2 * p_83_plus[2] - Lambda ^2 + p_83_plus[0] ^2 + p_83_plus[1] ^2
   + p_83_plus[2] ^2 ] <= 0
 qc84: [ w0 ^2 + 2 w0 * p_84_plus[0] + w1 ^2 + 2 w1 * p_84_plus[1] + w2 ^2
   + 2 w2 * p_84_plus[2] - Lambda ^2 + p_84_plus[0] ^2 + p_84_plus[1] ^2
   + p_84_plus[2] ^2 ] <= 0
 qc85: [ w0 ^2 + 2 w0 * p_85_plus[0] + w1 ^2 + 2 w1 * p_85_plus[1] + w2 ^2
   + 2 w2 * p_85_plus[2] - Lambda ^2 + p_85_plus[0] ^2 + p_85_plus[1] ^2
   + p_85_plus[2] ^2 ] <= 0
 qc86: [ w0 ^2 + 2 w0 * p_86_plus[0] + w1 ^2 + 2 w1 * p_86_plus[1] + w2 ^2
   + 2 w2 * p_86_plus[2] - Lambda ^2 + p_86_plus[0] ^2 + p_86_plus[1] ^2
   + p_86_plus[2] ^2 ] <= 0
 qc87: [ w0 ^2 + 2 w0 * p_87_plus[0] + w1 ^2 + 2 w1 * p_87_plus[1] + w2 ^2
   + 2 w2 * p_87_plus[2] - Lambda ^2 + p_87_plus[0] ^2 + p_87_plus[1] ^2
   + p_87_plus[2] ^2 ] <= 0
 qc88: [ w0 ^2 + 2 w0 * p_88_plus[0] + w1 ^2 + 2 w1 * p_88_plus[1] + w2 ^2
   + 2 w2 * p_88_plus[2] - Lambda ^2 + p_88_plus[0] ^2 + p_88_plus[1] ^2
   + p_88_plus[2] ^2 ] <= 0
 qc89: [ w0 ^2 + 2 w0 * p_89_plus[0] + w1 ^2 + 2 w1 * p_89_plus[1] + w2 ^2
   + 2 w2 * p_89_plus[2] - Lambda ^2 + p_89_plus[0] ^2 + p_89_plus[1] ^2
   + p_89_plus[2] ^2 ] <= 0
 qc90: [ w0 ^2 + 2 w0 * p_90_plus[0] + w1 ^2 + 2 w1 * p_90_plus[1] + w2 ^2
   + 2 w2 * p_90_plus[2] - Lambda ^2 + p_90_plus[0] ^2 + p_90_plus[1] ^2
   + p_90_plus[2] ^2 ] <= 0
 qc91: [ w0 ^2 + 2 w0 * p_91_plus[0] + w1 ^2 + 2 w1 * p_91_plus[1] + w2 ^2
   + 2 w2 * p_91_plus[2] - Lambda ^2 + p_91_plus[0] ^2 + p_91_plus[1] ^2
   + p_91_plus[2] ^2 ] <= 0
 qc92: [ w0 ^2 + 2 w0 * p_92_plus[0] + w1 ^2 + 2 w1 * p_92_plus[1] + w2 ^2
   + 2 w2 * p_92_plus[2] - Lambda ^2 + p_92_plus[0] ^2 + p_92_plus[1] ^2
   + p_92_plus[2] ^2 ] <= 0
 qc93: [ w0 ^2 + 2 w0 * p_93_plus[0] + w1 ^2 + 2 w1 * p_93_plus[1] + w2 ^2
   + 2 w2 * p_93_plus[2] - Lambda ^2 + p_93_plus[0] ^2 + p_93_plus[1] ^2
   + p_93_plus[2] ^2 ] <= 0
 qc94: [ w0 ^2 + 2 w0 * p_94_plus[0] + w1 ^2 + 2 w1 * p_94_plus[1] + w2 ^2
   + 2 w2 * p_94_plus[2] - Lambda ^2 + p_94_plus[0] ^2 + p_94_plus[1] ^2
   + p_94_plus[2] ^2 ] <= 0
 qc95: [ w0 ^2 + 2 w0 * p_95_plus[0] + w1 ^2 + 2 w1 * p_95_plus[1] + w2 ^2
   + 2 w2 * p_95_plus[2] - Lambda ^2 + p_95_plus[0] ^2 + p_95_plus[1] ^2
   + p_95_plus[2] ^2 ] <= 0
 qc96: [ w0 ^2 + 2 w0 * p_96_plus[0] + w1 ^2 + 2 w1 * p_96_plus[1] + w2 ^2
   + 2 w2 * p_96_plus[2] - Lambda ^2 + p_96_plus[0] ^2 + p_96_plus[1] ^2
   + p_96_plus[2] ^2 ] <= 0
 qc97: [ w0 ^2 + 2 w0 * p_97_plus[0] + w1 ^2 + 2 w1 * p_97_plus[1] + w2 ^2
   + 2 w2 * p_97_plus[2] - Lambda ^2 + p_97_plus[0] ^2 + p_97_plus[1] ^2
   + p_97_plus[2] ^2 ] <= 0
 qc98: [ w0 ^2 + 2 w0 * p_98_plus[0] + w1 ^2 + 2 w1 * p_98_plus[1] + w2 ^2
   + 2 w2 * p_98_plus[2] - Lambda ^2 + p_98_plus[0] ^2 + p_98_plus[1] ^2
   + p_98_plus[2] ^2 ] <= 0
 qc99: [ w0 ^2 + 2 w0 * p_99_plus[0] + w1 ^2 + 2 w1 * p_99_plus[1] + w2 ^2
   + 2 w2 * p_99_plus[2] - Lambda ^2 + p_99_plus[0] ^2 + p_99_plus[1] ^2
   + p_99_plus[2] ^2 ] <= 0
Bounds
 w0 free
 w1 free
 w2 free
End
