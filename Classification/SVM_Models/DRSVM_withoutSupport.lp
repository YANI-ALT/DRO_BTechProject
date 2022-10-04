\ Model DRSVM_withoutSupport
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
 data[0]_constr2: - w0 - 5.1 w1 - 3.5 w2 - 0.1 Lambda - s_0 <= -1
 data[1]_constr1: w0 + 4.9 w1 + 3 w2 - s_1 <= -1
 data[1]_constr2: - w0 - 4.9 w1 - 3 w2 - 0.1 Lambda - s_1 <= -1
 data[2]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_2 <= -1
 data[2]_constr2: - w0 - 4.7 w1 - 3.2 w2 - 0.1 Lambda - s_2 <= -1
 data[3]_constr1: w0 + 4.6 w1 + 3.1 w2 - s_3 <= -1
 data[3]_constr2: - w0 - 4.6 w1 - 3.1 w2 - 0.1 Lambda - s_3 <= -1
 data[4]_constr1: w0 + 5 w1 + 3.6 w2 - s_4 <= -1
 data[4]_constr2: - w0 - 5 w1 - 3.6 w2 - 0.1 Lambda - s_4 <= -1
 data[5]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_5 <= -1
 data[5]_constr2: - w0 - 5.4 w1 - 3.9 w2 - 0.1 Lambda - s_5 <= -1
 data[6]_constr1: w0 + 4.6 w1 + 3.4 w2 - s_6 <= -1
 data[6]_constr2: - w0 - 4.6 w1 - 3.4 w2 - 0.1 Lambda - s_6 <= -1
 data[7]_constr1: w0 + 5 w1 + 3.4 w2 - s_7 <= -1
 data[7]_constr2: - w0 - 5 w1 - 3.4 w2 - 0.1 Lambda - s_7 <= -1
 data[8]_constr1: w0 + 4.4 w1 + 2.9 w2 - s_8 <= -1
 data[8]_constr2: - w0 - 4.4 w1 - 2.9 w2 - 0.1 Lambda - s_8 <= -1
 data[9]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_9 <= -1
 data[9]_constr2: - w0 - 4.9 w1 - 3.1 w2 - 0.1 Lambda - s_9 <= -1
 data[10]_constr1: w0 + 5.4 w1 + 3.7 w2 - s_10 <= -1
 data[10]_constr2: - w0 - 5.4 w1 - 3.7 w2 - 0.1 Lambda - s_10 <= -1
 data[11]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_11 <= -1
 data[11]_constr2: - w0 - 4.8 w1 - 3.4 w2 - 0.1 Lambda - s_11 <= -1
 data[12]_constr1: w0 + 4.8 w1 + 3 w2 - s_12 <= -1
 data[12]_constr2: - w0 - 4.8 w1 - 3 w2 - 0.1 Lambda - s_12 <= -1
 data[13]_constr1: w0 + 4.3 w1 + 3 w2 - s_13 <= -1
 data[13]_constr2: - w0 - 4.3 w1 - 3 w2 - 0.1 Lambda - s_13 <= -1
 data[14]_constr1: w0 + 5.8 w1 + 4 w2 - s_14 <= -1
 data[14]_constr2: - w0 - 5.8 w1 - 4 w2 - 0.1 Lambda - s_14 <= -1
 data[15]_constr1: w0 + 5.7 w1 + 4.4 w2 - s_15 <= -1
 data[15]_constr2: - w0 - 5.7 w1 - 4.4 w2 - 0.1 Lambda - s_15 <= -1
 data[16]_constr1: w0 + 5.4 w1 + 3.9 w2 - s_16 <= -1
 data[16]_constr2: - w0 - 5.4 w1 - 3.9 w2 - 0.1 Lambda - s_16 <= -1
 data[17]_constr1: w0 + 5.1 w1 + 3.5 w2 - s_17 <= -1
 data[17]_constr2: - w0 - 5.1 w1 - 3.5 w2 - 0.1 Lambda - s_17 <= -1
 data[18]_constr1: w0 + 5.7 w1 + 3.8 w2 - s_18 <= -1
 data[18]_constr2: - w0 - 5.7 w1 - 3.8 w2 - 0.1 Lambda - s_18 <= -1
 data[19]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_19 <= -1
 data[19]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_19 <= -1
 data[20]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_20 <= -1
 data[20]_constr2: - w0 - 5.4 w1 - 3.4 w2 - 0.1 Lambda - s_20 <= -1
 data[21]_constr1: w0 + 5.1 w1 + 3.7 w2 - s_21 <= -1
 data[21]_constr2: - w0 - 5.1 w1 - 3.7 w2 - 0.1 Lambda - s_21 <= -1
 data[22]_constr1: w0 + 4.6 w1 + 3.6 w2 - s_22 <= -1
 data[22]_constr2: - w0 - 4.6 w1 - 3.6 w2 - 0.1 Lambda - s_22 <= -1
 data[23]_constr1: w0 + 5.1 w1 + 3.3 w2 - s_23 <= -1
 data[23]_constr2: - w0 - 5.1 w1 - 3.3 w2 - 0.1 Lambda - s_23 <= -1
 data[24]_constr1: w0 + 4.8 w1 + 3.4 w2 - s_24 <= -1
 data[24]_constr2: - w0 - 4.8 w1 - 3.4 w2 - 0.1 Lambda - s_24 <= -1
 data[25]_constr1: w0 + 5 w1 + 3 w2 - s_25 <= -1
 data[25]_constr2: - w0 - 5 w1 - 3 w2 - 0.1 Lambda - s_25 <= -1
 data[26]_constr1: w0 + 5 w1 + 3.4 w2 - s_26 <= -1
 data[26]_constr2: - w0 - 5 w1 - 3.4 w2 - 0.1 Lambda - s_26 <= -1
 data[27]_constr1: w0 + 5.2 w1 + 3.5 w2 - s_27 <= -1
 data[27]_constr2: - w0 - 5.2 w1 - 3.5 w2 - 0.1 Lambda - s_27 <= -1
 data[28]_constr1: w0 + 5.2 w1 + 3.4 w2 - s_28 <= -1
 data[28]_constr2: - w0 - 5.2 w1 - 3.4 w2 - 0.1 Lambda - s_28 <= -1
 data[29]_constr1: w0 + 4.7 w1 + 3.2 w2 - s_29 <= -1
 data[29]_constr2: - w0 - 4.7 w1 - 3.2 w2 - 0.1 Lambda - s_29 <= -1
 data[30]_constr1: w0 + 4.8 w1 + 3.1 w2 - s_30 <= -1
 data[30]_constr2: - w0 - 4.8 w1 - 3.1 w2 - 0.1 Lambda - s_30 <= -1
 data[31]_constr1: w0 + 5.4 w1 + 3.4 w2 - s_31 <= -1
 data[31]_constr2: - w0 - 5.4 w1 - 3.4 w2 - 0.1 Lambda - s_31 <= -1
 data[32]_constr1: w0 + 5.2 w1 + 4.1 w2 - s_32 <= -1
 data[32]_constr2: - w0 - 5.2 w1 - 4.1 w2 - 0.1 Lambda - s_32 <= -1
 data[33]_constr1: w0 + 5.5 w1 + 4.2 w2 - s_33 <= -1
 data[33]_constr2: - w0 - 5.5 w1 - 4.2 w2 - 0.1 Lambda - s_33 <= -1
 data[34]_constr1: w0 + 4.9 w1 + 3.1 w2 - s_34 <= -1
 data[34]_constr2: - w0 - 4.9 w1 - 3.1 w2 - 0.1 Lambda - s_34 <= -1
 data[35]_constr1: w0 + 5 w1 + 3.2 w2 - s_35 <= -1
 data[35]_constr2: - w0 - 5 w1 - 3.2 w2 - 0.1 Lambda - s_35 <= -1
 data[36]_constr1: w0 + 5.5 w1 + 3.5 w2 - s_36 <= -1
 data[36]_constr2: - w0 - 5.5 w1 - 3.5 w2 - 0.1 Lambda - s_36 <= -1
 data[37]_constr1: w0 + 4.9 w1 + 3.6 w2 - s_37 <= -1
 data[37]_constr2: - w0 - 4.9 w1 - 3.6 w2 - 0.1 Lambda - s_37 <= -1
 data[38]_constr1: w0 + 4.4 w1 + 3 w2 - s_38 <= -1
 data[38]_constr2: - w0 - 4.4 w1 - 3 w2 - 0.1 Lambda - s_38 <= -1
 data[39]_constr1: w0 + 5.1 w1 + 3.4 w2 - s_39 <= -1
 data[39]_constr2: - w0 - 5.1 w1 - 3.4 w2 - 0.1 Lambda - s_39 <= -1
 data[40]_constr1: w0 + 5 w1 + 3.5 w2 - s_40 <= -1
 data[40]_constr2: - w0 - 5 w1 - 3.5 w2 - 0.1 Lambda - s_40 <= -1
 data[41]_constr1: w0 + 4.5 w1 + 2.3 w2 - s_41 <= -1
 data[41]_constr2: - w0 - 4.5 w1 - 2.3 w2 - 0.1 Lambda - s_41 <= -1
 data[42]_constr1: w0 + 4.4 w1 + 3.2 w2 - s_42 <= -1
 data[42]_constr2: - w0 - 4.4 w1 - 3.2 w2 - 0.1 Lambda - s_42 <= -1
 data[43]_constr1: w0 + 5 w1 + 3.5 w2 - s_43 <= -1
 data[43]_constr2: - w0 - 5 w1 - 3.5 w2 - 0.1 Lambda - s_43 <= -1
 data[44]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_44 <= -1
 data[44]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_44 <= -1
 data[45]_constr1: w0 + 4.8 w1 + 3 w2 - s_45 <= -1
 data[45]_constr2: - w0 - 4.8 w1 - 3 w2 - 0.1 Lambda - s_45 <= -1
 data[46]_constr1: w0 + 5.1 w1 + 3.8 w2 - s_46 <= -1
 data[46]_constr2: - w0 - 5.1 w1 - 3.8 w2 - 0.1 Lambda - s_46 <= -1
 data[47]_constr1: w0 + 4.6 w1 + 3.2 w2 - s_47 <= -1
 data[47]_constr2: - w0 - 4.6 w1 - 3.2 w2 - 0.1 Lambda - s_47 <= -1
 data[48]_constr1: w0 + 5.3 w1 + 3.7 w2 - s_48 <= -1
 data[48]_constr2: - w0 - 5.3 w1 - 3.7 w2 - 0.1 Lambda - s_48 <= -1
 data[49]_constr1: w0 + 5 w1 + 3.3 w2 - s_49 <= -1
 data[49]_constr2: - w0 - 5 w1 - 3.3 w2 - 0.1 Lambda - s_49 <= -1
 data[50]_constr1: - w0 - 6.3 w1 - 3.3 w2 - s_50 <= -1
 data[50]_constr2: w0 + 6.3 w1 + 3.3 w2 - 0.1 Lambda - s_50 <= -1
 data[51]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_51 <= -1
 data[51]_constr2: w0 + 5.8 w1 + 2.7 w2 - 0.1 Lambda - s_51 <= -1
 data[52]_constr1: - w0 - 7.1 w1 - 3 w2 - s_52 <= -1
 data[52]_constr2: w0 + 7.1 w1 + 3 w2 - 0.1 Lambda - s_52 <= -1
 data[53]_constr1: - w0 - 6.3 w1 - 2.9 w2 - s_53 <= -1
 data[53]_constr2: w0 + 6.3 w1 + 2.9 w2 - 0.1 Lambda - s_53 <= -1
 data[54]_constr1: - w0 - 6.5 w1 - 3 w2 - s_54 <= -1
 data[54]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_54 <= -1
 data[55]_constr1: - w0 - 7.6 w1 - 3 w2 - s_55 <= -1
 data[55]_constr2: w0 + 7.6 w1 + 3 w2 - 0.1 Lambda - s_55 <= -1
 data[56]_constr1: - w0 - 4.9 w1 - 2.5 w2 - s_56 <= -1
 data[56]_constr2: w0 + 4.9 w1 + 2.5 w2 - 0.1 Lambda - s_56 <= -1
 data[57]_constr1: - w0 - 7.3 w1 - 2.9 w2 - s_57 <= -1
 data[57]_constr2: w0 + 7.3 w1 + 2.9 w2 - 0.1 Lambda - s_57 <= -1
 data[58]_constr1: - w0 - 6.7 w1 - 2.5 w2 - s_58 <= -1
 data[58]_constr2: w0 + 6.7 w1 + 2.5 w2 - 0.1 Lambda - s_58 <= -1
 data[59]_constr1: - w0 - 7.2 w1 - 3.6 w2 - s_59 <= -1
 data[59]_constr2: w0 + 7.2 w1 + 3.6 w2 - 0.1 Lambda - s_59 <= -1
 data[60]_constr1: - w0 - 6.5 w1 - 3.2 w2 - s_60 <= -1
 data[60]_constr2: w0 + 6.5 w1 + 3.2 w2 - 0.1 Lambda - s_60 <= -1
 data[61]_constr1: - w0 - 6.4 w1 - 2.7 w2 - s_61 <= -1
 data[61]_constr2: w0 + 6.4 w1 + 2.7 w2 - 0.1 Lambda - s_61 <= -1
 data[62]_constr1: - w0 - 6.8 w1 - 3 w2 - s_62 <= -1
 data[62]_constr2: w0 + 6.8 w1 + 3 w2 - 0.1 Lambda - s_62 <= -1
 data[63]_constr1: - w0 - 5.7 w1 - 2.5 w2 - s_63 <= -1
 data[63]_constr2: w0 + 5.7 w1 + 2.5 w2 - 0.1 Lambda - s_63 <= -1
 data[64]_constr1: - w0 - 5.8 w1 - 2.8 w2 - s_64 <= -1
 data[64]_constr2: w0 + 5.8 w1 + 2.8 w2 - 0.1 Lambda - s_64 <= -1
 data[65]_constr1: - w0 - 6.4 w1 - 3.2 w2 - s_65 <= -1
 data[65]_constr2: w0 + 6.4 w1 + 3.2 w2 - 0.1 Lambda - s_65 <= -1
 data[66]_constr1: - w0 - 6.5 w1 - 3 w2 - s_66 <= -1
 data[66]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_66 <= -1
 data[67]_constr1: - w0 - 7.7 w1 - 3.8 w2 - s_67 <= -1
 data[67]_constr2: w0 + 7.7 w1 + 3.8 w2 - 0.1 Lambda - s_67 <= -1
 data[68]_constr1: - w0 - 7.7 w1 - 2.6 w2 - s_68 <= -1
 data[68]_constr2: w0 + 7.7 w1 + 2.6 w2 - 0.1 Lambda - s_68 <= -1
 data[69]_constr1: - w0 - 6 w1 - 2.2 w2 - s_69 <= -1
 data[69]_constr2: w0 + 6 w1 + 2.2 w2 - 0.1 Lambda - s_69 <= -1
 data[70]_constr1: - w0 - 6.9 w1 - 3.2 w2 - s_70 <= -1
 data[70]_constr2: w0 + 6.9 w1 + 3.2 w2 - 0.1 Lambda - s_70 <= -1
 data[71]_constr1: - w0 - 5.6 w1 - 2.8 w2 - s_71 <= -1
 data[71]_constr2: w0 + 5.6 w1 + 2.8 w2 - 0.1 Lambda - s_71 <= -1
 data[72]_constr1: - w0 - 7.7 w1 - 2.8 w2 - s_72 <= -1
 data[72]_constr2: w0 + 7.7 w1 + 2.8 w2 - 0.1 Lambda - s_72 <= -1
 data[73]_constr1: - w0 - 6.3 w1 - 2.7 w2 - s_73 <= -1
 data[73]_constr2: w0 + 6.3 w1 + 2.7 w2 - 0.1 Lambda - s_73 <= -1
 data[74]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_74 <= -1
 data[74]_constr2: w0 + 6.7 w1 + 3.3 w2 - 0.1 Lambda - s_74 <= -1
 data[75]_constr1: - w0 - 7.2 w1 - 3.2 w2 - s_75 <= -1
 data[75]_constr2: w0 + 7.2 w1 + 3.2 w2 - 0.1 Lambda - s_75 <= -1
 data[76]_constr1: - w0 - 6.2 w1 - 2.8 w2 - s_76 <= -1
 data[76]_constr2: w0 + 6.2 w1 + 2.8 w2 - 0.1 Lambda - s_76 <= -1
 data[77]_constr1: - w0 - 6.1 w1 - 3 w2 - s_77 <= -1
 data[77]_constr2: w0 + 6.1 w1 + 3 w2 - 0.1 Lambda - s_77 <= -1
 data[78]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_78 <= -1
 data[78]_constr2: w0 + 6.4 w1 + 2.8 w2 - 0.1 Lambda - s_78 <= -1
 data[79]_constr1: - w0 - 7.2 w1 - 3 w2 - s_79 <= -1
 data[79]_constr2: w0 + 7.2 w1 + 3 w2 - 0.1 Lambda - s_79 <= -1
 data[80]_constr1: - w0 - 7.4 w1 - 2.8 w2 - s_80 <= -1
 data[80]_constr2: w0 + 7.4 w1 + 2.8 w2 - 0.1 Lambda - s_80 <= -1
 data[81]_constr1: - w0 - 7.9 w1 - 3.8 w2 - s_81 <= -1
 data[81]_constr2: w0 + 7.9 w1 + 3.8 w2 - 0.1 Lambda - s_81 <= -1
 data[82]_constr1: - w0 - 6.4 w1 - 2.8 w2 - s_82 <= -1
 data[82]_constr2: w0 + 6.4 w1 + 2.8 w2 - 0.1 Lambda - s_82 <= -1
 data[83]_constr1: - w0 - 6.3 w1 - 2.8 w2 - s_83 <= -1
 data[83]_constr2: w0 + 6.3 w1 + 2.8 w2 - 0.1 Lambda - s_83 <= -1
 data[84]_constr1: - w0 - 6.1 w1 - 2.6 w2 - s_84 <= -1
 data[84]_constr2: w0 + 6.1 w1 + 2.6 w2 - 0.1 Lambda - s_84 <= -1
 data[85]_constr1: - w0 - 7.7 w1 - 3 w2 - s_85 <= -1
 data[85]_constr2: w0 + 7.7 w1 + 3 w2 - 0.1 Lambda - s_85 <= -1
 data[86]_constr1: - w0 - 6.3 w1 - 3.4 w2 - s_86 <= -1
 data[86]_constr2: w0 + 6.3 w1 + 3.4 w2 - 0.1 Lambda - s_86 <= -1
 data[87]_constr1: - w0 - 6.4 w1 - 3.1 w2 - s_87 <= -1
 data[87]_constr2: w0 + 6.4 w1 + 3.1 w2 - 0.1 Lambda - s_87 <= -1
 data[88]_constr1: - w0 - 6 w1 - 3 w2 - s_88 <= -1
 data[88]_constr2: w0 + 6 w1 + 3 w2 - 0.1 Lambda - s_88 <= -1
 data[89]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_89 <= -1
 data[89]_constr2: w0 + 6.9 w1 + 3.1 w2 - 0.1 Lambda - s_89 <= -1
 data[90]_constr1: - w0 - 6.7 w1 - 3.1 w2 - s_90 <= -1
 data[90]_constr2: w0 + 6.7 w1 + 3.1 w2 - 0.1 Lambda - s_90 <= -1
 data[91]_constr1: - w0 - 6.9 w1 - 3.1 w2 - s_91 <= -1
 data[91]_constr2: w0 + 6.9 w1 + 3.1 w2 - 0.1 Lambda - s_91 <= -1
 data[92]_constr1: - w0 - 5.8 w1 - 2.7 w2 - s_92 <= -1
 data[92]_constr2: w0 + 5.8 w1 + 2.7 w2 - 0.1 Lambda - s_92 <= -1
 data[93]_constr1: - w0 - 6.8 w1 - 3.2 w2 - s_93 <= -1
 data[93]_constr2: w0 + 6.8 w1 + 3.2 w2 - 0.1 Lambda - s_93 <= -1
 data[94]_constr1: - w0 - 6.7 w1 - 3.3 w2 - s_94 <= -1
 data[94]_constr2: w0 + 6.7 w1 + 3.3 w2 - 0.1 Lambda - s_94 <= -1
 data[95]_constr1: - w0 - 6.7 w1 - 3 w2 - s_95 <= -1
 data[95]_constr2: w0 + 6.7 w1 + 3 w2 - 0.1 Lambda - s_95 <= -1
 data[96]_constr1: - w0 - 6.3 w1 - 2.5 w2 - s_96 <= -1
 data[96]_constr2: w0 + 6.3 w1 + 2.5 w2 - 0.1 Lambda - s_96 <= -1
 data[97]_constr1: - w0 - 6.5 w1 - 3 w2 - s_97 <= -1
 data[97]_constr2: w0 + 6.5 w1 + 3 w2 - 0.1 Lambda - s_97 <= -1
 data[98]_constr1: - w0 - 6.2 w1 - 3.4 w2 - s_98 <= -1
 data[98]_constr2: w0 + 6.2 w1 + 3.4 w2 - 0.1 Lambda - s_98 <= -1
 data[99]_constr1: - w0 - 5.9 w1 - 3 w2 - s_99 <= -1
 data[99]_constr2: w0 + 5.9 w1 + 3 w2 - 0.1 Lambda - s_99 <= -1
 qc0: [ w0 ^2 + w1 ^2 + w2 ^2 - Lambda ^2 ] <= 0
Bounds
 w0 free
 w1 free
 w2 free
End
