\ Model Logistic Regression
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Lambda + 0.25 s_0 + 0.25 s_1 + 0.25 s_2 + 0.25 s_3
Subject To
 exp_var_helper_constr_0: - w_0 - 5.3 w_1 - 3.7 w_2 + exp_var_helper_0 = 0
 log__constr_0: - exp_var_0 + log_helper_0 = 1
 constr1_0: - s_0 + log_var_0 <= 0
 exp_var_helper_constr2_0: w_0 + 5.3 w_1 + 3.7 w_2 + exp_var_helper2_0 = 0
 log__constr2_0: - exp_var2_0 + log_helper2_0 = 1
 constr2_0: - 0.01 Lambda - s_0 + log_var2_0 <= 0
 exp_var_helper_constr_1: - w_0 - 5 w_1 - 3.3 w_2 + exp_var_helper_1 = 0
 log__constr_1: - exp_var_1 + log_helper_1 = 1
 constr1_1: - s_1 + log_var_1 <= 0
 exp_var_helper_constr2_1: w_0 + 5 w_1 + 3.3 w_2 + exp_var_helper2_1 = 0
 log__constr2_1: - exp_var2_1 + log_helper2_1 = 1
 constr2_1: - 0.01 Lambda - s_1 + log_var2_1 <= 0
 exp_var_helper_constr_2: w_0 + 6.3 w_1 + 3.3 w_2 + exp_var_helper_2 = 0
 log__constr_2: - exp_var_2 + log_helper_2 = 1
 constr1_2: - s_2 + log_var_2 <= 0
 exp_var_helper_constr2_2: - w_0 - 6.3 w_1 - 3.3 w_2 + exp_var_helper2_2
   = 0
 log__constr2_2: - exp_var2_2 + log_helper2_2 = 1
 constr2_2: - 0.01 Lambda - s_2 + log_var2_2 <= 0
 exp_var_helper_constr_3: w_0 + 5.8 w_1 + 2.7 w_2 + exp_var_helper_3 = 0
 log__constr_3: - exp_var_3 + log_helper_3 = 1
 constr1_3: - s_3 + log_var_3 <= 0
 exp_var_helper_constr2_3: - w_0 - 5.8 w_1 - 2.7 w_2 + exp_var_helper2_3
   = 0
 log__constr2_3: - exp_var2_3 + log_helper2_3 = 1
 constr2_3: - 0.01 Lambda - s_3 + log_var2_3 <= 0
 qc0: [ w_0 ^2 + w_1 ^2 + w_2 ^2 - Lambda ^2 ] <= 0
Bounds
 w_0 free
 w_1 free
 w_2 free
 s_0 free
 s_1 free
 s_2 free
 s_3 free
General Constraints
 exp_var_constr_0: exp_var_0 = EXP ( exp_var_helper_0 )
 log_var_constr_0: log_var_0 = LOG ( log_helper_0 )
 exp_var_constr2_0: exp_var2_0 = EXP ( exp_var_helper2_0 )
 log_var_constr20: log_var2_0 = LOG ( log_helper2_0 )
 exp_var_constr_1: exp_var_1 = EXP ( exp_var_helper_1 )
 log_var_constr_1: log_var_1 = LOG ( log_helper_1 )
 exp_var_constr2_1: exp_var2_1 = EXP ( exp_var_helper2_1 )
 log_var_constr21: log_var2_1 = LOG ( log_helper2_1 )
 exp_var_constr_2: exp_var_2 = EXP ( exp_var_helper_2 )
 log_var_constr_2: log_var_2 = LOG ( log_helper_2 )
 exp_var_constr2_2: exp_var2_2 = EXP ( exp_var_helper2_2 )
 log_var_constr22: log_var2_2 = LOG ( log_helper2_2 )
 exp_var_constr_3: exp_var_3 = EXP ( exp_var_helper_3 )
 log_var_constr_3: log_var_3 = LOG ( log_helper_3 )
 exp_var_constr2_3: exp_var2_3 = EXP ( exp_var_helper2_3 )
 log_var_constr23: log_var2_3 = LOG ( log_helper2_3 )
End
