#feedback = 1 (good) 0 (bad), COUNT = round number
def calculate_rep(feedback:int, eval_score:int, COUNT:int, numerator:int, denominator:int):
    combine_scores = ((feedback*2)+eval_score)/2
    if combine_scores>=5.5:
        alpha_i, beta_i = 1, 0 
    else:
        alpha_i, beta_i = 0, 1
    total_count = 5
    #let freshness k = 0.9
    # if feedback == 1:
    #     alpha_i, beta_i = 1, 0 
    # else:
    #     alpha_i, beta_i = 0, 1
    numerator += (alpha_i*0.9**(total_count - COUNT)) + 1
    denominator += ((alpha_i*0.9**(total_count - COUNT)) + (0.9**(total_count - COUNT)*beta_i)) + 2
    cur_reputation = numerator/denominator
    if COUNT > total_count:
        total_count += 5

    return cur_reputation, numerator, denominator