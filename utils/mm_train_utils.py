import random 

def _get_next_sentence(sent, next_sent, all_sents):
    if random.random() < 0.5:
        is_next = True 
    else:
        next_sent = random.choice(random.choice(all_sents))
        next_sent = False

    return sent, next_sent, is_next

def _get_nsp_data():
    pass

def _generate_mlm_tokens():
    pass


