def generate_palindromic_sentence(cores, fillers, depth=0, max_depth=3):
    # Base case: pick a palindromic core (like 'aba', 'non', 'level')
    if depth == max_depth:
        return random.choice(cores)

    core = random.choice(cores)
    filler = random.choice(fillers)
    inner = generate_palindromic_sentence(cores, fillers, depth + 1)
    
    candidate = core + " " + filler + " " + inner + " " + filler[::-1] + " " + core[::-1]
    
    if is_valid_english(candidate):  # use language model or dictionary filter
        return candidate
    else:
        return generate_palindromic_sentence(cores, fillers, depth)  # retry
