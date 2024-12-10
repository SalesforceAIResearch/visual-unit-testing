from transformers import AutoTokenizer

def get_unit_test_prompt(batch, system_prompt, in_context_examples, model_type, tokenizer=None, program=''):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    if not isinstance(program, list):
        program= ['']*len(batch)
    batch = [tokenizer.bos_token + system_prompt+' '+in_context_examples.replace('INSERT_QUERY_HERE', s).replace('INSERT_PROGRAM_HERE', program[i]) for i,s in enumerate(batch)]
    return batch

def extract_unit_tests(text):
    if isinstance(text, str):
        text = [text]
    tests = []        
    for t in text:
        try:
            test_text = t.split('1.')[1]
        except:
            continue
        tests_raw = test_text.split("\n")
        tests_raw = list(set([t for t in tests_raw if 'Answer:' in t and ('Image Caption:' in t or 'Google Search Query:' in t)]))
        tests_raw = [t.strip() for t in tests_raw]
        if 'Google Search Query:' in tests_raw[0]:
            tests.extend([(l.split('Google Search Query: ')[1].split('Answer:')[0].strip(), l.split('Answer: ')[-1].strip()) for l in tests_raw])
        else:
            tests.extend([(l.split('Image Caption: ')[1].split('Answer:')[0].strip(), l.split('Answer: ')[-1].strip()) for l in tests_raw])
    return tests

def get_grounded_diffusion_prompt(batch, system_prompt, in_context_examples, model_type, tokenizer=None):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    batch = [tokenizer.bos_token + system_prompt+' '+in_context_examples.replace('{prompt}', s) for s in batch]
    return batch

def get_phrase_indices(
        pipe,
        prompt,
        phrases,
        token_map=None,
        add_suffix_if_not_found=False,
        verbose=False,
    ):
        for obj in phrases:
            # Suffix the prompt with object name for attention guidance if object is not in the prompt, using "|" to separate the prompt and the suffix
            if obj not in prompt:
                prompt += "| " + obj

        if token_map is None:
            # We allow using a pre-computed token map.
            token_map = pipe.get_token_map(prompt=prompt, padding="do_not_pad", verbose=verbose)
        token_map_str = " ".join(token_map)

        phrase_indices = []

        for obj in phrases:
            phrase_token_map = pipe.get_token_map(prompt=obj, padding="do_not_pad", verbose=verbose)
            # Remove <bos> and <eos> in substr
            phrase_token_map = phrase_token_map[1:-1]
            phrase_token_map_len = len(phrase_token_map)
            phrase_token_map_str = " ".join(phrase_token_map)

            # Count the number of token before substr
            # The substring comes with a trailing space that needs to be removed by minus one in the index.
            obj_first_index = len(token_map_str[: token_map_str.index(phrase_token_map_str) - 1].split(" "))

            obj_position = list(range(obj_first_index, obj_first_index + phrase_token_map_len))
            phrase_indices.append(obj_position)

        if add_suffix_if_not_found:
            return phrase_indices, prompt

        return phrase_indices