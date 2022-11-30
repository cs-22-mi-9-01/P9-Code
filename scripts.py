

def remove_unwanted_symbols(dict): 
    while True:
        target_key = None
        for key in dict.keys():
            if ' ' in key:
                target_key = key
                break
        if target_key is not None:
            dict[target_key.replace(' ', ' ')] = dict.pop(target_key)
        else:
            break