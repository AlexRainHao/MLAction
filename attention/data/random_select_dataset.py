from typing import Dict, List, Any, Optional
import random
import codecs

def load_origin_dataset(fname = "eng-fra.txt"):
    with codecs.open(fname, encoding = 'utf-8') as f:
        lines = f.readlines()
        
    return lines

def random_split(lines: List[str], N = 10000, ratio = 0.7):
    non_empty_lines = list(filter(lambda x: x.strip() != '', lines))
    random_index = random.choices(range(len(non_empty_lines)), k = N)
    
    train_lines = [non_empty_lines[idx] for idx in random_index[:int(N * ratio)]]
    test_lines = [non_empty_lines[idx] for idx in random_index[int(N * ratio):]]
    
    return train_lines, test_lines

def dumps_to_text(train_lines: List[str], test_lines: List[str]):
    with codecs.open("train.txt", 'w', encoding = 'utf-8') as f:
        f.writelines(train_lines)
        
    with codecs.open("test.txt", 'w', encoding = 'utf-8') as f:
        f.writelines(test_lines)
        
    print('finished')

# ori_lines = load_origin_dataset()
# train_lines, test_lines = random_split(ori_lines)
# dumps_to_text(train_lines, test_lines)