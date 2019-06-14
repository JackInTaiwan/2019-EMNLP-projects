from torch.utils.data import Dataset



class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(data) == len(labels))
        self.__len = len(data)


    def __len__(self):
        return self.__len


    def __getitem__(self, index):
        datum = self.data[index]
        label = self.labels[index]
        [question, idx_entity_s, idx_entity_e] = datum
        return (question, idx_entity_s, idx_entity_e), label



def produce_labels(fp_i, fp_o=None):
    """
    Read one .txt input file in logical form and do parsing work to produce and save
    one .txt file denoting all labels (logical relations) based on the input file.
    """

    output, count = set(), 0
    category_output = set()

    with open(fp_i) as f:
        for i, line in enumerate(f.readlines()):
            terms = line.strip().split(' ')
            if terms[0] == '<logical':
                for term in terms:
                    if len(term) > 4 and term[:4] == 'mso:':
                        label = term.split(':')[1]
                        category = label.split('.')[0]
                        category_output.add(category)
                        output.add(label)
                        count += 1
        
    output = sorted(list(output))

    if fp_o:
        with open(fp_o, 'w') as f:
            for item in output:
                f.write(item)
                f.write('\n')

        with open('categories.txt', 'w') as f:
            for item in category_output:
                f.write(item)
                f.write('\n')
    
    return output, category_output



def word_voc(fp, lines_limit=10000):
    with open(fp, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        data = {}
        for i, line in enumerate(f):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
            if i > lines_limit: break
    return data



def transform_data(fp_i, fp_o, fp_label, fp_cate):
    import numpy as np
    cate_table, label_table = {}, {}
    data = []
    # [question, idx_cate, idx_label, idx_entity_s, idx_entity_e]

    with open(fp_cate) as f:
        for i, line in enumerate(f.readlines()):
            cate_table[line.strip()] = i
    
    with open(fp_label) as f:
        for i, line in enumerate(f.readlines()):
            label_table[line.strip()] = i

    with open(fp_i) as f:
        for i, line in enumerate(f.readlines()):
            if i % 5 == 0:
                datum = []
                question = line.strip().split('>\t')[-1]
                datum.append(question)
            if i % 5 == 1:
                terms = line.strip().split(' ')
                if terms[0] == '<logical':
                    for term in terms:
                        if len(term) > 4 and term[:4] == 'mso:':
                            label = int(label_table[term.split(':')[1]])
                            category = int(cate_table[term.split(':')[1].split('.')[0]])
            if i % 5 == 2:
                idx = line.strip().split()[-1][1:-1]
                datum += list(map(int, idx.split(',')))
                datum += [category, label]
            if i % 5 == 3:
                q_type = line.strip().split()[-1]
                if q_type == 'single-relation':
                    data.append(datum)
    
    data = np.array(data, dtype=np.object)
    np.save(fp_o, data)

    return data



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-t', '--task', type=str, choices=['labels', 'transform'], required=True, help='task types')
    subparsers = parser.add_subparsers()
    
    sub_parser_lables = subparsers.add_parser('labels')
    sub_parser_lables.add_argument('-f', '--fp', type=str, help='file path')
    sub_parser_lables.add_argument('-o', '--output', type=str, help='output file path')
    
    sub_parser_transform = subparsers.add_parser('transform')
    sub_parser_transform.add_argument('-f', '--fp', type=str, help='file path')
    sub_parser_transform.add_argument('-o', '--output', type=str, help='output file path')
    sub_parser_transform.add_argument('-fl', '--f-label', type=str, help='label file path')
    sub_parser_transform.add_argument('-fc', '--f-cate', type=str, help='category file path')

    args = parser.parse_args()
    vars().update(vars(args))


    if task == 'labels':
        produce_labels(fp, output)
    
    elif task == 'transform':
        transform_data(fp, output, f_label, f_cate)