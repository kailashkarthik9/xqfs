import json
from os import path
from tqdm import tqdm

data_file = 'data/cnndm/rtt_qfs/valid.qfs.single.rtt.json'
data = json.load(open(data_file))

output_dir = 'data/cnndm/rtt_hqvist/test'
documents_dir = path.join(output_dir, 'documents')
entities_dir = path.join(output_dir, 'entities')
queries_dir = path.join(output_dir, 'queries')
references_dir = path.join(output_dir, 'references')

doc_id_map = dict()

for instance in tqdm(data):
    article = ' '.join(instance['article'])
    query = instance['query']
    summary = ' '.join(instance['summary'])
    doc_id, query_id = instance['id'].split('_')
    if doc_id not in doc_id_map:
        doc_id_map[doc_id] = len(doc_id_map) + 1
        with open(path.join(documents_dir, str(doc_id_map[doc_id]) + '.txt'), 'w') as fp:
            fp.write(article)
        with open(path.join(entities_dir, str(doc_id_map[doc_id]) + '.txt'), 'w') as fp:
            pass
    with open(path.join(queries_dir, str(doc_id_map[doc_id]) + '.' + str(query_id) + '.txt'), 'w') as fp:
        fp.write(query)
    with open(path.join(references_dir, 'A.' + str(doc_id_map[doc_id]) + '.' + str(query_id) + '.txt'), 'w') as fp:
        fp.write(summary)
