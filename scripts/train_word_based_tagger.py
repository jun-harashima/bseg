import torch
from bseg.model.word_based_tagger import WordBasedTagger
from bseg.dataset.one_input_dataset import OneInputDataset


EMBEDDING_DIM = 2
HIDDEN_DIM = 4

examples = [
    (('名詞', '助詞', '動詞'), ('人参', 'を', '切る')),
    (('副詞', '動詞'), ('ざっくり', '切る')),
    (('名詞', '助詞', '形容詞', '動詞'), ('葱', 'は', '細く', '刻む'))
]
dataset = OneInputDataset(examples)

model = WordBasedTagger(EMBEDDING_DIM, HIDDEN_DIM, [len(dataset.word_to_index)],
                        dataset.tag_to_index, batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'word_based_tagger.model')

model.load_state_dict(torch.load('word_based_tagger.model'))
examples = [
    (('名詞', '助詞', '動詞'), ('葱', 'を', '切る')),
    (('副詞', '動詞'), ('細く', '切る')),
    (('名詞', '助詞', '形容詞', '動詞'), ('大根', 'は', 'ざっくり', '刻む'))
]
dataset = OneInputDataset(examples)
results = model.test(dataset)
print(results)
