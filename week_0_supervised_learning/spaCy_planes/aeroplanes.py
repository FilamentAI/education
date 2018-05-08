"""

Using spaCy to recognize aeroplane names, based on [1]. Trains a blank model
with an NER pipe, then tests it on the training examples.

[1]: https://github.com/explosion/spacy/blob/master/examples/training/train_ner.py

"""
import spacy
import random


TRAINING_SET = [
        ('The Boeing 747 is an American wide-body commercial ', {
            'entities': [(4, 14, 'AEROPLANE')]
        }),
        ]
N_ITERATIONS = 100
DROPOUT = 0.5


def main():
    # create a blank model
    nlp = spacy.blank('en')
    # add a new NER pipe to it to recognize AEROPLANE entities
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    ner.add_label('AEROPLANE')

    # training, disable all pipes except for NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for _ in range(N_ITERATIONS):
            losses = {}
            random.shuffle(TRAINING_SET)
            for text, annotations in TRAINING_SET:
                nlp.update([text], [annotations], drop=DROPOUT,
                           sgd=optimizer, losses=losses)

    # checking if the model works on training examples now
    for text, _ in TRAINING_SET:
        print('Detecting entities in text: {}'.format(text))
        doc = nlp(text)
        print([(ent.text, ent.label_) for ent in doc.ents])


if __name__ == '__main__':
    main()
