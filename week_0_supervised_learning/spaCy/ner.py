import spacy
import sys


class SpacyNer:

    def __init__(self):
        # Set up spacy nlp to use english as the vocabulary.
        self.nlp = spacy.load('en')

    def main(self):
        if len(sys.argv) < 2:
            # Check text has been paseed in.
            print("You have not passed in text to perform NER on.")
        else:
            """ Perform ner on the argument which has been passed into the
            command line."""
            doc = self.nlp(sys.argv[1])
            for ent in doc.ents:
                print(ent, ent.label_)


if __name__ == "__main__":
    """ Instantiate new object and run main if file is simply called from
    command line."""
    spacy_class = SpacyNer()
    spacy_class.main()
