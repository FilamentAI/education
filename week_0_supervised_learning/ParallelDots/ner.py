from paralleldots import set_api_key, get_api_key, ner
from paralleldots import similarity, ner, taxonomy, sentiment, keywords, intent, emotion, abuse

import sys
import os.path


class Parallel:

    def __init__(self):
        if os.path.isfile('settings.cfg'):
            """ If settings file already exists then no need to set up api key,
             this file is automatically generated by the set and get api key
             functions."""
            print("Settings file already exists with appropriate API details.")
        else:
            # Setup api keys, this creates the settings.cfg file.
            set_api_key("") # Please enter your API key here.
            get_api_key()

    def main(self):
        if len(sys.argv) < 2:
            # Check text has been passed in via the command line.
            print("You have not passed in text to perform NER on.")
        else:
            """ Perform ner on the argument which has been passed into the
            command line."""
            print(ner(sys.argv[1]))


if __name__ == "__main__":
    """ Instantiate new object and run main if file is simply called from
    command line."""
    parallel = Parallel()
    parallel.main()