import datasets
import json
import os

root = "./dataset"
if not os.path.exists(root):
    os.makedirs(root)


def get_translate_wmt_dataset():
    import os
    dir_name = "wmt"
    file_name = "news-commentary-v13.en"
    file_path = os.path.join(root, dir_name, file_name)
    if not os.path.exists(file_path):
        dir_path = os.path.join(root, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        url = "http://data.statmt.org/wmt18/translation-task/news-commentary-v13.en.gz"
        print("INFO: Downloading dataset wmt")
        os.system("curl -s -o {}.gz {}".format(file_path, url))
        print("INFO: Extracting dataset wmt")
        os.system("gzip -d {}.gz".format(file_path))
    with open(file_path, "r") as f:
        lines = f.readlines()
        data = [l.strip() for l in lines]
    return data


def get_summarize_cnn_dataset():
    import os
    print("INFO: Loading dataset cnn_dailymail")
    data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")
    return data["article"]


def get_chatbot_convAI_dataset():
    import os
    dir_name = "convAI"
    file_name = "data_tolokers.json"
    file_path = os.path.join(root, dir_name, file_name)
    if not os.path.exists(file_path):
        dir_path = os.path.join(root, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        url = "http://convai.io/data/data_tolokers.json"
        print("INFO: Downloading dataset convAI")
        os.system("curl -s -o {} {}".format(file_path, url))
    dataset = []
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        for each in data:
            for line in each['dialog']:
                dataset.append(line['text'])
    return dataset


def get_chatbot_cornell_dataset():
    import os
    dir_name = "cornell"
    zip_name = "cornell_movie_dialogs_corpus.zip"
    corpus_name = "cornell movie-dialogs corpus"
    dir_path = os.path.join(root, dir_name)
    zip_path = os.path.join(root, dir_name, zip_name)
    corpus_path = os.path.join(root, dir_name, corpus_name)
    if not os.path.exists(corpus_path):
        dir_path = os.path.join(root, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        print("INFO: Downloading dataset cornell_movie_dialogs_corpus")
        os.system("curl -s -o {} {}".format(zip_path, url))
        print("INFO: Extracting dataset cornell_movie_dialogs_corpus")
        os.system("unzip -q {} -d {}".format(zip_path, dir_path))
        os.system("rm -rf {} {}".format(zip_path, os.path.join(dir_path, "__MACOSX")))

    # from https://github.com/pytorch/tutorials/blob/master/beginner_source/chatbot_tutorial.py
    import csv
    import re
    import os
    import unicodedata
    import codecs
    from io import open

    corpus = corpus_path

    # Splits each line of the file into a dictionary of fields
    def loadLines(fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines

    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile('L[0-9]+')
                lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    # Extracts pairs of sentences from conversations
    def extractSentencePairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    # print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    # print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    # print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    # print("\nSample lines from file:")
    # printLines(datafile)

    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    class Voc:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

        # Remove words below a certain count threshold
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True

            keep_words = []

            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)

            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            ))

            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3 # Count default tokens

            for word in keep_words:
                self.addWord(word)

    MAX_LENGTH = 50  # Maximum sentence length to consider

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    # Read query/response pairs and return a voc object
    def readVocs(datafile, corpus_name):
        # print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Voc(corpus_name)
        return voc, pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(pairs):
        return [pair for pair in pairs if filterPair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(corpus, corpus_name, datafile, save_dir):
        # print("Start preparing training data ...")
        voc, pairs = readVocs(datafile, corpus_name)
        # print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = filterPairs(pairs)
        # print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        # print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        # print("Counted words:", voc.num_words)
        return voc, pairs


    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    dataset = []
    for pair in pairs:
        dataset.append(pair[0])

    return dataset

