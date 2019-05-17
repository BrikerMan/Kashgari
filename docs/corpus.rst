Corpus
===============
Kashgari provides several build-in corpus for testing.

Chinese Daily Ner Corpus
------------------------
Chinese Ner corpus cotains 20864 train samples, 4636 test samples and 2318 valid samples.

Usage::

    from kashgari.corpus import ChineseDailyNerCorpus

    train_x, train_y = ChineseDailyNerCorpus.load_data('train')
    test_x, test_y = ChineseDailyNerCorpus.load_data('test')
    valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

Data Sample::

    >>> x[0] 
    ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门']
    >>> y[0] 
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']
