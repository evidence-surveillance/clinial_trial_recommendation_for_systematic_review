import spacy
import string


def get_stopwords():
    pubmed_stopwords = 'a, about, again, all, almost, also, although, always, among, an, and, another, any, are, as, ' \
                       'at, be, because, been, before, being, between, both, but, by,can, could,did, do, does, done, ' \
                       'due, during,each, either, enough, especially, etc, for, found, from, further, had, has, ' \
                       'have, having, here, how, however, i, if, in, into, is, it, its, itself, just, kg, km, made, ' \
                       'mainly, make, may, mg, might, ml, mm, most, mostly, must, nearly, neither, no, nor, ' \
                       'obtained, of, often, on, our, overall, perhaps, pmid, quite, rather, really, regarding, ' \
                       'seem, seen, several, should, show, showed, shown, shows, significantly, since, so, some, ' \
                       'such, than, that, the, their, theirs, them, then, there, therefore, these, they, this, ' \
                       'those, through, thus, to, upon, various, very, was, we, were, what, when, which, while, ' \
                       'with, within, without, would'.replace(' ', '').split(',')
    # nlp = spacy.load("en_core_web_sm")
    stopwords = ['nothing', 'between', 'how', 'top', 'due', 'whereafter', 'do', 'he', 'at', 'while', 'anyhow', 'besides', 'only', 'ours', 'may', 'third', 'much', 'several', 'anyway', 'together', 'also', 'fifteen', 'any', 'even', 'becomes', 'noone', 'their', 'her', 'it', 'whenever', 'whereupon', 'see', 'though', 'please', 'used', 'what', 'an', 'that', 'doing', 'towards', 'per', 'never', 'else', 'moreover', 'did', 'we', 'am', 'a', 'show', "'ll", 'these', 'for', 'hers', 'us', 'would', 'elsewhere', 'wherein', 'each', 'beyond', 'therein', 'which', 'whatever', 'within', 'yours', 'himself', 'thereafter', 'n’t', 'up', 'sixty', 'they', 'latterly', 'toward', 'among', 'seem', 'until', 'more', 'during', 'is', 'either', 'being', 'something', 'themselves', 'before', 'neither', 'herein', 'made', 'many', 'go', 'take', 'indeed', 'whether', 'ourselves', 'move', 'nine', '‘ll', 'your', 'serious', 'quite', 'name', 'whence', 'other', 'rather', 'sometimes', 'seemed', 'others', 'down', '‘re', 'enough', 'somehow', '’ll', 'various', 'his', 'former', 'after', 'over', 'none', 'now', 'mostly', 'really', 'might', 'own', 'all', 'herself', 'onto', 'was', 'had', 'amount', 'through', 'nevertheless', 'by', 'beside', 'around', 'but', '‘s', 'does', 'this', 'them', 'against', 'six', 'on', 'into', 'last', 'to', 'most', 'well', 'often', 'ten', 'who', 'yourselves', 'every', 'everywhere', 'although', 'anyone', 'became', 'always', 'without', 'just', 'except', 'with', 'nowhere', 'those', 'wherever', 'n‘t', 'full', 'hereby', "'ve", 'sometime', 'done', 'someone', 'using', 'than', 'off', 'then', 'forty', '’m', 'eleven', 'have', 'some', '’s', 'in', 'must', 'should', 'empty', 'will', 'otherwise', 'give', 'alone', 'or', 'latter', 'as', 'and', 'no', 'out', 'hence', 'i', 'five', 'hundred', 'when', 'say', 'everything', 'could', 'front', 'again', 'meanwhile', 'unless', 'yourself', 'its', '’re', 'further', 'already', '’d', 'along', 'hereupon', 'under', 'here', "'re", 'same', 'get', 'whose', 'whereby', 'seems', 'has', 'once', 'formerly', 'my', 'anywhere', 'regarding', 'nor', '‘m', 'still', 'namely', 'put', 'cannot', 'our', 'nobody', 'however', 'both', 'been', 'itself', 'thereupon', 'can', 'you', 'such', 'why', 'yet', 'thereby', 'thus', 'another', 'thence', 'four', "'d", 'one', 'twelve', 'whole', 'afterwards', 'least', 'less', 'from', 'becoming', 'few', "'m", 'she', 'be', 'beforehand', '‘ve', 'seeming', 'part', 'are', 'make', 'next', 'mine', 'first', 'where', 'since', 'thru', 'about', 'almost', 'below', 'across', 'become', 'there', 'him', 'whom', 'behind', 'three', 'ever', 'so', 'myself', 'via', 'me', '’ve', 'amongst', 'not', 'hereafter', 'above', 'whoever', 'were', 'of', 'throughout', 'call', 'whither', 'eight', 'twenty', 'bottom', 'the', 'because', '‘d', 're', 'very', 'therefore', 'keep', 'somewhere', 'side', 'everyone', 'back', 'if', 'upon', 'perhaps', 'whereas', 'too', "n't", 'two', 'fifty', 'ca', 'anything', "'s"]
    #
    # list(nlp.Defaults.stop_words)
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    # list(string.punctuation)
    single_chars = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', '‘', '’']
    allstopwords = stopwords + punctuations + single_chars + pubmed_stopwords
    allstopwords = list(set(allstopwords))
    return allstopwords


def spacy_tokenization(nlp, text):
    if not isinstance(text, str):
        print(text)
    doc = nlp(text)
    return [[token.text.strip() for token in sent if token.text.strip()] for sent in doc.sents]


def tokenization_func(x):

    return [t.strip() for t in x if t.strip()]


def bracket_replace(text):
    return text.replace('"', '').replace('NULL', '')\
        .replace('{', '').replace('}', '')\
        .replace('[', '').replace(']', '')\
        .replace('(', '').replace(')', '')
