import re
from pathlib import Path
import pyarabic.araby as araby

#Our constants as per lil mahy's lil knowledge:

# tashkeel:
DAMMA= araby.DAMMA
FATHA = araby.FATHA
KASRA = araby.KASRA
TANWEEN_DAMMA = araby.DAMMATAN
TANWEEN_FATHA = araby.FATHATAN
TANWEEN_KASRA = araby.KASRATAN
SHADDA = araby.SHADDA
SUKOON = araby.SUKUN


#seperate eight diacritics into four groups: shortvowels, double case-endings, Shadda, and Sukoon.
SHORT_VOWELS = sorted((DAMMA, FATHA, KASRA))
DOUBLE_CASE_ENDINGS = sorted((TANWEEN_DAMMA, TANWEEN_FATHA, TANWEEN_KASRA))

#All diacritics
DIACRITICS_SET = frozenset(SHORT_VOWELS + DOUBLE_CASE_ENDINGS + [SHADDA, SUKOON])


#All arabic letters:
LIST_OF_ARABIC_LETTERS=['ى', 'ع', 'ظ', 'ح', 'ر', 'س', 'ي', 'ش', 'ض', 'ق', ' ', 'ث', 'ل', 'ص', 'ط', 'ك', 'آ', 'م', 'ا', 'إ', 'ه', 'ز', 'ء', 'أ', 'ف', 'ؤ', 'غ', 'ج', 'ئ', 'د', 'ة', 'خ', 'و', 'ب', 'ذ', 'ت', 'ن']

# punctuations
PUNCTUATIONS = ["،", ":", "؛", "-", "؟"]


#constants for our model:
DEFAULT_WINDOW_SIZE = 150
DEFAULT_SLIDING_STEP = DEFAULT_WINDOW_SIZE // 5
DEFAULT_EMBEDDING_SIZE = 128
DEFAULT_LSTM_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.4
MODELS_DIR = Path('../models/')   #model .h5 directory 
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TRAIN_STEPS = 100

#MAPPING NEED TO BE FIXED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! HASHTABLES?????????????

#look up tables - which are hash tables in the paper
CHAR_MAPPING_ENCODING = {
    '’ ’': 1,
    '0-9': 2,
    araby.HAMZA: 3,
    araby.ALEF_MADDA: 4,
    araby.ALEF_HAMZA_ABOVE: 5,
    araby.WAW_HAMZA: 6,
    araby.ALEF_HAMZA_BELOW: 7,
    araby.YEH_HAMZA: 8,
    araby.ALEF: 9,
    araby.BEH: 10,
    araby.TEH_MARBUTA: 11,
    araby.TEH: 12,
    araby.THEH: 13,
    araby.JEEM: 14,
    araby.HAH: 15,
    araby.KHAH: 16,
    araby.DAL: 17,
    araby.THAL: 18,
    araby.REH: 19,
    araby.ZAIN: 20,
    araby.SEEN: 21,
    araby.SHEEN: 22,
    araby.SAD: 23,
    araby.DAD: 24,
    araby.TAH: 25,
    araby.ZAH: 26,
    araby.AIN: 27,
    araby.GHAIN: 28,
    araby.FEH: 29,
    araby.MOON[12]: 30,
    araby.KAF: 31,
    araby.LAM: 32,
    araby.MEEM: 33,
    araby.NOON: 34,
    araby.HEH: 35,
    araby.WAW: 36,
    araby.YEHLIKE[2]: 37,
    araby.YEH: 38,
    ' ':0
}

SHORT_VOWELS_MAPPING_ENCODING={
    'None':0,
    araby.FATHA: 1,
    araby.DAMMA: 2,
    araby.KASRA: 3
}

DOUBLE_CASE_ENDINGS_MAPPING_ENCODING={
    'None':0,
    araby.TANWEEN_FATHA: 1,
    araby.TANWEEN_DAMMA: 2,
    araby.TANWEEN_KASRA: 3
}

SHADDA_MAPPING_ENCODING={
    'None':0,
    araby.SHADDA:1
}
SUKUN_MAPPING_ENCODING={
    'None':0,
    araby.SUKOON:1
}

DECODING_CHAR_MAPPING = {value: key for key, value in CHAR_MAPPING_ENCODING.items()}
DECODING_SHORT_VOWELS_MAPPING = {value: key for key, value in SHORT_VOWELS_MAPPING_ENCODING.items()}
DECODING_DOUBLE_CASE_ENDINGS_MAPPING = {value: key for key, value in DOUBLE_CASE_ENDINGS_MAPPING_ENCODING.items()}
DECODING_SHADDA_MAPPING = {value: key for key, value in SHADDA_MAPPING_ENCODING.items()}
DECODING_SUKUN_MAPPING = {value: key for key, value in SUKUN_MAPPING_ENCODING.items()}
#I think we need the hash map because the decoding is not accurate here, and the keys of chaprmapping are all read as 0


### The following is from the paper############################################################
# DAMMA = 'ُ'
# FATHA = 'َ'
# KASRA = 'ِ'
# TANWEEN_DAMMA = 'ٌ'
# TANWEEN_FATHA = 'ً'
# TANWEEN_KASRA = 'ٍ'
# SHADDA = 'ّ'
# SUKOON = 'ْ'
# SHORT_VOWELS = sorted((DAMMA, FATHA, KASRA))
# DOUBLE_CASE_ENDINGS = sorted((TANWEEN_DAMMA, TANWEEN_FATHA, TANWEEN_KASRA))

# ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
# ARABIC_PATTERN = re.compile(r'[%s]' % ''.join(ARABIC_LETTERS))
# DIACRITICS = frozenset(SHORT_VOWELS + DOUBLE_CASE_ENDINGS + [SHADDA, SUKOON])
# DIACRITICS_PATTERN = re.compile(r'[%s]' % ''.join(DIACRITICS))
# DIGIT = '0'
# DIGIT_PATTERN = re.compile(r'\d')
# SENTENCE_SEPARATORS = ';,،؛.:؟!'
# SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([%s](?!\w)|\n)' % SENTENCE_SEPARATORS)

# DEFAULT_WINDOW_SIZE = 150
# DEFAULT_SLIDING_STEP = DEFAULT_WINDOW_SIZE // 5
# DEFAULT_EMBEDDING_SIZE = 128
# DEFAULT_LSTM_SIZE = 128
# DEFAULT_DROPOUT_RATE = 0.4
# DEFAULT_PARAMS_DIR = Path('params/')
# DEFAULT_BATCH_SIZE = 1024
# DEFAULT_TRAIN_STEPS = 100
# DEFAULT_EARLY_STOPPING_STEPS = 10
# DEFAULT_MONITOR_METRIC = 'val_loss'
# DEFAULT_DIACRITIZATION_LINES_COUNT = 100

# CHARS = sorted(ARABIC_LETTERS.union({DIGIT, ' '}))