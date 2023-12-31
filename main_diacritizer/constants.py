import re
from pathlib import Path
import pyarabic.araby as araby

#Our constants as per lil mahy's lil knowledge:


# Diacritics (8):
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

# starting from 0
DIGIT = '0'

# as in the lookup we need any digit from 0-9 in index 2 (1+1)
DIGIT_REGEX = re.compile(r'\d') 

# all valid inputs that will be used in the look up table
LIST_OF_VALID_INPUT_LETTERS=LIST_OF_ARABIC_LETTERS+['0']

# all valid inputs that will be used in the look up table after sorting them
SORTED_VALID_INPUT_LETTERS= sorted(LIST_OF_VALID_INPUT_LETTERS)



#constants for our model:
DEFAULT_WINDOW_SIZE = 150
DEFAULT_SLIDING_STEP = DEFAULT_WINDOW_SIZE // 5
DEFAULT_EMBEDDING_SIZE = 128
DEFAULT_LSTM_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.4        # to avoid overfitting
MODELS_DIR = Path('../models/')   #model .h5 directory 
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TRAIN_STEPS = 100









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