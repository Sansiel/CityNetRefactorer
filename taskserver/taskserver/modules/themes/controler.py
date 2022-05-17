# import imp
from re import T
import string
from numpy import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pymystem3 import Mystem

# import nltk
# nltk.download('wordnet')

stopwords = stopwords.words('russian')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
my_stem = Mystem()

# Tf/df и последня лекция

themes = [
    'Прогноз развития литературных традиций.',
    'Н. М. Карамзин «Сиерра Морена» – яркий образец лирической прозы русского романтического направления XVIII века.',
    'Поэтические традиции XIX века в творчестве А. Н. Апухтина. Апухтин А. Н. Стихотворение «День ли царит, тишина ли ночная…». Анализ стихотворения.',
    'Л. Н. Толстой. «Народные рассказы» - подлинная энциклопедия народной жизни.',
    'А. П. Чехов. «В рождественскую ночь». Иронический парадокс в рождественском рассказе.',
    'Традиции литературы XX века. Малый эпический жанр.',
    'А. М. Горький «Макар Чудра».',
    '«Макар Чудра» Герои неоромантизма.',
    'Обобщение пройденного.',
    'Контрольная работа № 1.',
    'А. И. Куприн рассказ «Габринус».',
    '«Живое и мертвое» в рассказе Куприна А. И.',
    'Две героини, две судьбы в рассказе Куприна А. И. «Габринус».',
    'Ю. П. Казаков. «Двое в декабре». Смысл названия рассказа.'
]


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if len(word) > 1])
    text = ' '.join([word for word in text.split() if word not in stopwords])

    # print([stemmer.stem(word) for word in text.split()])
    # print([lemmatizer.lemmatize(word) for word in text.split()])
    # print([my_stem.lemmatize(word) for word in text.split()])

    # Стемизация
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # Лематизация
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    # MyStem (лучше с русским работает)
    text = ' '.join(my_stem.lemmatize(text)).strip()
    return text


cleaned = list(map(clean_string, themes))
print(cleaned)

vectorize = CountVectorizer().fit_transform(cleaned)
vectors = vectorize.toarray()
csim = cosine_similarity(vectors)
# print(csim)

# [[1.         0.         0.         0.         0.         0.                 0.         0.         0.         0.         0.         0.                   0.         0.        ],
#  [0.         1.         0.06804138 0.         0.         0.10910895         0.         0.         0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.06804138 1.         0.         0.         0.17817416         0.         0.         0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         1.         0.         0.                 0.         0.         0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         1.         0.                 0.         0.         0.         0.         0.         0.15811388           0.1118034  0.        ]
#  [0.         0.10910895 0.17817416 0.         0.         1.                 0.         0.         0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 1.         0.57735027 0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 0.57735027 1.         0.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 0.         0.         1.         0.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 0.         0.         0.         1.         0.         0.                   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 0.         0.         0.         0.         1.         0.2236068            0.31622777 0.        ]
#  [0.         0.         0.         0.         0.15811388 0.                 0.         0.         0.         0.         0.2236068  1.                   0.42426407 0.        ]
#  [0.         0.         0.         0.         0.1118034  0.                 0.         0.         0.         0.         0.31622777 0.42426407           1.         0.        ]
#  [0.         0.         0.         0.         0.         0.                 0.         0.         0.         0.         0.         0.                   0.         1.        ]]

listNumsStrs = list()
exceptNumsStrs = list()
for i in range(0, len(csim) - 1):
    # print(i,csim[i])
    for j in range(i + 1, len(csim[i])):
        if csim[i][j] > 0.3:
            listNumsStrs.append([i, j])
            if i not in exceptNumsStrs:
                exceptNumsStrs.append(i)
            if j not in exceptNumsStrs:
                exceptNumsStrs.append(j)

print(listNumsStrs)
# print(exceptNumsStrs)
# [[6, 7], [10, 11], [10, 12], [11, 12]]


# --------------------------------------------------------------------------


newThemes = dict()
for i in range(0, len(themes)):
    if i not in exceptNumsStrs:
        newThemes[i + 1] = themes[i]
# print(newThemes)


endedNumsStrs = dict()
for i in listNumsStrs:
    newTheme = ""
    if i[1] in endedNumsStrs:
        i[0] = endedNumsStrs[i[1]]
    for word in cleaned[i[0]].split():
        if word in cleaned[i[1]]:
            newTheme = newTheme + ' ' + word
    newThemes[i[0] + 1] = newTheme
    endedNumsStrs[i[1]] = i[0]

print(newThemes)