import nltk
import ssl

ssl._create_default_httpscontext = ssl._create_unverified_context
nltk.download('stopwords')