import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_wordcloud(texts):
    wc = WordCloud(width=800, height=400).generate(' '.join(texts))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
