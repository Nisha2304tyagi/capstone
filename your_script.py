import os
import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
nltk.data.path.append(os.getenv('NLTK_DATA'))
# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

def brand_insights(df, top_n=10, filter_negative=False):
    # Step 1: Data Preparation
    df['Review Text'] = df['Review Text'].fillna('')
    df['review'] = df['Review Text'].apply(lambda x: word_tokenize(x))

    # Step 2: Categorization
    product_keywords = ['product', 'quality', 'features', 'performance', 'design', 'functionality', 'size', 'durability', 
                        'reliability', 'sturdy', 'color', 'style', 'comfort', 'usability', 'specifications', 'condition']
    delivery_keywords = ['shipping', 'delivery', 'arrived', 'packaging', 'timely', 'delay', 'late', 'waiting', 'dispatch', 
                         'courier', 'tracking', 'on-time', 'schedule', 'logistics', 'arrival', 'transit', 'handling', 'driver']
    price_keywords = ['price', 'value', 'cost', 'expensive', 'cheap', 'affordable', 'worth', 'discount', 'deal', 'bargain', 
                      'pricing', 'reasonable', 'overpriced', 'underpriced', 'investment']
    customer_care_keywords = ['support', 'service', 'help', 'response', 'issue', 'complaint', 'customer', 'assistance', 
                              'support team', 'help desk', 'contact', 'follow-up', 'problem resolution', 'communication']

    def categorize_review(text):
        if any(keyword in text for keyword in product_keywords):
            return 'product'
        elif any(keyword in text for keyword in delivery_keywords):
            return 'delivery'
        elif any(keyword in text for keyword in price_keywords):
            return 'price'
        elif any(keyword in text for keyword in customer_care_keywords):
            return 'customer care'
        else:
            return 'others'

    df['category'] = df['review'].apply(categorize_review)

    # Step 3: Stopwords Removal
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    extra_chars_to_remove = {"'", ","}

    def clean_review(review_tokens):
        return [word.lower() for word in review_tokens if word.lower() not in stop_words 
                and word not in punctuations and word not in extra_chars_to_remove]

    df['cleared_review'] = df['review'].apply(clean_review)
    
    # Step 4: Sentiment Analysis
    sia = SentimentIntensityAnalyzer()

    def vader_sentiment(text):
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            return 'POSITIVE', compound_score
        elif compound_score <= -0.05:
            return 'NEGATIVE', compound_score
        else:
            return 'NEUTRAL', compound_score

    df[['sentiment_label', 'sentiment_score']] = df['Review Text'].apply(lambda x: pd.Series(vader_sentiment(x)))

    if filter_negative:
        df = df[df['sentiment_label'] == 'NEGATIVE']

    # Step 5: Visualization - Sentiment Distribution by Category
    sentiment_counts = df.groupby(['category', 'sentiment_label']).size().unstack().fillna(0)
    fig, axes = plt.subplots(1, len(sentiment_counts), figsize=(15, 8), subplot_kw=dict(aspect="equal"))

    for i, (category, counts) in enumerate(sentiment_counts.iterrows()):
        axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'{category.capitalize()} Sentiment Distribution')

    plt.savefig('static/output/pie_chart.png')
    plt.close()

    # Step 6: Key Complaint Extraction - Top Negative Terms by Category
    negative_reviews = df[df['sentiment_label'] == 'NEGATIVE']
    top_keywords_by_category = {}

    def get_top_keywords(text_series, top_n=10):
        all_words = [word for text in text_series for word in text]
        common_words = Counter(all_words).most_common(top_n)
        return dict(common_words)

    for category in negative_reviews['category'].unique():
        category_reviews = negative_reviews[negative_reviews['category'] == category]['cleared_review']
        top_keywords_by_category[category] = get_top_keywords(category_reviews, top_n=top_n)

    fig, axes = plt.subplots(1, len(top_keywords_by_category), figsize=(18, 6), sharey=True)

    for i, (category, keywords) in enumerate(top_keywords_by_category.items()):
        axes[i].bar(keywords.keys(), keywords.values(), color='blue')
        axes[i].set_title(f'Top {top_n} Negative Terms - {category.capitalize()}')
        axes[i].set_xticklabels(keywords.keys(), rotation=45, ha='right')
        axes[i].set_xlabel('Keywords')
        axes[i].set_ylabel('Frequency' if i == 0 else "")

    plt.savefig('static/output/bar_chart.png')
    plt.close()
