import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
import tweepy
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji
import re
from langdetect import detect, DetectorFactory
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

DetectorFactory.seed = 0

class MultilingualEmotionAnalyzer:
    """Многоязычный анализатор эмоций и тональности"""
    
    def __init__(self):
        # Инициализация моделей для разных языков
        self.sentiment_models = {
            'english': pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            ),
            'russian': pipeline(
                "sentiment-analysis", 
                model="blanchefort/rubert-base-cased-sentiment"
            ),
            'chinese': pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-chinanews-chinese"
            ),
            'multilingual': pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        }
        
        # Модели для анализа эмоций
        self.emotion_models = {
            'english': pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            ),
            'multilingual': pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
        }
        
        # Дополнительные анализаторы
        self.vader = SentimentIntensityAnalyzer()
        
        # Культурные модификаторы для интерпретации эмоций
        self.cultural_modifiers = {
            'english': {
                'sarcasm_markers': ['yeah right', 'sure thing', 'oh great', 'wonderful'],
                'intensity_multiplier': 1.0,
                'politeness_weight': 0.3
            },
            'russian': {
                'sarcasm_markers': ['ну да', 'конечно же', 'как же', 'замечательно'],
                'intensity_multiplier': 1.2,  # Русские более эмоционально экспрессивны
                'politeness_weight': 0.2
            },
            'chinese': {
                'sarcasm_markers': ['当然', '太好了', '真的吗'],
                'intensity_multiplier': 0.8,  # Более сдержанная культура
                'politeness_weight': 0.5
            }
        }
        
        # Эмодзи анализ
        self.emotion_emojis = {
            'joy': ['😊', '😀', '😁', '😄', '😃', '🙂', '😋', '😆', '😂', '🤣'],
            'sadness': ['😢', '😭', '😞', '😔', '😟', '😕', '☹️', '🙁'],
            'anger': ['😠', '😡', '🤬', '😤', '💢', '👿'],
            'fear': ['😨', '😰', '😱', '🙀', '😧'],
            'surprise': ['😮', '😯', '😲', '🤯', '😳'],
            'love': ['😍', '🥰', '😘', '💕', '❤️', '💖', '💝']
        }
        
    def preprocess_social_media_text(self, text: str, language: str) -> str:
        """Предобработка текста из социальных сетей"""
        
        # Сохранение важных эмоциональных маркеров
        text = self.preserve_emotional_markers(text)
        
        # Обработка специфичных элементов соцсетей
        text = re.sub(r'@\w+', '[USER]', text)  # Упоминания пользователей
        text = re.sub(r'#(\w+)', r'\1', text)   # Хештеги -> слова
        text = re.sub(r'http\S+|www\S+', '[URL]', text)  # Ссылки
        
        # Обработка повторяющихся символов (но сохранение эмоциональности)
        text = re.sub(r'([!?.]){3,}', r'\1\1\1', text)
        text = re.sub(r'([a-zA-Zа-яА-Я])\1{2,}', r'\1\1', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preserve_emotional_markers(self, text: str) -> str:
        """Сохранение эмоциональных маркеров"""
        
        # Замена некоторых интернет-сокращений на полные формы
        replacements = {
            'lol': 'laugh out loud',
            'omg': 'oh my god', 
            'wtf': 'what the f',
            'imho': 'in my humble opinion',
            'тлдр': 'слишком длинно не читал',
            'кек': 'смешно',
            'лол': 'смешно'
        }
        
        text_lower = text.lower()
        for abbrev, full_form in replacements.items():
            text_lower = text_lower.replace(abbrev, full_form)
            
        return text_lower
    
    def extract_emoji_emotions(self, text: str) -> Dict[str, int]:
        """Извлечение эмоций из эмодзи"""
        emoji_emotions = defaultdict(int)
        
        # Извлечение всех эмодзи из текста
        emojis_in_text = [c for c in text if c in emoji.EMOJI_DATA]
        
        # Подсчет эмоций на основе эмодзи
        for emoji_char in emojis_in_text:
            for emotion, emoji_list in self.emotion_emojis.items():
                if emoji_char in emoji_list:
                    emoji_emotions[emotion] += 1
        
        return dict(emoji_emotions)
    
    def detect_sarcasm(self, text: str, language: str) -> float:
        """Детекция сарказма с учетом культурных особенностей"""
        
        sarcasm_score = 0.0
        text_lower = text.lower()
        
        # Культурные маркеры сарказма
        cultural_markers = self.cultural_modifiers.get(language, {}).get('sarcasm_markers', [])
        
        for marker in cultural_markers:
            if marker in text_lower:
                sarcasm_score += 0.3
        
        # Общие индикаторы сарказма
        # Контрастные конструкции
        contrast_patterns = [
            r'but\s+\w+',  # but really
            r'а\s+на\s+самом\s+деле',  # а на самом деле  
            r'но\s+на\s+самом\s+деле',  # но на самом деле
            r'但是实际上'  # но на самом деле (китайский)
        ]
        
        for pattern in contrast_patterns:
            if re.search(pattern, text_lower):
                sarcasm_score += 0.2
        
        # Чрезмерные положительные эпитеты в негативном контексте
        positive_words = ['amazing', 'wonderful', 'perfect', 'замечательно', 'отлично', '完美']
        negative_context = ['not', 'never', 'no', 'не', 'нет', 'никогда', '不', '没']
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_context)
        
        if has_positive and has_negative:
            sarcasm_score += 0.4
        
        # Множественные знаки препинания
        if re.search(r'[!?]{2,}', text):
            sarcasm_score += 0.1
            
        # Кавычки вокруг слов
        if re.search(r'"[^"]*"', text):
            sarcasm_score += 0.2
            
        return min(sarcasm_score, 1.0)
    
    def analyze_comprehensive_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """Комплексный анализ тональности и эмоций"""
        
        results = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
                
            try:
                # Определение языка
                detected_lang = detect(text)
                lang_mapping = {'en': 'english', 'ru': 'russian', 'zh': 'chinese'}
                language = lang_mapping.get(detected_lang, 'multilingual')
                
                # Предобработка
                processed_text = self.preprocess_social_media_text(text, language)
                
                # Анализ тональности
                sentiment_model = self.sentiment_models.get(language, self.sentiment_models['multilingual'])
                sentiment_result = sentiment_model(processed_text[:512])[0]
                
                # Анализ эмоций
                emotion_model = self.emotion_models.get(language, self.emotion_models['multilingual'])
                emotion_results = emotion_model(processed_text[:512])[0]
                
                # Преобразование результатов эмоций в словарь
                emotion_scores = {item['label']: item['score'] for item in emotion_results}
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                
                # VADER анализ (работает лучше для неформального текста)
                vader_scores = self.vader.polarity_scores(processed_text)
                
                # Анализ эмодзи
                emoji_emotions = self.extract_emoji_emotions(text)
                
                # Детекция сарказма
                sarcasm_score = self.detect_sarcasm(processed_text, language)
                
                # Культурная корректировка
                cultural_mod = self.cultural_modifiers.get(language, {})
                intensity_mult = cultural_mod.get('intensity_multiplier', 1.0)
                
                # Композитная метрика эмоциональной интенсивности
                emotional_intensity = (
                    abs(sentiment_result['score'] - 0.5) * 2 * intensity_mult +
                    dominant_emotion[1] * intensity_mult +
                    sum(emoji_emotions.values()) * 0.1
                ) / 2
                
                result = {
                    'text_id': i,
                    'original_text': text[:200] + '...' if len(text) > 200 else text,
                    'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text,
                    'detected_language': language,
                    
                    # Основная тональность
                    'sentiment_label': sentiment_result['label'],
                    'sentiment_score': sentiment_result['score'],
                    'sentiment_confidence': sentiment_result['score'],
                    
                    # VADER тональность
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu'],
                    
                    # Эмоции
                    'dominant_emotion': dominant_emotion[0],
                    'dominant_emotion_score': dominant_emotion[1],
                    **{f'emotion_{k}': v for k, v in emotion_scores.items()},
                    
                    # Эмодзи эмоции
                    **{f'emoji_{k}': v for k, v in emoji_emotions.items()},
                    'total_emojis': sum(emoji_emotions.values()),
                    
                    # Специальные характеристики
                    'sarcasm_score': sarcasm_score,
                    'emotional_intensity': emotional_intensity,
                    'text_length': len(text),
                    'processed_length': len(processed_text),
                    
                    # Культурный контекст
                    'cultural_intensity_modifier': intensity_mult,
                    'is_likely_sarcastic': sarcasm_score > 0.5
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing text {i}: {str(e)[:100]}")
                continue
        
        return pd.DataFrame(results)
    
    def compare_cross_cultural_emotions(self, results_df: pd.DataFrame, 
                                      topic_keywords: List[str]) -> Dict:
        """Сравнение эмоций между культурами по определенной теме"""
        
        # Фильтрация по теме
        topic_pattern = '|'.join(topic_keywords)
        topic_mask = results_df['original_text'].str.contains(topic_pattern, case=False, na=False)
        topic_data = results_df[topic_mask]
        
        if len(topic_data) == 0:
            return {'error': 'No data found for the specified topic'}
        
        # Группировка по языкам
        cultural_comparison = {}
        
        for language in topic_data['detected_language'].unique():
            lang_data = topic_data[topic_data['detected_language'] == language]
            
            # Базовая статистика тональности
            sentiment_stats = {
                'avg_sentiment_score': lang_data['sentiment_score'].mean(),
                'avg_vader_compound': lang_data['vader_compound'].mean(),
                'avg_emotional_intensity': lang_data['emotional_intensity'].mean(),
                'avg_sarcasm_score': lang_data['sarcasm_score'].mean(),
                'sample_size': len(lang_data)
            }
            
            # Распределение эмоций
            emotion_columns = [col for col in lang_data.columns if col.startswith('emotion_')]
            emotion_averages = {}
            
            for col in emotion_columns:
                emotion_name = col.replace('emotion_', '')
                emotion_averages[emotion_name] = lang_data[col].mean()
            
            # Топ доминирующие эмоции
            dominant_emotions = lang_data['dominant_emotion'].value_counts().head().to_dict()
            
            # Использование эмодзи
            emoji_columns = [col for col in lang_data.columns if col.startswith('emoji_')]
            emoji_usage = {}
            
            for col in emoji_columns:
                emoji_type = col.replace('emoji_', '')
                emoji_usage[emoji_type] = lang_data[col].sum()
            
            cultural_comparison[language] = {
                'sentiment_statistics': sentiment_stats,
                'emotion_averages': emotion_averages,
                'dominant_emotions': dominant_emotions,
                'emoji_usage': emoji_usage,
                'cultural_insights': self.generate_cultural_insights(lang_data, language)
            }
        
        return cultural_comparison
    
    def generate_cultural_insights(self, lang_data: pd.DataFrame, language: str) -> List[str]:
        """Генерация культурных инсайтов"""
        insights = []
        
        # Анализ интенсивности эмоций
        avg_intensity = lang_data['emotional_intensity'].mean()
        if avg_intensity > 0.7:
            insights.append(f"{language} speakers show high emotional intensity in discussions")
        elif avg_intensity < 0.3:
            insights.append(f"{language} speakers tend to be more emotionally restrained")
        
        # Анализ сарказма
        avg_sarcasm = lang_data['sarcasm_score'].mean()
        if avg_sarcasm > 0.3:
            insights.append(f"High sarcasm usage detected in {language} posts")
        
        # Анализ эмодзи
        total_emojis = lang_data['total_emojis'].sum()
        if total_emojis > len(lang_data) * 0.5:
            insights.append(f"{language} speakers frequently use emojis to express emotions")
        
        # Анализ длины текстов
        avg_length = lang_data['text_length'].mean()
        if avg_length > 200:
            insights.append(f"{language} users tend to write longer, more detailed posts")
        elif avg_length < 100:
            insights.append(f"{language} users prefer concise, brief expressions")
        
        return insights
    
    def create_emotion_dashboard(self, results_df: pd.DataFrame) -> go.Figure:
        """Создание интерактивного дашборда эмоций"""
        
        # Создание subplot'ов
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sentiment Distribution by Language', 
                          'Emotion Intensity Comparison',
                          'Sarcasm vs Emotional Intensity', 
                          'Emoji Usage by Language'],
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Распределение тональности по языкам
        sentiment_counts = results_df.groupby(['detected_language', 'sentiment_label']).size().unstack(fill_value=0)
        
        for i, lang in enumerate(sentiment_counts.index):
            fig.add_trace(
                go.Bar(
                    name=lang,
                    x=sentiment_counts.columns,
                    y=sentiment_counts.loc[lang],
                    text=sentiment_counts.loc[lang],
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # 2. Box plot эмоциональной интенсивности
        for lang in results_df['detected_language'].unique():
            lang_data = results_df[results_df['detected_language'] == lang]
            fig.add_trace(
                go.Box(
                    y=lang_data['emotional_intensity'],
                    name=lang,
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # 3. Scatter plot: сарказм vs эмоциональная интенсивность
        colors = {'english': 'blue', 'russian': 'red', 'chinese': 'green'}
        
        for lang in results_df['detected_language'].unique():
            lang_data = results_df[results_df['detected_language'] == lang]
            fig.add_trace(
                go.Scatter(
                    x=lang_data['sarcasm_score'],
                    y=lang_data['emotional_intensity'],
                    mode='markers',
                    name=lang,
                    marker=dict(color=colors.get(lang, 'gray'), opacity=0.6),
                    text=[text[:50] + '...' for text in lang_data['original_text']],
                    hovertemplate='%{text}<br>Sarcasm: %{x}<br>Intensity: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Использование эмодзи
        emoji_cols = [col for col in results_df.columns if col.startswith('emoji_')]
        emoji_totals = {}
        
        for lang in results_df['detected_language'].unique():
            lang_data = results_df[results_df['detected_language'] == lang]
            emoji_totals[lang] = lang_data[emoji_cols].sum().sum()
        
        fig.add_trace(
            go.Bar(
                x=list(emoji_totals.keys()),
                y=list(emoji_totals.values()),
                name='Total Emojis',
                text=list(emoji_totals.values()),
                textposition='auto',
                marker_color=['lightblue', 'lightcoral', 'lightgreen'][:len(emoji_totals)]
            ),
            row=2, col=2
        )
        
        # Обновление layout
        fig.update_layout(
            height=800,
            title_text="Cross-Cultural Emotion Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Sentiment", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_yaxes(title_text="Emotional Intensity", row=1, col=2)
        
        fig.update_xaxes(title_text="Sarcasm Score", row=2, col=1)
        fig.update_yaxes(title_text="Emotional Intensity", row=2, col=1)
        
        fig.update_xaxes(title_text="Language", row=2, col=2)
        fig.update_yaxes(title_text="Total Emojis", row=2, col=2)
        
        return fig


class SocialMediaDataCollector:
    """Сборщик данных из социальных сетей"""
    
    def __init__(self, twitter_bearer_token: str = None, reddit_credentials: Dict = None):
        self.twitter_bearer_token = twitter_bearer_token
        self.reddit_credentials = reddit_credentials
        
        # Инициализация клиентов
        if twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
        
        if reddit_credentials:
            self.reddit_client = praw.Reddit(**reddit_credentials)
    
    def collect_twitter_data(self, query: str, lang: str = None, max_results: int = 100) -> List[Dict]:
        """Сбор данных из Twitter"""
        
        if not self.twitter_client:
            return []
        
        tweets = []
        
        try:
            # Поиск твитов
            search_results = self.twitter_client.search_recent_tweets(
                query=query,
                lang=lang,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang']
            )
            
            if search_results.data:
                for tweet in search_results.data:
                    tweets.append({
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'language': tweet.lang,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'source': 'twitter'
                    })
        
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
        
        return tweets
    
    def collect_reddit_data(self, subreddit_name: str, query: str = None, limit: int = 100) -> List[Dict]:
        """Сбор данных из Reddit"""
        
        if not self.reddit_client:
            return []
        
        posts = []
        
        try:
            subreddit = self.reddit_client.subreddit(subreddit_name)
            
            if query:
                # Поиск по ключевому слову
                search_results = subreddit.search(query, limit=limit)
            else:
                # Горячие посты
                search_results = subreddit.hot(limit=limit)
            
            for post in search_results:
                posts.append({
                    'text': post.title + ' ' + (post.selftext or ''),
                    'created_at': pd.to_datetime(post.created_utc, unit='s'),
                    'author_id': str(post.author) if post.author else 'deleted',
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit_name,
                    'source': 'reddit'
                })
        
        except Exception as e:
            print(f"Error collecting Reddit data: {e}")
        
        return posts

# Пример использования системы
if __name__ == "__main__":
    # Инициализация анализатора
    analyzer = MultilingualEmotionAnalyzer()
    
    # Пример данных для анализа
    sample_texts = [
        # Английские тексты
        "I absolutely love this new AI technology! It's going to change everything 😍 #AI #technology",
        "Great job on the economy... really wonderful how everything is falling apart 🙄",
        "Feeling anxious about climate change. What can we do to help our planet? 😟🌍",
        
        # Русские тексты  
        "Какая замечательная погода сегодня! Дождь, холод, просто прелесть 😒",
        "Очень рад новым технологиям в области ИИ! Будущее уже здесь 🚀",
        "Переживаю за экологию нашей планеты. Нужно что-то делать! 😰🌱",
        
        # Китайские тексты
        "这个新的人工智能技术真的很棒！我很兴奋 😊",
        "天气真好啊，又下雨了 😑", 
        "对气候变化感到担忧，我们应该做些什么 😟"
    ]
    
    # Анализ тональности и эмоций
    print("Анализ тональности и эмоций...")
    results = analyzer.analyze_comprehensive_sentiment(sample_texts)
    
    # Сохранение результатов
    results.to_csv('multilingual_emotion_analysis.csv', index=False, encoding='utf-8')
    print(f"Результаты сохранены в файл: multilingual_emotion_analysis.csv")
    
    # Кросс-культурное сравнение
    topic_keywords = ['AI', 'technology', 'ИИ', 'технология', '人工智能', '技术']
    cultural_comparison = analyzer.compare_cross_cultural_emotions(results, topic_keywords)
    
    # Вывод результатов сравнения
    print("\n=== КРОСС-КУЛЬТУРНОЕ СРАВНЕНИЕ ЭМОЦИЙ ===")
    for language, data in cultural_comparison.items():
        print(f"\n{language.upper()}:")
        print(f"  Средняя тональность: {data['sentiment_statistics']['avg_sentiment_score']:.3f}")
        print(f"  Эмоциональная интенсивность: {data['sentiment_statistics']['avg_emotional_intensity']:.3f}")
        print(f"  Уровень сарказма: {data['sentiment_statistics']['avg_sarcasm_score']:.3f}")
        print(f"  Размер выборки: {data['sentiment_statistics']['sample_size']}")
        
        if data['cultural_insights']:
            print("  Культурные особенности:")
            for insight in data['cultural_insights']:
                print(f"    - {insight}")
    
    # Создание дашборда
    print("\nСоздание интерактивного дашборда...")
    dashboard_fig = analyzer.create_emotion_dashboard(results)
    dashboard_fig.write_html('emotion_analysis_dashboard.html')
    print("Дашборд сохранен в: emotion_analysis_dashboard.html")
