import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from transformers import pipeline
import json
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud
import os
import warnings
import time
from io import BytesIO

##################################################################################################
# Configurazione della pagina
#################################################################################################
st.set_page_config(
    page_title="Academic Research Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializzazione delle variabili di sessione
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Stile CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 500;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #666;
        font-size: 0.8rem;
    }
    .dashboard-metrics {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .model-metric {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 23%;
        margin-bottom: 15px;
    }
    .topic-list {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 10px;
    }
    .dataset-info {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .performance-metric {
        background-color: #f5f9ff;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .performance-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .performance-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2196F3;
        margin-bottom: 5px;
    }
    .performance-label {
        font-size: 0.9rem;
        color: #555;
        font-weight: 500;
    }
    .recommendation-card {
        border-left: 4px solid #2196F3;
        padding-left: 15px;
        margin-bottom: 15px;
        transition: all 0.2s ease;
    }
    .recommendation-card:hover {
        background-color: #f8f9fa;
    }
    /* Nuovi stili migliorati */
    .article-card {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .article-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #2196F3;
    }
    
    .article-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 10px;
    }
    
    .tag {
        display: inline-block;
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .tab-container {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 20px;
        margin-top: 20px;
    }
    
    .example-button {
        padding: 5px 10px;
        background-color: #f0f7ff;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-right: 5px;
        cursor: pointer;
        border: 1px solid #d0e8ff;
        transition: all 0.2s ease;
    }
    
    .example-button:hover {
        background-color: #d0e8ff;
        border-color: #2196F3;
    }
    
    .compare-button {
        padding: 5px 10px;
        background-color: #E3F2FD;
        color: #1565C0;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-top: 5px;
        cursor: pointer;
        border: 1px solid #BBDEFB;
        transition: all 0.2s ease;
    }
    
    .compare-button:hover {
        background-color: #BBDEFB;
    }
    
    .comparison-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
    }
    
    .comparison-table th, .comparison-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    .comparison-table th {
        background-color: #f2f2f2;
    }
    
    .search-history-item {
        padding: 5px 8px;
        margin-bottom: 5px;
        background-color: #f5f5f5;
        border-radius: 4px;
        font-size: 0.9rem;
        cursor: pointer;
    }
    
    .search-history-item:hover {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Layout principale con logo ed intestazione
col1, col2 = st.columns([1, 5])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Luiss_logo.svg/320px-Luiss_logo.svg.png", width=120)

with col2:
    st.markdown('<div class="main-header">📚 Academic Research Analyzer</div>', unsafe_allow_html=True)
    st.markdown('##### Deloitte x Luiss: Data Science in Action Project')
    

# Sidebar con informazioni sul progetto
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Deloitte.svg/320px-Deloitte.svg.png", width=200)
    st.markdown("### About this project")
    st.info(
        "This application uses advanced NLP techniques to analyze academic "
        "publications. It can identify research themes through topic modeling, "
        "generate concise summaries, and recommend related articles."
    )
    
    # Features nella sidebar
    st.markdown("### Features")
    st.markdown("- 🔍 **Topic Discovery**: Identifies key research themes")
    st.markdown("- 📝 **Text Summarization**: Creates concise article summaries")
    st.markdown("- 📚 **Content Recommendation**: Suggests related articles")
    st.markdown("- 📊 **Topic Visualization**: Displays topic analysis charts")
    
    # Cronologia delle ricerche
    if st.session_state.search_history:
        st.markdown("### Recent Searches")
        for hist_item in st.session_state.search_history:
            hist_button = st.button(f"🔍 {hist_item[:25]}..." if len(hist_item) > 25 else f"🔍 {hist_item}", 
                                    key=f"hist_{hist_item}", 
                                    help=f"Search for: {hist_item}")
            if hist_button:
                st.session_state.article_title = hist_item
                st.experimental_rerun()
    
    # Model metrics in sidebar
    st.markdown("### Model Performance")
    
    # Load or generate model metrics
    try:
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except FileNotFoundError:
        st.error("❌ Model metrics file not found. Please run the Jupyter notebook first to generate metrics.")
        model_metrics = {
            "precision_at_k": 0.75,
            "precision_conf_interval": 0.05,
            "topic_coherence": 0.65,
            "topic_diversity": 0.82,
            "summarization_rouge": 0.45
        }

    # Display metrics
    precision = model_metrics.get('precision_at_k', 0)
    conf_interval = model_metrics.get('precision_conf_interval', 0)

    # Verifica che entrambi i valori siano numerici
    precision_str = f"{precision:.2f}" if isinstance(precision, (int, float)) else str(precision)
    conf_str = f"{conf_interval:.2f}" if isinstance(conf_interval, (int, float)) else str(conf_interval)

    # Combina i valori formattati
    st.metric("Precision@5", f"{precision_str} ± {conf_str}")

    topic_coherence = model_metrics.get('topic_coherence', 0)
    # Assicurati che il valore sia un numero prima di formattarlo
    if isinstance(topic_coherence, (int, float)):
        st.metric("Topic Coherence", f"{topic_coherence:.2f}")
    else:
        # Se non è un numero, converti esplicitamente o mostra un valore di default
        st.metric("Topic Coherence", str(topic_coherence))

    topic_diversity = model_metrics.get('topic_diversity', 0)
    if isinstance(topic_diversity, (int, float)):
        st.metric("Topic Diversity", f"{topic_diversity:.2f}")
    else:
        st.metric("Topic Diversity", str(topic_diversity))
    
    
    st.markdown("### Project Team")
    st.markdown("Simone Moroni, Gabriele Gogli, Matteo Piccirilli")
    st.markdown("Deloitte x Luiss Data Science in Action - 2025")

##################################################################################################
# Funzioni di caricamento del dataset e dei modelli
#################################################################################################
# Verifica che esistono i file necessari
@st.cache_data
def check_required_files():
    """Verifica tutti i file necessari e restituisce informazioni sullo stato"""
    required_files = {
        'articles_with_summaries.parquet': 'dataset',
        'best_bayes_topic_model.pkl': 'topic model',
        'embedding_model.pkl': 'embedding model',
        'embeddings.pkl': 'article embeddings',
        'topic_assignments.pkl': 'topic assignments',
        'topic_labels.json': 'topic labels',
        'model_metrics.json': 'model metrics'
    }
    
    status = {}
    
    for file_path, file_type in required_files.items():
        status[file_path] = os.path.exists(file_path)
    
    return status

# Uso nella parte iniziale dell'app
file_status = check_required_files()
missing_files = [f for f, exists in file_status.items() if not exists]

if missing_files:
    st.sidebar.warning(f"⚠️ Missing files: {', '.join(missing_files)}. Some features may not be available.")

    # Verifica se mancano file critici
    critical_files = ['articles_with_summaries.parquet', 'best_bayes_topic_model.pkl']
    if any(f in missing_files for f in critical_files):
        st.warning("Core functionality will be limited due to missing critical files.")


# Caricamento del dataset
@st.cache_data
def load_dataset():
    """
    Carica il dataset con gestione avanzata degli errori e feedback visivo.
    """
    try:
        # Visualizzazione di un messaggio di caricamento
        with st.spinner("Loading dataset..."):
            df = pd.read_parquet("articles_with_summaries.parquet")
            
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please make sure 'articles_with_summaries.parquet' exists in the current directory.")
        # Creazione di un DataFrame vuoto con le colonne necessarie
        return pd.DataFrame(columns=['title', 'abstract', 'authors', 'doi', 'publication_year', 'summary', 'text'])
    except pd.errors.ParserError:
        st.error("❌ Error parsing the dataset file. The file might be corrupted.")
        return pd.DataFrame()
    except PermissionError:
        st.error("❌ Permission denied when trying to access the dataset file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error loading dataset: {str(e)}")
        return pd.DataFrame()

# Caricamento delle etichette dei topic (da un file JSON creato nel notebook Jupyter)
@st.cache_data
def load_topic_labels():
    """
    Carica le etichette dei topic dal file JSON generato nel notebook Jupyter.
    Non effettua alcuna generazione di etichette in Streamlit.
    """
    try:
        with open('topic_labels.json', 'r') as f:
            labels = json.load(f)
        # Mostra un messaggio di successo solo se troviamo effettivamente delle etichette
        if labels:
            st.sidebar.success(f"✅ Loaded {len(labels)} topic labels from external file")
        return labels
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Topic labels file not found. Please run the Jupyter notebook first to generate topic labels.")
        return {}
    except json.JSONDecodeError:
        st.sidebar.error("❌ Invalid JSON format in the topic labels file.")
        return {}
    except Exception as e:
        st.sidebar.error(f"❌ Error loading topic labels: {str(e)}")
        return {}

# Caricamento del modello di summarization
@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():
    """
    Carica il modello di summarization con gestione degli errori.
    """
    try:
        from transformers import pipeline
        model_name = "Falconsai/text_summarization"
        # Impostazione ottimizzata per ridurre l'uso di memoria
        summarizer = pipeline("summarization", model=model_name, device=-1)  # -1 = CPU
        return summarizer
    except ModuleNotFoundError:
        st.error("❌ Required module 'transformers' not found. Please install it with 'pip install transformers'.")
        return None
    except OSError as e:
        st.error(f"❌ Error downloading the summarization model: {str(e)}")
        st.warning("Summarization features will not be available.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading summarization model: {str(e)}")
        st.warning("Summarization features will not be available.")
        return None

# Caricamento dei modelli NLP
@st.cache_resource
def load_models():
    """
    Carica i modelli NLP con verifica approfondita e gestione degli errori.
    """
    try:
        with st.spinner("Loading NLP models..."):
            try:
                # Caricamento del modello di topic modeling
                topic_model = joblib.load('best_bayes_topic_model.pkl')
                
                # Verifica che l'embedding_model sia presente e valido
                if not hasattr(topic_model, 'embedding_model') or topic_model.embedding_model is None:
                    st.warning("⚠️ The topic model doesn't have a valid embedding model. Topic analysis will be limited.")
                
                # Verifica l'integrità del modello
                if not hasattr(topic_model, 'get_topic'):
                    st.error("❌ Il topic model caricato non ha il metodo 'get_topic'. Potrebbe essere corrotto.")
                    topic_model = None
            except FileNotFoundError:
                st.error("❌ Topic model file not found. Please run the Jupyter notebook first to generate the model.")
                topic_model = None
            except Exception as e:
                st.error(f"❌ Error loading topic model: {str(e)}")
                topic_model = None
            
            try:
                # Caricamento del modello di embedding
                embedding_model = joblib.load('embedding_model.pkl')
            except FileNotFoundError:
                st.warning("⚠️ Embedding model file not found. Recommendation features will be limited.")
                embedding_model = None
            except Exception as e:
                st.error(f"❌ Error loading embedding model: {str(e)}")
                embedding_model = None
            
            try:
                # Caricamento degli embedding degli articoli
                article_embeddings = joblib.load('embeddings.pkl')
            except FileNotFoundError:
                st.warning("⚠️ Article embeddings file not found. Recommendations will not be accurate.")
                article_embeddings = None
            except Exception as e:
                st.error(f"❌ Error loading article embeddings: {str(e)}")
                article_embeddings = None
            
            try:
                # Preassegnazione dei topic agli articoli
                topic_assignments = joblib.load('topic_assignments.pkl')
            except FileNotFoundError:
                st.warning("⚠️ Topic assignments file not found. Topic filtering will not be available.")
                topic_assignments = None
            except Exception as e:
                st.error(f"❌ Error loading topic assignments: {str(e)}")
                topic_assignments = None
            
            # Verifica compatibilità di embedding model e article embeddings
            if embedding_model is not None and article_embeddings is not None:
                try:
                    # Genera un embedding di test per verificare la compatibilità
                    test_embedding = embedding_model.encode("test", convert_to_tensor=True)
                    if hasattr(article_embeddings, 'shape') and len(article_embeddings) > 0:
                        # Verifica che la dimensione dell'embedding sia compatibile
                        first_emb = article_embeddings[0] if isinstance(article_embeddings, list) else article_embeddings[0]
                        if len(first_emb) != len(test_embedding):
                            st.warning("⚠️ Dimensioni degli embedding non compatibili. Il sistema di raccomandazione potrebbe non funzionare correttamente.")
                except Exception as e:
                    st.warning(f"⚠️ Test di compatibilità dei modelli fallito: {str(e)}")
            
            return topic_model, embedding_model, article_embeddings, topic_assignments
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

##################################################################################################
# Funzioni di estrazione dell'abstract
#################################################################################################

def extract_abstract(inverted_index):
    """
    Estrae il testo dell'abstract da un indice invertito di OpenAlex.
    """
    if not inverted_index:
        return None
        
    if isinstance(inverted_index, dict):
        # Crea un dizionario temporaneo con le posizioni come chiavi e le parole come valori
        positions_to_words = {}
        for word, positions in inverted_index.items():
            for position in positions:
                positions_to_words[position] = word
        
        # Ricostruisci il testo ordinando le parole per posizione
        sorted_positions = sorted(positions_to_words.keys())
        text = ' '.join(positions_to_words[pos] for pos in sorted_positions)
        
        return text
    
    # Se non è un dizionario, potrebbe essere già una stringa o None
    return inverted_index

##################################################################################################
# Provo ad estrapolare l'abstract, prima tramite open alex e poi cercando su altre API oppure scaricando direttamente il pdf...
#################################################################################################


def search_openalex(title):
    """
    Versione efficace e semplificata per cercare un articolo solo su OpenAlex.
    
    Args:
        title (str): Titolo dell'articolo da cercare
        
    Returns:
        dict: Dati dell'articolo con abstract o None
    """
    # Sanitizza il titolo per la ricerca
    sanitized_title = title.strip()
    
    # Prova prima con una ricerca esatta (tra virgolette)
    url = "https://api.openalex.org/works"
    
    # Prova diverse strategie di ricerca
    search_strategies = [
        {"filter": f"title.search:\"{sanitized_title}\""},  # Ricerca esatta
        {"filter": f"title.search:{sanitized_title}"},      # Ricerca standard
        {"search": sanitized_title}                         # Ricerca full-text
    ]
    
    headers = {
        "User-Agent": "Academic-Research-Analyzer/1.0 (mailto:your@email.com)"
    }
    
    for strategy in search_strategies:
        try:
            response = requests.get(url, params=strategy, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    article = results[0]
                    
                    # Estrai l'abstract
                    abstract = extract_abstract(article.get("abstract_inverted_index"))
                    
                    # Se non abbiamo trovato un abstract, prova a cercarlo in altri campi
                    if not abstract:
                        # Alcuni articoli hanno l'abstract in altri formati
                        abstract = article.get("abstract")
                    
                    
                    # Costruisci il risultato
                    return {
                        "title": article.get("title"),
                        "abstract": abstract,
                        "doi": article.get("doi", "").replace("https://doi.org/", ""),
                        "year": article.get("publication_year"),
                        "authors": [a["author"]["display_name"] for a in article.get("authorships", [])],
                        "abstract_source": "OpenAlex"
                    }
                
                # Se questa strategia non ha trovato risultati, passiamo alla prossima
                
        except requests.RequestException as e:
            st.error(f"❌ Network error during OpenAlex search: {str(e)}")
        except ValueError as e:
            st.error(f"❌ Error parsing OpenAlex response: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error during OpenAlex search: {str(e)}")
    
    # Se siamo arrivati qui, nessuna strategia ha funzionato
    st.warning("⚠️ Article not found on OpenAlex")
    return None
##################################################################################################
# Topic Modeling e Visualizzazione
#################################################################################################

# Funzione per ottenere l'etichetta di un topic dal dizionario di etichette precaricato
def get_topic_label(topic_id, topic_model, topic_labels=None):
    """
    Restituisce un'etichetta leggibile per un topic dalle etichette precaricate.
    Non genera etichette, ma usa solo quelle precaricate da Jupyter.
    """
    # Converti topic_id in stringa per il confronto con le chiavi del dizionario
    topic_id_str = str(topic_id)
    
    # Controlla se esiste un'etichetta personalizzata
    if topic_labels and topic_id_str in topic_labels:
        return topic_labels[topic_id_str]
    
    # Se non c'è un'etichetta, restituisci un valore di fallback basic
    if topic_id == -1:
        return "Outlier"
    else:
        return f"Topic {topic_id}"

# Creazione di wordcloud (cached)
@st.cache_data
def create_wordcloud(topic_terms):
    """Crea e renderizza un wordcloud dai termini del topic."""
    wc_data = {term: weight for term, weight in topic_terms}
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate_from_frequencies(wc_data)
    
    # Crea la figura
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Funzione per visualizzare i topic di un articolo
def visualize_article_topics(article_text, topic_model, topic_labels=None):
    """
    Visualizza i topic di un articolo con grafici interattivi.
    Versione migliorata che forza l'assegnazione del topic più simile per gli outlier.
    """
    # Inizializza topic_id come -1 (outlier) come valore di default
    topic_id = -1
    if topic_model is None:
        st.warning("⚠️ Topic model not available. Some visualizations will not be shown.")
        return topic_id
    
    # Verifica che l'embedding_model all'interno del topic_model sia valido
    if not hasattr(topic_model, 'embedding_model') or topic_model.embedding_model is None:
        st.error("❌ Topic model's embedding model is not available or properly loaded.")
        return topic_id
    
    try:
        # Trasformazione standard senza parametri extra
        topics, topic_probs = topic_model.transform([article_text])
        topic_id = topics[0]
        
        # Forza l'assegnazione a un topic se è un outlier
        assigned_forcefully = False
        original_topic = topic_id
        
        if topic_id == -1:
            # Verifica il formato di topic_probs con controlli più robusti
            if hasattr(topic_probs, 'shape'):
                # Array numpy
                if len(topic_probs.shape) > 1 and topic_probs.shape[1] > 0:
                    max_prob_index = topic_probs[0].argmax()
                    topic_id = max_prob_index
                    assigned_forcefully = True
                st.info(f"📌 The article was initially classified as an outlier, but has been assigned to the most similar topic: {topic_id}")
            # Se il formato è diverso e topic_probs è una lista/tupla di array
            elif isinstance(topic_probs, (list, tuple)) and len(topic_probs) > 0 :
                if hasattr(topic_probs[0], 'argmax'):
                    max_prob_index = topic_probs[0].argmax()
                    topic_id = max_prob_index
                    assigned_forcefully = True
                st.info(f"📌 The article was initially classified as an outlier, but has been assigned to the most similar topic: {topic_id}")
        # Estrazione delle parole chiave del topic
        topic_keywords = topic_model.get_topic(topic_id)
        
        # Se non abbiamo keywords per questo topic, prova altri topic
        if not topic_keywords:
            st.warning(f"⚠️ No significant keywords found for topic {topic_id}.")
            
            # Prova i topic principali fino a trovarne uno con keywords
            topic_info = topic_model.get_topic_info()
            candidate_topics = topic_info[topic_info['Topic'] != -1].sort_values('Count', ascending=False)['Topic'].tolist()
            
            for candidate_topic in candidate_topics:
                candidate_keywords = topic_model.get_topic(candidate_topic)
                if candidate_keywords:
                    topic_id = candidate_topic
                    topic_keywords = candidate_keywords
                    assigned_forcefully = True
                    st.info(f"📌 Using topic {topic_id} because it contains significant keywords")
                    break
            
            # Se ancora non abbiamo keywords, prova sequenzialmente tutti i topic
            if not topic_keywords:
                for i in range(min(50, len(topic_model.get_topic_info()))):
                    if i != -1:  # Salta il topic outlier
                        candidate_keywords = topic_model.get_topic(i)
                        if candidate_keywords:
                            topic_id = i
                            topic_keywords = candidate_keywords
                            assigned_forcefully = True
                            st.info(f"📌 Using topic {topic_id} as a last resort")
                            break
        
        # Se ancora non abbiamo keywords, mostra un errore e termina
        if not topic_keywords:
            st.error("❌ Unable to find significant keywords for any topic. Cannot proceed with the analysis.")
            return topic_id
        
        # Ottieni l'etichetta leggibile del topic
        topic_label = get_topic_label(topic_id, topic_model, topic_labels)
        
        # Visualizza chiaramente che il topic è stato assegnato manualmente
        if assigned_forcefully:
            st.markdown(f"### {topic_label} (manually assigned)")
            st.write("The article was manually assigned to this topic because it was initially classified as an outlier.")
        else:
            st.markdown(f"### {topic_label}")
        
        # Creazione del DataFrame per la visualizzazione
        keywords_df = pd.DataFrame(topic_keywords, columns=["Keyword", "Score"]).sort_values(by="Score", ascending=False)
        
        # Visualizzazione delle parole chiave con Plotly
        fig = px.bar(keywords_df, 
                    x="Score", 
                    y="Keyword", 
                    orientation="h", 
                    title="🔍 Key Topic Terms", 
                    color="Score", 
                    color_continuous_scale="blues",
                    labels={"Score": "Relevance Score", "Keyword": ""},
                    height=400)
        
        fig.update_layout(
            xaxis_title="Relevance Score",
            yaxis_title="",
            font=dict(family="Arial", size=14),
            hoverlabel=dict(bgcolor="white", font_size=14),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Aggiungi visualizzazione grafico a torta dei topic
        if hasattr(topic_probs, 'shape') and topic_probs.shape[1] > 0:
            # Ottieni i top 5 topic più probabili
            top_n = min(5, topic_probs.shape[1])
            top_indices = topic_probs[0].argsort()[-top_n:][::-1]
            top_probs = topic_probs[0][top_indices]
            
            # Crea un DataFrame per la visualizzazione
            top_topics_df = pd.DataFrame({
                "Topic": [get_topic_label(idx, topic_model, topic_labels) for idx in top_indices],
                "Probability": top_probs
            })
            
            # Crea un grafico a torta interattivo
            fig = px.pie(top_topics_df, 
                        values='Probability', 
                        names='Topic',
                        title='Topic Distribution',
                        hole=0.4)
                        
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostra informazioni sul topic
        if assigned_forcefully:
            st.markdown(f"**Original Topic ID**: {original_topic} (outlier)")
            st.markdown(f"**Assigned Topic ID**: {topic_id}")
        else:
            st.markdown(f"**Topic ID**: {topic_id}")
        
        # Estrai altri articoli dello stesso topic per mostrare il contesto
        topic_info = topic_model.get_topic_info()
        topic_size = topic_info.loc[topic_info['Topic'] == topic_id, 'Count'].values[0] if topic_id in topic_info['Topic'].values else 0
        
        st.markdown(f"**Topic size**: {topic_size} articles")
        
        # Se l'utente vuole vedere più dettagli sul perché questa assegnazione è stata fatta
        if assigned_forcefully:
            with st.expander("ℹ️ Topic assignment details"):
                st.write("""
                This article was initially classified as an 'outlier', which means that the topic model 
                didn't assign it with certainty to any specific thematic cluster.
                
                To provide you with useful analysis anyway, we've forced the assignment to the most similar topic based on the 
                text characteristics. This allows you to see which topic is closest to the article's content,
                even if it's not a perfect match.
                """)
                
                # Se abbiamo le probabilità, mostriamole
                if hasattr(topic_probs, 'shape') and topic_probs.shape[1] > 0:
                    # Ottieni i top 5 topic più probabili
                    top_indices = topic_probs[0].argsort()[-5:][::-1]
                    top_probs = topic_probs[0][top_indices]
                    
                    # Crea un DataFrame per mostrarli
                    top_topics_df = pd.DataFrame({
                        "Topic ID": top_indices,
                        "Probability": top_probs,
                        "Label": [get_topic_label(idx, topic_model, topic_labels) for idx in top_indices]
                    })
                    
                    st.write("### Top 5 most probable topics:")
                    st.dataframe(top_topics_df)
        
    except AttributeError as e:
        st.error(f"❌ Error in topic model: {str(e)}")
        st.info("This usually happens when the models weren't correctly loaded. Please run the Jupyter notebook first to generate all required models.")
        return topic_id
    except Exception as e:
        st.error(f"❌ Error visualizing topics: {str(e)}")
        import traceback
        st.code(traceback.format_exc())  # Mostra lo stack trace completo
        return topic_id
        
    return topic_id

##################################################################################################
# Recommendation System
#################################################################################################

# Sistema di raccomandazione
@st.cache_data
def get_article_recommendations(article_text, df, _embedding_model, _article_embeddings, topic_id=None, top_k=5):
    """
    Ottiene raccomandazioni di articoli similari con debug esteso e gestione degli errori robusta.
    Le variabili con underscore (_) non vengono hashate da Streamlit.
    """
    import numpy as np
    import torch
    
    # Debug information
    st.write("🔍 Finding similar articles...")
    
    # Validate inputs
    if _embedding_model is None:
        st.warning("⚠️ Embedding model not available")
        return df.sample(min(top_k, len(df)))
        
    if _article_embeddings is None:
        st.warning("⚠️ Article embeddings not available")
        return df.sample(min(top_k, len(df)))
    
    if df is None or len(df) == 0:
        st.warning("⚠️ Dataset is empty")
        return pd.DataFrame()
    
    # Check if we have a valid topic ID and if topic column exists in DataFrame
    topic_based_recommendation = False
    if topic_id is not None and topic_id != -1 and 'topic' in df.columns:
        similar_topic_df = df[df['topic'] == topic_id]
        if len(similar_topic_df) >= top_k:
            st.info(f"📚 Recommending {top_k} articles from topic {topic_id}")
            topic_based_recommendation = True
            return similar_topic_df.sample(min(top_k, len(similar_topic_df)))
    
    # If we reach here, we need to use embedding-based recommendation
    st.info("📚 Using semantic similarity for recommendations")
    
    try:
        # Generate embedding for the input article
        st.write("Generating embedding for input article...")
        
        # Create embedding with verbose error handling
        try:
            # Try different approaches to handle the article text
            if isinstance(article_text, str) and len(article_text.strip()) > 0:
                article_embedding = _embedding_model.encode(article_text, convert_to_tensor=True)
                st.write("✅ Successfully generated embedding for input article")
            else:
                st.warning("⚠️ Input article text is empty or invalid")
                return df.sample(min(top_k, len(df)))
                
        except Exception as e:
            st.error(f"❌ Error generating embedding: {str(e)}")
            return df.sample(min(top_k, len(df)))
        
        # Debug information
        st.write(f"Input embedding shape: {article_embedding.shape}")
        
        # Normalize the embedding if needed
        try:
            if hasattr(torch.nn.functional, 'normalize'):
                article_embedding = torch.nn.functional.normalize(article_embedding.unsqueeze(0), dim=1)
                st.write("✅ Normalized input embedding")
            else:
                # If we can't normalize, at least ensure it's 2D
                if len(article_embedding.shape) == 1:
                    article_embedding = article_embedding.unsqueeze(0)
        except Exception as e:
            st.error(f"❌ Error normalizing embedding: {str(e)}")
            # Try to continue without normalization
            if len(article_embedding.shape) == 1:
                article_embedding = article_embedding.unsqueeze(0)
        
        # Convert stored embeddings to tensor if needed
        st.write(f"Article embeddings type: {type(_article_embeddings)}")
        
        try:
            if isinstance(_article_embeddings, list):
                # Convert list of arrays to tensor
                emb_tensor = torch.tensor(np.array(_article_embeddings))
            elif isinstance(_article_embeddings, np.ndarray):
                emb_tensor = torch.tensor(_article_embeddings)
            elif isinstance(_article_embeddings, torch.Tensor):
                emb_tensor = _article_embeddings
            else:
                st.error(f"❌ Unsupported embedding format: {type(_article_embeddings)}")
                return df.sample(min(top_k, len(df)))
                
            st.write(f"Embeddings tensor shape: {emb_tensor.shape}")
            
            # Check if dimensions match
            if emb_tensor.shape[1] != article_embedding.shape[1]:
                st.error(f"❌ Dimension mismatch: {emb_tensor.shape[1]} vs {article_embedding.shape[1]}")
                return df.sample(min(top_k, len(df)))
                
        except Exception as e:
            st.error(f"❌ Error processing embeddings: {str(e)}")
            return df.sample(min(top_k, len(df)))
        
        # Calculate similarity
        try:
            # Usa batch processing per calcoli più efficienti
            st.write("Calculating similarity...")
            batch_size = 1000  # Aumenta o diminuisci in base alla memoria disponibile
            similarities = []

            for i in range(0, emb_tensor.shape[0], batch_size):
                batch = emb_tensor[i:i+batch_size]
                batch_sim = torch.nn.functional.cosine_similarity(
                    batch, 
                    article_embedding, 
                    dim=1
                ).cpu().numpy()
                similarities.extend(batch_sim)
            
            similarities = np.array(similarities)
            
            st.write(f"Generated {len(similarities)} similarity scores")
            
            # Debug: Show top similarity scores
            top_scores = sorted(similarities, reverse=True)[:5]
            st.write(f"Top 5 similarity scores: {[f'{score:.4f}' for score in top_scores]}")
            
        except Exception as e:
            st.error(f"❌ Error calculating similarity: {str(e)}")
            return df.sample(min(top_k, len(df)))
        
        # Check DataFrame and embeddings alignment
        if len(similarities) != len(df):
            st.warning(f"⚠️ Number of similarity scores ({len(similarities)}) doesn't match DataFrame length ({len(df)})")
            # Try to use as many as we can
            similarities = similarities[:min(len(similarities), len(df))]
        
        # Find most similar articles
        try:
            # Get indices of most similar articles (descending order)
            indices = np.argsort(similarities)[::-1]
            
            # Remove exact matches (similarity > 0.999)
            mask = similarities[indices] < 0.999
            filtered_indices = indices[mask]
            
            # If we filtered out everything, fall back to original indices
            if len(filtered_indices) == 0:
                filtered_indices = indices
                
            # Take only top_k
            result_indices = filtered_indices[:top_k]
            result_similarities = similarities[result_indices]
            
            # Debug
            st.write(f"Selected {len(result_indices)} articles for recommendation")
            
            # Check if we have valid indices
            if len(result_indices) == 0:
                st.warning("⚠️ No similar articles found")
                return df.sample(min(top_k, len(df)))
                
            # Create recommendation DataFrame
            try:
                recommended_df = df.iloc[result_indices].copy()
                recommended_df["similarity"] = [float(s) for s in result_similarities]  
                st.write(f"✅ Successfully created recommendation DataFrame with {len(recommended_df)} articles")
                return recommended_df
            except Exception as e:
                st.error(f"❌ Error creating recommendation DataFrame: {str(e)}")
                return df.sample(min(top_k, len(df)))
                
        except Exception as e:
            st.error(f"❌ Error finding similar articles: {str(e)}")
            return df.sample(min(top_k, len(df)))
            
    except Exception as e:
        st.error(f"❌ Unexpected error in recommendation system: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return df.sample(min(top_k, len(df)))

##################################################################################################
# Analisi Avanzata
#################################################################################################

# Analisi avanzata
def show_advanced_analysis(topic_model, article_text, topic_labels=None):
    """
    Mostra analisi avanzate dei topic con visualizzazioni interattive.
    """
    st.write("Starting advanced analysis for the inserted article...")
    
    if topic_model is None:
        st.warning("⚠️ Topic model not available. Advanced analysis cannot be performed.")
        return
    
    try:
        topic_info = topic_model.get_topic_info() 
        topic_info = topic_info.rename(columns={
            'Topic': 'Topic ID',
            'Count': 'Document Count',
            'Name': 'Representative Terms'
        })
        
        # Calcola il totale dei documenti per le percentuali
        total_docs = topic_info['Document Count'].sum()
        
        # Analisi del topic
        topics, topic_probs = topic_model.transform([article_text])
        topic_id = topics[0]
        st.write(f"Document assigned to topic {topic_id}")
         # Analisi del topic dell'articolo corrente
        st.markdown("### 🔎 Current Article Topic Analysis")
        
        if topic_id == -1:
            st.warning("⚠️ The current article doesn't belong to any specific topic cluster. Trying to assign it to the most similar topic...")
        
            # Verifica se abbiamo probabilità di topic disponibili
            if isinstance(topic_probs, np.ndarray) and len(topic_probs.shape) > 1 and topic_probs.shape[1] > 0:
            # Trova l'indice del topic con la probabilità più alta
                max_prob_index = topic_probs[0].argmax()
                topic_id = max_prob_index
                assigned_forcefully = True
                st.info(f"📌 The article was initially classified as an outlier, but has been assigned to the most similar topic: {topic_id}")
            # Se il formato è diverso e topic_probs è una lista/tupla di array
            elif isinstance(topic_probs, (list, tuple)) and len(topic_probs) > 0 and isinstance(topic_probs[0], np.ndarray):
                max_prob_index = topic_probs[0].argmax()
                topic_id = max_prob_index
                assigned_forcefully = True
                st.info(f"📌 The article was initially classified as an outlier, but has been assigned to the most similar topic: {topic_id}")
        
        # Visualizzazione dei termini del topic
        topic_terms = topic_model.get_topic(topic_id)
            
        # Etichetta leggibile
        topic_label = get_topic_label(topic_id, topic_model, topic_labels)
            
        # Visualizzazione a colonne
        cols = st.columns(3)
            
        with cols[0]:
            st.metric("Topic ID", topic_id)
            
        with cols[1]:
            topic_size = topic_info.loc[topic_info['Topic ID'] == topic_id, 'Document Count'].values[0] if topic_id in topic_info['Topic ID'].values else 0
            st.metric("Topic Size", f"{topic_size} articles")
            
        with cols[2]:
                if topic_size > 0 and total_docs > 0:
                    topic_percentage = (topic_size / total_docs * 100).round(2)
                    st.metric("% of Corpus", f"{topic_percentage}%")
                else:
                    st.metric("% of Corpus", "N/A")
            
            # Visualizzazione dei termini rappresentativi
        st.markdown(f"#### Topic Label: {topic_label}")
        st.markdown("#### Most Representative Terms")
            
        # Usa la funzione cached per il wordcloud
        fig = create_wordcloud(topic_terms)
        st.pyplot(fig)
            
    except Exception as e:
        st.error(f"❌ Error in advanced analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())  # Questo mostrerà lo stack trace completo

##################################################################################################
# Interfaccia utente principale
#################################################################################################

# Caricamento dei dati e modelli
df = load_dataset()
topic_labels = load_topic_labels()  # Carica le etichette dal file JSON generato da Jupyter
topic_model, embedding_model, article_embeddings, topic_assignments = load_models()
summarizer = load_summarizer()

# Assegna i topic precomputati al DataFrame se disponibili
if topic_assignments is not None and df is not None and len(df) > 0:
    if len(topic_assignments) == len(df):
        df['topic'] = topic_assignments
        st.sidebar.success("✅ Topic assignments loaded successfully")
    else:
        st.sidebar.warning(f"⚠️ Topic assignments length ({len(topic_assignments)}) doesn't match dataset length ({len(df)})")

if article_embeddings is not None:
    st.sidebar.success(f"✅ Loaded embeddings for {article_embeddings.shape[0] if hasattr(article_embeddings, 'shape') else len(article_embeddings)} articles")
    
    # Verify embeddings format
    if isinstance(article_embeddings, list):
        st.sidebar.info("Converting embeddings list to tensor...")
        article_embeddings = np.array(article_embeddings)
        st.sidebar.success("Embeddings converted to numpy array")
        
    # Verify embeddings-dataset alignment    
    if hasattr(article_embeddings, 'shape') and article_embeddings.shape[0] != len(df):
        st.sidebar.warning(f"⚠️ Embeddings count ({article_embeddings.shape[0]}) doesn't match dataset size ({len(df)})")
else:
    st.sidebar.warning("⚠️ Article embeddings not loaded - recommendations will be random")

# Pulizia e preparazione dei dati
df["abstract"] = df["abstract"].fillna("").astype(str)
df["doi"] = df["doi"].fillna("").astype(str)

# Inizializzazione della variabile di confronto degli articoli
if 'compared_articles' not in st.session_state:
    st.session_state.compared_articles = []

# Layout principale dell'applicazione
tabs = st.tabs(["🔎 Article Analysis", "📊 Topic Explorer", "📚 Browse Dataset"])

with tabs[0]:
    # Tab di ricerca articoli - FOCUSED ONLY ON ARTICLE ANALYSIS
    st.markdown("## 🔎 Article Analysis")
    
    # Input per la ricerca
    # Aggiungi un esempio iniziale di articolo per facilitare l'uso
    article_title = st.text_input("📝 Enter the title of an academic article:", 
                                key="article_title")

 

        
    # Opzioni avanzate
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            apply_summarization = st.checkbox("📝 Summarize abstract", value=True)
            max_summary_length = st.slider("Maximum summary length", 50, 250, 150)
        
        with col2:
            show_advanced = st.checkbox("📊 Show advanced topic analysis", value=False)
            recommended_count = st.slider("Number of recommended articles", 3, 10, 5)
    
    # Pulsante di ricerca
    search_col1, search_col2 = st.columns([1, 4])
    
    with search_col1:
        search_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with search_col2:
        st.markdown("")  # Spazio vuoto per allineamento
    
    # Logica di ricerca
    if search_button and article_title:
        # Inizializza le barre di progresso e di stato
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Aggiorna gli indicatori durante il flusso di lavoro
        status_text.text("🔍 Searching in local dataset...")
        progress_bar.progress(10)
        
        # Cerco prima nel dataset locale
        st.write("🔍 Searching in local dataset...")
        matched_article = df[df['title'].str.lower().str.strip() == article_title.lower().strip()]

        progress_bar.progress(30)

        if not matched_article.empty:
            st.success("✅ Article found in local dataset!")
            article_data = matched_article.iloc[0].to_dict()
            article_text = article_data['abstract']
            
            # Aggiungi alla cronologia se non è già presente
            if article_title not in st.session_state.search_history:
                st.session_state.search_history.append(article_title)
                # Limita la cronologia a 10 elementi
                if len(st.session_state.search_history) > 10:
                    st.session_state.search_history.pop(0)
            
            # Visualizzazione delle informazioni sull'articolo
            with st.container():
                st.markdown('<div class="article-card">', unsafe_allow_html=True)
                
                st.markdown(f"### 📚 {article_data.get('title', article_title)}")
                
                # Dettagli dell'articolo
                col1, col2 = st.columns(2)
                
                with col1:
                    if isinstance(article_data.get('authors'), list):
                        authors_str = ", ".join(article_data['authors'][:3])
                        if len(article_data['authors']) > 3:
                            authors_str += f" and {len(article_data['authors']) - 3} more"
                        st.markdown(f"**Authors:** {authors_str}")
                    
                    if article_data.get('publication_year'):
                        st.markdown(f"**Year:** {article_data['publication_year']}")
                
                with col2:
                    if article_data.get('doi'):
                        st.markdown(f"[🔗 Open original article](https://doi.org/{article_data['doi']})")
                    
                    if isinstance(article_data.get('topics'), list) and article_data['topics']:
                        topics_str = ", ".join(article_data['topics'][:3])
                        st.markdown(f"**Topics:** {topics_str}")
                
                # Abstract o Riassunto
                abstract_tab, summary_tab = st.tabs(["Original Abstract", "AI Summary"])
                
                with abstract_tab:
                    st.markdown(f"{article_text}")
                
                with summary_tab:
                    if apply_summarization and summarizer is not None:
                        if 'summary' in article_data and article_data['summary']:
                            st.markdown(f"{article_data['summary']}")
                        else:
                            with st.spinner("Generating summary..."):
                                try:
                                    summary = summarizer(article_text, max_length=max_summary_length, min_length=50, do_sample=False)[0]['summary_text']
                                    st.markdown(f"{summary}")
                                except Exception as e:
                                    st.error(f"❌ Error generating summary: {str(e)}")
                                    st.markdown(article_text[:500] + "...")
                    else:
                        st.info("Enable summarization in Advanced Options to see AI-generated summary")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress(50)
            status_text.text("📊 Analyzing topics...")
            
            # Visualizzazione dei topic
            st.markdown("### 🔍 Topic Analysis")
            article_topic_id = visualize_article_topics(article_text, topic_model, topic_labels)
            
            progress_bar.progress(70)
            status_text.text("🔍 Finding similar articles...")
            
            # Ricerca articoli simili
            st.markdown("### 📚 Similar Articles")
            with st.spinner("Finding similar articles..."):
                # Add this line to show the topic ID being used
                st.write(f"Using topic ID {article_topic_id} for recommendations")
    
                recommended_articles = get_article_recommendations(article_text, df, embedding_model, 
                                                                 article_embeddings, topic_id=article_topic_id, top_k=recommended_count)
    
                # Check if we got recommendations
                if recommended_articles is None or len(recommended_articles) == 0:
                    st.warning("⚠️ No similar articles found")
                else:
                    st.success(f"✅ Found {len(recommended_articles)} similar articles")
                    
                    # Create tabs for different recommendation views
                    rec_tabs = st.tabs(["Card View", "Table View"])
                    
                    with rec_tabs[0]:
                        # Visualizzazione degli articoli raccomandati (Card View)
                        for idx, row in recommended_articles.iterrows():
                            similarity_pct = int(row.get('similarity', 0) * 100)
                            
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                with st.expander(f"📄 {row['title']} (Similarity: {similarity_pct}%)"):
                                    # Dettagli dell'articolo raccomandato
                                    rcol1, rcol2 = st.columns(2)
                                    
                                    with rcol1:
                                        if isinstance(row.get('authors'), list):
                                            authors_str = ", ".join(row['authors'][:3])
                                            if len(row['authors']) > 3:
                                                authors_str += f" and {len(row['authors']) - 3} more"
                                            st.markdown(f"**Authors:** {authors_str}")
                                        
                                        if row.get('publication_year'):
                                            st.markdown(f"**Year:** {row['publication_year']}")
                                    
                                    with rcol2:
                                        if row.get('doi'):
                                            st.markdown(f"[🔗 Open original article](https://doi.org/{row['doi']})")
                                        
                                        if 'topics' in row and row['topics']:
                                            topics_str = ", ".join(row['topics'][:3])
                                            st.markdown(f"**Topics:** {topics_str}")
                                        elif 'topic' in row and row['topic'] != -1:
                                            topic_label = get_topic_label(row['topic'], topic_model, topic_labels)
                                            st.markdown(f"**Topic:** {topic_label}")
                                    
                                    # Abstract o sommario
                                    if 'summary' in row and row['summary']:
                                        st.markdown(f"**Summary:** {row['summary']}")
                                    else:
                                        st.markdown(f"**Abstract:** {row['abstract'][:300]}...")
                            
                            with col2:
                                # Aggiungi bottone per confrontare gli articoli
                                if st.button("Compare", key=f"compare_{idx}"):
                                    if idx in st.session_state.compared_articles:
                                        st.session_state.compared_articles.remove(idx)
                                    elif len(st.session_state.compared_articles) < 3:  # Limitare a 3 articoli
                                        st.session_state.compared_articles.append(idx)
                                    else:
                                        st.warning("⚠️ You can compare up to 3 articles at a time.")
                    
                    with rec_tabs[1]:
                        # Table view
                        table_data = []
                        for _, row in recommended_articles.iterrows():
                            author_str = ""
                            if isinstance(row.get('authors'), list):
                                author_str = ", ".join(row['authors'][:2])
                                if len(row['authors']) > 2:
                                    author_str += " et al."
                                
                            topic_str = ""
                            if 'topic' in row and row['topic'] != -1:
                                topic_str = get_topic_label(row['topic'], topic_model, topic_labels)
                            
                            table_data.append({
                                "Title": row['title'],
                                "Authors": author_str,
                                "Year": row.get('publication_year', ""),
                                "Topic": topic_str,
                                "Similarity": f"{row.get('similarity', 0)*100:.1f}%"
                            })
                            
                        st.dataframe(pd.DataFrame(table_data))
                    
                    # Se ci sono articoli da confrontare, mostra il pannello di confronto
                    if st.session_state.compared_articles:
                        st.markdown("### 📊 Article Comparison")
                        comparison_df = pd.DataFrame({
                            "Title": [recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['title'] for idx in st.session_state.compared_articles],
                            "Authors": [", ".join(recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['authors'][:2]) + " et al." 
                                    if isinstance(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('authors'), list) and len(recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['authors']) > 2 
                                    else ", ".join(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('authors', [])) for idx in st.session_state.compared_articles],
                            "Year": [recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('publication_year', "") for idx in st.session_state.compared_articles],
                            "Topic": [get_topic_label(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('topic', -1), topic_model, topic_labels) 
                                    for idx in st.session_state.compared_articles],
                            "Similarity": [f"{recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('similarity', 0)*100:.1f}%" for idx in st.session_state.compared_articles]
                        })
                        
                        st.dataframe(comparison_df)
                        
                        # Visualizzazione interattiva della similarità
                        similarity_data = []
                        for idx in st.session_state.compared_articles:
                            row = recommended_articles.iloc[recommended_articles.index.get_loc(idx)]
                            similarity_data.append({
                                "Article": row['title'][:30] + "..." if len(row['title']) > 30 else row['title'],
                                "Similarity": row.get('similarity', 0) * 100
                            })
                        
                        similarity_df = pd.DataFrame(similarity_data)
                        fig = px.bar(similarity_df, x='Article', y='Similarity', 
                                    title='Similarity to Query Article (%)',
                                    color='Similarity',
                                    color_continuous_scale='blues')
                        st.plotly_chart(fig, use_container_width=True)
                
                        # Aggiungi pulsante per ripulire la selezione
                        if st.button("Clear comparison"):
                            st.session_state.compared_articles = []
                            st.experimental_rerun()

            progress_bar.progress(90)
            
            # Analisi avanzata
            if show_advanced:
                st.markdown("---")
                st.markdown("## 📊 Advanced Topic Analysis")
                show_advanced_analysis(topic_model, article_text, topic_labels)
                
            # Completa l'operazione
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        
        else:
            progress_bar.progress(40)
            status_text.text("🔍 Article not found locally, searching online...")
            
            # L'articolo non è presente nel dataset, lo cerco su OpenAlex
            st.warning("⚠️ Article not found in local dataset. Searching on OpenAlex...")
            
            article_data = search_openalex(article_title)
            if article_data:
                # Aggiungi alla cronologia se non è già presente
                if article_title not in st.session_state.search_history:
                    st.session_state.search_history.append(article_title)
                    # Limita la cronologia a 10 elementi
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history.pop(0)
                        
                st.success("✅ Article found on OpenAlex!")
                
                abstract = article_data.get('abstract','')
                title = article_data.get('title', '')
                
                if not abstract:
                    st.warning("⚠️ Abstract not available for this article. Please upload the abstract for better analysis.")
                
                article_text = (abstract or '') + ' ' + title
                
                progress_bar.progress(50)
                
                # Visualizzazione delle informazioni sull'articolo
                with st.container():
                    st.markdown('<div class="article-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"### 📚 {title}")
                    
                    # Dettagli dell'articolo
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if isinstance(article_data.get('authors'), list):
                            authors_str = ", ".join(article_data['authors'][:3])
                            if len(article_data['authors']) > 3:
                                authors_str += f" and {len(article_data['authors']) - 3} more"
                            st.markdown(f"**Authors:** {authors_str}")
                        
                        if article_data.get('year'):
                            st.markdown(f"**Year:** {article_data['year']}")
                    
                    with col2:
                        if article_data.get('doi'):
                            st.markdown(f"[🔗 Open original article](https://doi.org/{article_data['doi']})")
                    
                    # Abstract o sommario
                    st.markdown("#### Abstract")
                    st.markdown(f"{abstract}")
                    
                    # Riassunto generato da AI se richiesto
                    if apply_summarization and summarizer is not None and abstract:
                        st.markdown("#### AI-Generated Summary")
                        with st.spinner("Generating summary..."):
                            try:
                                summary = summarizer(abstract, max_length=max_summary_length, min_length=50, do_sample=False)[0]['summary_text']
                                st.markdown(f"{summary}")
                            except Exception as e:
                                st.error(f"❌ Error generating summary: {str(e)}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                progress_bar.progress(70)
                status_text.text("📊 Analyzing topics...")
                
                # Analisi del topic
                if abstract:
                    st.markdown("### 🔍 Topic Analysis")
                    article_topic_id = visualize_article_topics(article_text, topic_model, topic_labels)
                    
                    progress_bar.progress(80)
                    status_text.text("🔍 Finding similar articles...")

                    # Ricerca articoli simili
                    st.markdown("### 📚 Similar Articles")
                    with st.spinner("Finding similar articles..."):
                        # Add this line to show the topic ID being used
                        st.write(f"Using topic ID {article_topic_id} for recommendations")
            
                        recommended_articles = get_article_recommendations(article_text, df, embedding_model, 
                                                                        article_embeddings, topic_id=article_topic_id, top_k=recommended_count)
            
                        # Check if we got recommendations
                        if recommended_articles is None or len(recommended_articles) == 0:
                            st.warning("⚠️ No similar articles found")
                        else:
                            st.success(f"✅ Found {len(recommended_articles)} similar articles")
                            
                            # Create tabs for different recommendation views
                            rec_tabs = st.tabs(["Card View", "Table View"])
                            
                            with rec_tabs[0]:
                                # Visualizzazione degli articoli raccomandati (Card View)
                                for idx, row in recommended_articles.iterrows():
                                    similarity_pct = int(row.get('similarity', 0) * 100)
                                    
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        with st.expander(f"📄 {row['title']} (Similarity: {similarity_pct}%)"):
                                            # Dettagli dell'articolo raccomandato
                                            rcol1, rcol2 = st.columns(2)
                                            
                                            with rcol1:
                                                if isinstance(row.get('authors'), list):
                                                    authors_str = ", ".join(row['authors'][:3])
                                                    if len(row['authors']) > 3:
                                                        authors_str += f" and {len(row['authors']) - 3} more"
                                                    st.markdown(f"**Authors:** {authors_str}")
                                                
                                                if row.get('publication_year'):
                                                    st.markdown(f"**Year:** {row['publication_year']}")
                                            
                                            with rcol2:
                                                if row.get('doi'):
                                                    st.markdown(f"[🔗 Open original article](https://doi.org/{row['doi']})")
                                                
                                                if 'topics' in row and row['topics']:
                                                    topics_str = ", ".join(row['topics'][:3])
                                                    st.markdown(f"**Topics:** {topics_str}")
                                                elif 'topic' in row and row['topic'] != -1:
                                                    topic_label = get_topic_label(row['topic'], topic_model, topic_labels)
                                                    st.markdown(f"**Topic:** {topic_label}")
                                            
                                            # Abstract o sommario
                                            if 'summary' in row and row['summary']:
                                                st.markdown(f"**Summary:** {row['summary']}")
                                            else:
                                                st.markdown(f"**Abstract:** {row['abstract'][:300]}...")
                                    
                                    with col2:
                                        # Aggiungi bottone per confrontare gli articoli
                                        if st.button("Compare", key=f"compare_{idx}"):
                                            if idx in st.session_state.compared_articles:
                                                st.session_state.compared_articles.remove(idx)
                                            elif len(st.session_state.compared_articles) < 3:  # Limitare a 3 articoli
                                                st.session_state.compared_articles.append(idx)
                                            else:
                                                st.warning("⚠️ You can compare up to 3 articles at a time.")
                            
                            with rec_tabs[1]:
                                # Table view
                                table_data = []
                                for _, row in recommended_articles.iterrows():
                                    author_str = ""
                                    if isinstance(row.get('authors'), list):
                                        author_str = ", ".join(row['authors'][:2])
                                        if len(row['authors']) > 2:
                                            author_str += " et al."
                                        
                                    topic_str = ""
                                    if 'topic' in row and row['topic'] != -1:
                                        topic_str = get_topic_label(row['topic'], topic_model, topic_labels)
                                    
                                    table_data.append({
                                        "Title": row['title'],
                                        "Authors": author_str,
                                        "Year": row.get('publication_year', ""),
                                        "Topic": topic_str,
                                        "Similarity": f"{row.get('similarity', 0)*100:.1f}%"
                                    })
                                    
                                st.dataframe(pd.DataFrame(table_data))
                            
                            # Se ci sono articoli da confrontare, mostra il pannello di confronto
                            if st.session_state.compared_articles:
                                st.markdown("### 📊 Article Comparison")
                                comparison_df = pd.DataFrame({
                                    "Title": [recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['title'] for idx in st.session_state.compared_articles],
                                    "Authors": [", ".join(recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['authors'][:2]) + " et al." 
                                            if isinstance(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('authors'), list) and len(recommended_articles.iloc[recommended_articles.index.get_loc(idx)]['authors']) > 2 
                                            else ", ".join(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('authors', [])) for idx in st.session_state.compared_articles],
                                    "Year": [recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('publication_year', "") for idx in st.session_state.compared_articles],
                                    "Topic": [get_topic_label(recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('topic', -1), topic_model, topic_labels) 
                                            for idx in st.session_state.compared_articles],
                                    "Similarity": [f"{recommended_articles.iloc[recommended_articles.index.get_loc(idx)].get('similarity', 0)*100:.1f}%" for idx in st.session_state.compared_articles]
                                })
                                
                                st.dataframe(comparison_df)
                                
                                # Visualizzazione interattiva della similarità
                                similarity_data = []
                                for idx in st.session_state.compared_articles:
                                    row = recommended_articles.iloc[recommended_articles.index.get_loc(idx)]
                                    similarity_data.append({
                                        "Article": row['title'][:30] + "..." if len(row['title']) > 30 else row['title'],
                                        "Similarity": row.get('similarity', 0) * 100
                                    })
                                
                                similarity_df = pd.DataFrame(similarity_data)
                                fig = px.bar(similarity_df, x='Article', y='Similarity', 
                                            title='Similarity to Query Article (%)',
                                            color='Similarity',
                                            color_continuous_scale='blues')
                                st.plotly_chart(fig, use_container_width=True)
                        
                                # Aggiungi pulsante per ripulire la selezione
                                if st.button("Clear comparison"):
                                    st.session_state.compared_articles = []
                                    st.experimental_rerun()
                            
                    # Analisi avanzata
                    if show_advanced:
                        st.markdown("---")
                        st.markdown("## 📊 Advanced Topic Analysis")
                        show_advanced_analysis(topic_model, article_text, topic_labels)
                else:
                    st.warning("⚠️ Cannot perform topic analysis without an abstract.")
                
                # Completa l'operazione
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
            else:
                progress_bar.progress(100)
                status_text.text("❌ Article not found.")
                st.error("❌ Article not found. Please try with another title.")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()

with tabs[1]:
    # ENHANCED TOPIC EXPLORER TAB with dataset analysis info
    st.markdown("## 📊 Topic Explorer & Dataset Analysis")
    
    if topic_model is None:
        st.warning("⚠️ Topic model not available. Please run the notebook to generate the model first.")
    else:
        # Add dataset info section
        st.markdown("### 📋 Dataset Overview")
        
        # Dataset metrics in cards
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Articles", len(df))
        with metrics_cols[1]:
            topic_info = topic_model.get_topic_info()
            outliers = topic_info.loc[topic_info['Topic'] == -1, 'Count'].values[0] if -1 in topic_info['Topic'].values else 0
            st.metric("Outlier Articles", outliers)
        with metrics_cols[2]:
            unique_topics = len(topic_info[topic_info['Topic'] != -1])
            st.metric("Unique Topics", unique_topics)
        with metrics_cols[3]:
            avg_articles_per_topic = int(topic_info[topic_info['Topic'] != -1]['Count'].mean())
            st.metric("Avg Articles/Topic", avg_articles_per_topic)
            
        # Add model performance metrics dashboard
        st.markdown("### 📊 Model Performance Metrics")
        
        perf_metrics_cols = st.columns(3)
        with perf_metrics_cols[0]:
            # Estrai il valore e verifica che sia un numero
            coherence = model_metrics.get('topic_coherence', 0)
            # Formatta il valore solo se è un numero
            coherence_formatted = f"{coherence:.3f}" if isinstance(coherence, (int, float)) else str(coherence)
            
            st.markdown(
                f"""
                <div class="performance-metric">
                    <div class="performance-value">{coherence_formatted}</div>
                    <div class="performance-label">Topic Coherence</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with perf_metrics_cols[1]:
            # Estrai il valore e verifica che sia un numero
            diversity = model_metrics.get('topic_diversity', 0)
            # Formatta il valore solo se è un numero
            diversity_formatted = f"{diversity:.3f}" if isinstance(diversity, (int, float)) else str(diversity)
            
            st.markdown(
                f"""
                <div class="performance-metric">
                    <div class="performance-value">{diversity_formatted}</div>
                    <div class="performance-label">Topic Diversity</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with perf_metrics_cols[2]:
            # Estrai il valore e verifica che sia un numero
            precision = model_metrics.get('precision_at_k', 0)
            # Formatta il valore solo se è un numero
            precision_formatted = f"{precision:.3f}" if isinstance(precision, (int, float)) else str(precision)
            
            st.markdown(
                f"""
                <div class="performance-metric">
                    <div class="performance-value">{precision_formatted}</div>
                    <div class="performance-label">Recommendation Precision@5</div>
                </div>
                """, 
                unsafe_allow_html=True )
            
        # Add explanatory section for metrics
        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown("""
            - **Topic Coherence**: Measures how semantically coherent the topics are. Higher values indicate more coherent topics.
            - **Topic Diversity**: Measures how distinct the topics are from each other. Higher values indicate more diverse topics.
            - **Recommendation Precision@5**: Measures how accurate the recommendation system is at finding similar articles. Higher values indicate better recommendations.
            """)
        
        # Ottenimento delle informazioni sui topic
        topic_info = topic_model.get_topic_info()
        
        # Rimozione del topic -1 (outlier)
        filtered_topic_info = topic_info[topic_info['Topic'] != -1]
        
        # Creazione di un dizionario per la visualizzazione con etichette
        topic_options = {
            topic: get_topic_label(topic, topic_model, topic_labels)
            for topic in filtered_topic_info['Topic'].tolist()
        }
        
        # Show topic distribution first
        st.markdown("### 📊 Topic Distribution")
        
        # Visualizzazione interattiva del barchart dei topic
        try:
            # Modifica per includere le etichette leggibili
            topic_info_display = topic_info[topic_info['Topic'] != -1].head(15).copy()
            topic_info_display['Topic Label'] = topic_info_display['Topic'].apply(
                lambda x: get_topic_label(x, topic_model, topic_labels)
            )
            
            # Visualizzazione personalizzata
            fig = px.bar(
                topic_info_display, 
                x='Topic', 
                y='Count',
                hover_data=['Topic Label'],
                color='Count',
                color_continuous_scale='blues',
                title='Top 15 Topics by Document Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating topic barchart: {str(e)}")
        
        # Show topic similarity heatmap
        st.markdown("### 🗺️ Topic Similarity Map")
        
                    # Visualizzazione della heatmap di similarità
        try:
            fig_heatmap = topic_model.visualize_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating topic heatmap: {str(e)}")
        
        # Selezione del topic da esplorare con etichette leggibili
        st.markdown("### 🔍 Explore a Specific Topic")
        
        selected_topic = st.selectbox(
            "Select a topic to explore:",
            options=list(topic_options.keys()),
            format_func=lambda x: topic_options[x]
        )
        
        if selected_topic:
            # Visualizzazione dei dettagli del topic
            st.markdown(f"### {topic_options[selected_topic]}")
            
            # Dimensione del topic
            topic_size = topic_info.loc[topic_info['Topic'] == selected_topic, 'Count'].values[0]
            
            # Visualizzazione delle metriche
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents in Topic", topic_size)
            
            with col2:
                total_docs = topic_info['Count'].sum() - topic_info.loc[topic_info['Topic'] == -1, 'Count'].values[0]
                percentage = (topic_size / total_docs * 100).round(2)
                st.metric("% of Corpus", f"{percentage}%")
            
            with col3:
                # Topic simili
                try:
                    similar_topics = topic_model.find_topics(selected_topic, top_n=3)
                    similar_topics_labels = [f"{t}" for t in similar_topics[0]]
                    st.metric("Similar Topics", ", ".join(similar_topics_labels))
                except Exception as e:
                    st.metric("Similar Topics", "N/A")
            
            # Visualizzazione dei termini più rilevanti
            topic_terms = topic_model.get_topic(selected_topic)
            topic_terms_df = pd.DataFrame(topic_terms, columns=["Term", "Weight"]).head(20)
            
            # Visualizzazione dei termini
            fig = px.bar(topic_terms_df, 
                        x="Weight", 
                        y="Term", 
                        orientation="h", 
                        title=f"Top 20 Terms in Topic {selected_topic}",
                        color="Weight",
                        color_continuous_scale="blues",
                        labels={"Weight": "Relevance Score", "Term": ""},
                        height=500)
            
            fig.update_layout(xaxis_title="Relevance Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualizzazione in forma di word cloud
            st.markdown("### 🔤 Keyword Distribution")
            
            # Usa la funzione cached per il wordcloud
            fig = create_wordcloud(topic_terms)
            st.pyplot(fig)
            
            # Visualizzazione di documenti rappresentativi per questo topic
            st.markdown("### 📄 Representative Documents")
            
            # Verifica se esistono topic preassegnati
            if 'topic' in df.columns:
                # Usa i topic già assegnati
                topic_docs = df[df['topic'] == selected_topic]
            else:
                # Se necessario, filtra i documenti del dataset in base ai loro topic assegnati
                topic_docs = pd.DataFrame()
                
                # Se abbiamo topic_assignments e non abbiamo assegnato i topic al dataframe
                if topic_assignments is not None:
                    temp_df = df.copy()
                    temp_df['topic'] = topic_assignments
                    topic_docs = temp_df[temp_df['topic'] == selected_topic]
            
            if not topic_docs.empty:
                # Ordina per rilevanza (se disponibile)
                if 'relevance' in topic_docs.columns:
                    sorted_docs = topic_docs.sort_values('relevance', ascending=False)
                else:
                    sorted_docs = topic_docs
                
                # Mostra i primi 5 documenti rappresentativi
                for i, (_, doc) in enumerate(sorted_docs.head(5).iterrows(), 1):
                    with st.expander(f"Document {i}: {doc['title']}"):
                        # Mostra autori se disponibili
                        if isinstance(doc.get('authors'), list) and doc['authors']:
                            authors_str = ", ".join(doc['authors'][:3])
                            if len(doc['authors']) > 3:
                                authors_str += f" and {len(doc['authors']) - 3} more"
                            st.markdown(f"**Authors:** {authors_str}")
                        
                        # Mostra anno se disponibile
                        if 'publication_year' in doc and doc['publication_year']:
                            st.markdown(f"**Year:** {doc['publication_year']}")
                        
                        # Mostra abstract o riassunto
                        if 'summary' in doc and doc['summary']:
                            st.markdown(f"**Summary:** {doc['summary']}")
                        else:
                            st.markdown(f"**Abstract:** {doc['abstract'][:300]}...")
                        
                        # Link all'articolo originale
                        if 'doi' in doc and doc['doi']:
                            st.markdown(f"[🔗 Open original article](https://doi.org/{doc['doi']})")
            else:
                st.info("No documents found for this topic. Try analyzing articles first.")

with tabs[2]:
    # IMPLEMENTATION OF THE BROWSE DATASET TAB
    st.markdown("## 📚 Browse Dataset")
    
    # Add filtering options
    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(df['publication_year'].dropna().unique()) if 'publication_year' in df.columns else []
        year_filter = st.multiselect("Filter by Year", 
                                    options=years,
                                    default=[])
    
    with col2:
        if 'topic' in df.columns:
            topic_options = {
                topic_id: get_topic_label(topic_id, topic_model, topic_labels) 
                for topic_id in df['topic'].unique() if topic_id != -1
            }
            topic_filter = st.multiselect("Filter by Topic", 
                                         options=list(topic_options.keys()),
                                         format_func=lambda x: topic_options.get(x, f"Topic {x}"),
                                         default=[])
    
    with col3:
        search_term = st.text_input("Search in Titles/Abstracts", "")
    
    # Apply filters
    filtered_df = df.copy()
    if year_filter:
        filtered_df = filtered_df[filtered_df['publication_year'].isin(year_filter)]
    
    if 'topic' in df.columns and topic_filter:
        filtered_df = filtered_df[filtered_df['topic'].isin(topic_filter)]
    
    if search_term:
        search_mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) | 
            filtered_df['abstract'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    # Display filtered dataset
    st.write(f"Showing {len(filtered_df)} of {len(df)} articles")
    
    # Create a paginated display
    PAGE_SIZE = 10
    max_pages = max(1, (len(filtered_df) // PAGE_SIZE) + (1 if len(filtered_df) % PAGE_SIZE > 0 else 0))
    page_number = st.number_input("Page", min_value=1, 
                                 max_value=max_pages,
                                 value=1)
    
    # Pulsanti di navigazione
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        prev_button = st.button("◀️ Previous", disabled=page_number <= 1)
    with col3:
        next_button = st.button("Next ▶️", disabled=page_number >= max_pages)
    
    if prev_button and page_number > 1:
        page_number -= 1
    if next_button and page_number < max_pages:
        page_number += 1
    
    start_idx = (page_number - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(filtered_df))
    
    if not filtered_df.empty:
        # Mostra contatore di articoli
        st.write(f"Articles {start_idx + 1} to {end_idx} of {len(filtered_df)}")
        
        for i, (_, article) in enumerate(filtered_df.iloc[start_idx:end_idx].iterrows()):
            with st.expander(f"{i+1+start_idx}. {article['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Abstract**: {article['abstract'][:300]}..." if len(article['abstract']) > 300 else f"**Abstract**: {article['abstract']}")
                    
                    if 'summary' in article and article['summary']:
                        st.markdown(f"**Summary**: {article['summary']}")
                
                with col2:
                    if isinstance(article.get('authors'), list):
                        authors_str = ", ".join(article['authors'][:3])
                        if len(article['authors']) > 3:
                            authors_str += f" and {len(article['authors']) - 3} more"
                        st.markdown(f"**Authors**: {authors_str}")
                    
                    if article.get('publication_year'):
                        st.markdown(f"**Year**: {article['publication_year']}")
                    
                    if 'topic' in article and article['topic'] != -1:
                        topic_label = get_topic_label(article['topic'], topic_model, topic_labels)
                        st.markdown(f"**Topic**: {topic_label}")
                    
                    if article.get('doi'):
                        st.markdown(f"[🔗 Open original article](https://doi.org/{article['doi']})")
    else:
        st.warning("No articles match your search criteria. Try adjusting the filters.")
    
    # Add options to export filtered results
    if not filtered_df.empty:
        st.markdown("### 📊 Export Options")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export as CSV", key="export_csv"):
                # Prepare data for export
                export_df = filtered_df[['title', 'abstract', 'publication_year', 'doi']].copy()
                if 'topic' in filtered_df.columns:
                    export_df['topic'] = filtered_df['topic'].apply(
                        lambda x: get_topic_label(x, topic_model, topic_labels) if x != -1 else "Outlier"
                    )
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                
                # Create a download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="filtered_articles.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("Generate Summary Stats", key="summary_stats"):
                st.markdown("### 📊 Summary Statistics")
                
                # Topic distribution if available
                if 'topic' in filtered_df.columns:
                    topic_counts = filtered_df['topic'].value_counts()
                    topic_counts_df = pd.DataFrame({
                        'Topic': [get_topic_label(t, topic_model, topic_labels) for t in topic_counts.index],
                        'Count': topic_counts.values
                    })
                    
                    fig = px.pie(topic_counts_df, values='Count', names='Topic', 
                               title='Topic Distribution in Filtered Dataset')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Year distribution if available
                if 'publication_year' in filtered_df.columns and not filtered_df['publication_year'].isna().all():
                    year_counts = filtered_df['publication_year'].value_counts().sort_index()
                    fig = px.bar(x=year_counts.index, y=year_counts.values,
                               labels={'x': 'Year', 'y': 'Number of Articles'},
                               title='Articles by Publication Year')
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Statistiche avanzate
                st.markdown("### 📈 Advanced Statistics")
                
                # Crea una tabella con statistiche per ogni anno
                if 'publication_year' in filtered_df.columns and 'topic' in filtered_df.columns:
                    yearly_stats = []
                    for year in sorted(filtered_df['publication_year'].unique()):
                        year_data = filtered_df[filtered_df['publication_year'] == year]
                        top_topic = year_data['topic'].value_counts().index[0] if not year_data['topic'].empty else -1
                        top_topic_label = get_topic_label(top_topic, topic_model, topic_labels)
                        
                        yearly_stats.append({
                            'Year': year,
                            'Articles': len(year_data),
                            'Dominant Topic': top_topic_label,
                            'Unique Topics': year_data['topic'].nunique()
                        })
                    
                    yearly_stats_df = pd.DataFrame(yearly_stats)
                    st.dataframe(yearly_stats_df)
                    
                    # Crea un grafico che mostra l'evoluzione dei topic nel tempo
                    topic_evolution = []
                    for year in sorted(filtered_df['publication_year'].unique()):
                        year_data = filtered_df[filtered_df['publication_year'] == year]
                        for topic_id, count in year_data['topic'].value_counts().items():
                            if topic_id != -1:  # Escludi gli outlier
                                topic_evolution.append({
                                    'Year': year,
                                    'Topic': get_topic_label(topic_id, topic_model, topic_labels),
                                    'Count': count
                                })
                    
                    topic_evolution_df = pd.DataFrame(topic_evolution)
                    if not topic_evolution_df.empty:
                        fig = px.line(topic_evolution_df, x='Year', y='Count', color='Topic',
                                    title='Topic Evolution Over Time')
                        st.plotly_chart(fig, use_container_width=True)

# Aggiungi un footer con informazioni sul progetto
st.markdown("""
<div class="footer">
    <p>Academic Research Analyzer - Deloitte x Luiss: Data Science in Action Project</p>
    <p>Created by: Simone Moroni, Gabriele Gogli, Matteo Piccirilli - 2025</p>
</div>
""", unsafe_allow_html=True)