import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64
from PIL import Image
import seaborn as sns
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from io import StringIO
import nltk


# Descargar los recursos necesarios
nltk.download('vader_lexicon_es')
nltk.download('punkt')


# Cargar las stopwords del archivo "stopwords-es.txt"
with open('stopwords-es.txt', 'r', encoding='utf-8') as file:
    stopwords_es = file.read().splitlines()
    

# Cargar el lexicon AFINN en español
afinn_lexicon = pd.read_csv("lexico_afinn_en_es.csv", encoding="ISO-8859-1")




    # Mostrar la interfaz de usuario
st.sidebar.markdown("#### Con esta herramienta podes analizar y comparardiferentes textos")


# Variables para el PDF
pdf_buffer = BytesIO()
pdf_filename = "analisis_texto.pdf"

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #f8f9f5;
    color:#463f3c;
    font-size:9px%,
   
    
}
div.stButton > button:hover {
   opacity: 0.6;
    }
</style>""", unsafe_allow_html=True)

# Lógica de generación de texto y análisis de sentimientos

def main():

    
    
    
    image = Image.open('logoBA.png')

     

    # Crear la interfaz de usuario
    st.image(image, width=150)
    st.title("Herramienta de Análisis de Texto")
      
  
   # Función para cargar un archivo TXT desde el disco local
   # Dividir el espacio en columnas con porcentajes específicos
    col13, col14 = st.sidebar.columns([8, 2])
    def load_text_from_file():
        uploaded_file = col13.file_uploader("Cargar texto para analizar", type=["txt"])
        if uploaded_file is not None:
            return uploaded_file.read().decode()
        return None
    input_text = load_text_from_file()

    # Botón "Ver texto 1"
    descargar_texto = col14.button("Ver")
    
    st.sidebar.write("---")
    st.sidebar.markdown("#### Análisis de Sentimientos")

    frec_pal = st.sidebar.number_input("Frecuencia mínima de palabras a plotear", min_value=1, max_value=30, value=1, step=1)
    st.sidebar.write("   ")
    
    # Inicializar la lista de stopwords si aún no existe en st.session_state
    if 'new_stopwords' not in st.session_state:
        st.session_state.new_stopwords = []

    # Botón para borrar las stopwords
    col5,col7 = st.sidebar.columns([7,3])
    if col7.button("Borrar"):
        st.session_state.new_stopwords = []
    # Añadir un campo de entrada para que los usuarios ingresen palabras ad hoc
    new_stopword = col5.text_input("Ingresa stopwords adicionales:", "")

    # Verificar si la palabra ya está en la lista y agregarla si es necesario
    if new_stopword.strip() and new_stopword.lower() not in st.session_state.new_stopwords:
        st.session_state.new_stopwords.append(new_stopword.lower())
        
        # Agregar las palabras adicionales a la lista de stopwords_es si son válidas
    for word in st.session_state.new_stopwords:
        if word and word.lower() not in stopwords_es:
                stopwords_es.append(word.lower())
            

    

    
    # Mostrar la lista de stopwords actualizada
    if  new_stopword :
        st.write("Lista de Stopwords Adicionales:")
        st.write(st.session_state.new_stopwords)
   
    # Dividir el espacio en dos columnas
    col3, col4 = st.sidebar.columns([5,5])

                # Mostrar los  en las columnas respectivas
   
    analyze_sentiment_button = col3.button("Analizar ")

    boton_resumir = col4.button('Resumir')
        
    st.sidebar.write("  ")

    word_to_search = st.sidebar.text_input("Ingresa una palabra para buscar en el texto")
    
    st.sidebar.write("---")
    col_11, col_12 = st.sidebar.columns([8,2])
    def load_text_from_file_2():
        uploaded_file = col_11.file_uploader("Cargar el segundo texto para comparar:", type=["txt"])
        if uploaded_file is not None:
            return uploaded_file.read().decode()
        return None
    
    
    input_text_2 = load_text_from_file_2()
    
      # Botónes 
    descargar_texto_2 =col_12.button("Ver ")
    boton_comparar = st.sidebar.button("Comparar")
    
    st.sidebar.write("---")
    # Agregar la palabra adicional a la lista de stopwords si se ingresó una palabra válida
    if new_stopword and new_stopword not in stopwords_es:
            stopwords_es.append(new_stopword.lower())

    def join_afinn_scores(filtered_text, afinn):
        # Convertir filtered_text en un DataFrame con una sola columna llamada "Palabra"
        filtered_text_df = pd.DataFrame({'Palabra': filtered_text})

        # Fusionar el DataFrame de filtered_text con afinn_lexicon
        reviews_afinn = pd.merge(filtered_text_df, afinn, on='Palabra', how='inner')

        # Calcular las ocurrencias y sumar las puntuaciones por palabra
        reviews_afinn = reviews_afinn.groupby('Palabra').agg(
            occurences=('Palabra', 'size'),
            contribution=('Puntuacion', 'sum')
        ).reset_index()

        return reviews_afinn

    def separate_positive_negative(top_words):
        positive_words = top_words[top_words['contribution'] > 0]
        negative_words = top_words[top_words['contribution'] < 0]
        return positive_words, negative_words

    def plot_positivas(positive_words):
        fig, ax = plt.subplots()
        sns.barplot(data=positive_words, y='Palabra', x='contribution', color='green', ax=ax)
        ax.set_title("Plot de palabras positivas")
        ax.set_xlabel("Contribución")
        ax.set_ylabel("Palabra")
        return fig

    def plot_negativas(negative_words):
        fig, ax = plt.subplots()
        sns.barplot(data=negative_words, y='Palabra', x='contribution', color='red', ax=ax)
        ax.set_title("Plot de palabras negativas")
        ax.set_xlabel("Contribución")
        ax.set_ylabel("Palabra")
        return fig
    
    def download_text( df_common_tokens, df_top_tokens_1, df_top_tokens_2):
    
        # Crear un DataFrame para los resultados de la comparación
        df_common_tokens['Palabra'] = df_common_tokens.index
        df_common_tokens.reset_index(drop=True, inplace=True)

        # Guardar los resultados de la comparación en un archivo Excel
        with pd.ExcelWriter("resultados_comparacion.xlsx") as writer:
            df_common_tokens.to_excel(writer, sheet_name="Palabras comunes", index=False)
            df_top_tokens_1.to_excel(writer, sheet_name="Top Palabras Texto 1", index=True)
            df_top_tokens_2.to_excel(writer, sheet_name="Top Palabras Texto 2", index=True)

        # Codificar el archivo Excel a base64 para descargarlo
        with open("resultados_comparacion.xlsx", "rb") as file_excel:
            b64_excel = base64.b64encode(file_excel.read()).decode()
            href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="resultados_comparacion.xlsx">Descargar Resultados (Excel)</a>'
            st.markdown(href_excel, unsafe_allow_html=True)

                            
                        
    def comparer(input_text, input_text_2):
        # Procesar el primer texto de la misma manera que lo estás haciendo actualmente
        punctuation = set(string.punctuation)
        word_tokens = word_tokenize(input_text.lower())
        filtered_words = [word for word in word_tokens if word not in stopwords_es and word not in punctuation]
        filtered_text = ' '.join(filtered_words)

        # Contar la frecuencia de las palabras en el primer texto
        word_freq = Counter(filtered_words)
        word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['Frecuencia'])

        if input_text_2:
            # Procesar el segundo texto de la misma manera que lo estás haciendo actualmente
            punctuation = set(string.punctuation)
            word_tokens_2 = word_tokenize(input_text_2.lower())
            filtered_words_2 = [word for word in word_tokens_2 if word not in stopwords_es and word not in punctuation]
            filtered_text_2 = ' '.join(filtered_words_2)

            # Contar la frecuencia de las palabras en el segundo texto
            word_freq_2 = Counter(filtered_words_2)
            word_freq_df_2 = pd.DataFrame.from_dict(word_freq_2, orient='index', columns=['Frecuencia'])

            # Identificar los tokens que aparecen en ambos textos
            common_tokens = set(filtered_words).intersection(filtered_words_2)

            # Contar la frecuencia de los tokens comunes en cada texto
            common_tokens_freq = {token: [word_freq[token], word_freq_2[token]] for token in common_tokens}

            # Mostrar la lista de tokens comunes y su frecuencia en cada texto
            st.write("Palabras comunes y su frecuencia en cada texto:")
            st.write(pd.DataFrame(common_tokens_freq, index=['Frecuencia Texto 1', 'Frecuencia Texto 2']).T)

            # Mostrar los tokens únicos más frecuentes en cada texto
            st.write("Palabras únicas más frecuentes en cada texto:")
            st.write("Texto 1:")
            st.write(word_freq_df.nlargest(10, 'Frecuencia'))
            st.write("Texto 2:")
            st.write(word_freq_df_2.nlargest(10, 'Frecuencia'))
            
                    # Crear DataFrames para los resultados
            df_common_tokens = pd.DataFrame(common_tokens_freq, index=['Frecuencia Texto 1', 'Frecuencia Texto 2']).T
            df_top_tokens_1 = word_freq_df.nlargest(10, 'Frecuencia')
            df_top_tokens_2 = word_freq_df_2.nlargest(10, 'Frecuencia')

            # Guardar los DataFrames en un archivo Excel
            with pd.ExcelWriter("resultados_comparacion.xlsx") as writer:
                df_common_tokens.to_excel(writer, sheet_name="Palabras comunes")
                df_top_tokens_1.to_excel(writer, sheet_name="Top Palabras Texto 1")
                df_top_tokens_2.to_excel(writer, sheet_name="Top Palabras Texto 2")

            

        return  df_common_tokens, df_top_tokens_1, df_top_tokens_2
        
    if descargar_texto:
        if input_text:
            st.write(input_text)
        else:
            st.write('Hubo un error')  
            
    if descargar_texto_2:
        if input_text_2:
            st.write(input_text_2)
        else:
            st.write('Hubo un error')         

    if boton_comparar:
        if input_text:
            df_common_tokens, df_top_tokens_1, df_top_tokens_2 = comparer(input_text, input_text_2)
            # Descargar el resumen de ambos textos y los resultados de la comparación
            download_text(df_common_tokens, df_top_tokens_1, df_top_tokens_2)


            st.success("Resultados de la comparación guardados en 'resultados_comparacion.xlsx'.")

        else:
     
            st.write("Por favor, ingresa texto antes de comparar.")

    def resumir_text(input_text):
                with st.spinner('Processing...'):
                    time.sleep(2)
                    if len(input_text) < 1000 or len(input_text) > 50000:
                        st.error('Please upload a file between 1,000 and 10,000 characters')
                    else:
                        # Utilizar un tokenizador para español
                        tokenizer = Tokenizer("spanish")
                        parser = PlaintextParser.from_string(input_text, tokenizer)

                        # Generar resumen utilizando LexRank
                        lex_rank_summarizer = LexRankSummarizer()
                        lexrank_summary = lex_rank_summarizer(parser.document, sentences_count=3)
                        lexrank_summa = ''
                        for sentence in lexrank_summary:
                            lexrank_summa = lexrank_summa + str(sentence)

                        # Calcular resumen utilizando Scoring Model con stopwords en español
                        text = input_text
                        words = nltk.word_tokenize(text, language='spanish')  # Tokenización en español
                        freqTable = dict()
                        for word in words:
                            word = word.lower()
                            if word in stopwords_es:
                                continue
                            if word in freqTable:
                                freqTable[word] += 1
                            else:
                                freqTable[word] = 1
                        sentences = nltk.sent_tokenize(text, language='spanish')  # Tokenización de oraciones en español
                        sentenceValue = dict()
                        for sentence in sentences:
                            for word, freq in freqTable.items():
                                if word in sentence.lower():
                                    if sentence in sentenceValue:
                                        sentenceValue[sentence] += freq
                                    else:
                                        sentenceValue[sentence] = freq
                        sumValues = 0
                        for sentence in sentenceValue:
                            sumValues += sentenceValue[sentence]
                        average = int(sumValues / len(sentenceValue))
                        summary = ''
                        for sentence in sentences:
                            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.3 * average)):
                                summary += " " + sentence
                        scoring_summa = summary

                        # Mostrar los resúmenes generados
                        st.markdown('___')
                        st.write('LexRank Model')
                        st.success(lexrank_summa)

                        st.markdown('___')
                        st.write('Scoring Model')
                        st.success(scoring_summa)

                        # Agregar opción para descargar resumen en un archivo de texto
                        if st.button('Descargar Resumen'):
                            with open('resumen.txt', 'w', encoding='utf-8') as f:
                                f.write("Resumen generado por LexRank Model:\n\n")
                                f.write(lexrank_summa)
                                f.write("\n\nResumen generado por Scoring Model:\n\n")
                                f.write(scoring_summa)

                        

    if boton_resumir:
             resumir_text(input_text)


    

    if analyze_sentiment_button:
        if input_text:
            text_value = input_text

            # Eliminar stopwords y puntuación
            punctuation = set(string.punctuation)
            word_tokens = word_tokenize(text_value.lower())
            filtered_words = [word for word in word_tokens if word not in stopwords_es and word not in punctuation]
            filtered_text = ' '.join(filtered_words)
            

            
            # Plot de frecuencia de palabras
            word_freq = Counter(filtered_words)
            word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['Frecuencia'])
            word_freq_df = word_freq_df.sort_values(by='Frecuencia', ascending=False)
            word_freq_df = word_freq_df[word_freq_df['Frecuencia'] > frec_pal]

            # Plot de nube de palabras
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(filtered_text)

            # Calcular las palabras positivas y negativas
            reviews_afinn = join_afinn_scores(filtered_words, afinn_lexicon)
            top_words = reviews_afinn.nlargest(50, 'contribution')
            positive_words, negative_words = separate_positive_negative(top_words)
            

            # Crear el PDF con el texto, el resumen y los gráficos
            with st.spinner('Generando PDF...'):
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.barh(word_freq_df.index, word_freq_df['Frecuencia'])
                ax1.set_yticklabels(word_freq_df.index)
                ax1.set_ylabel("Palabras")
                ax1.set_xlabel("Frecuencia")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis('off')
                st.pyplot(fig2)

              # Crear dos figuras usando tus funciones de trazado (plot_positivas y plot_negativas)
                fig_pos = plot_positivas(positive_words)
                fig_neg = plot_negativas(negative_words)

                # Dividir el espacio en dos columnas
                col1, col2 = st.columns(2)

                # Mostrar los gráficos en las columnas respectivas
                with col1:
                    st.pyplot(fig_pos)

                with col2:
                    st.pyplot(fig_neg)

                # Crear el PDF
                with PdfPages(pdf_buffer) as pdf:
                    # Página 1: Texto generado
                    pdf.savefig(fig1, bbox_inches='tight')

                    # Página 3: Gráfico de nube de palabras
                    pdf.savefig(fig2, bbox_inches='tight')

                    # Página 4: Gráfico de palabras positivas
                    pdf.savefig(fig_pos, bbox_inches='tight')

                    # Página 5: Gráfico de palabras negativas
                    pdf.savefig(fig_neg, bbox_inches='tight')

            st.success("PDF generado correctamente.")

            # Descargar el PDF
            def download_pdf():
                pdf_buffer.seek(0)
                b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">Descargar PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

            download_pdf()

        else:
            st.write("Por favor, ingresa texto antes de generar el análisis de sentimientos.")

    # Verificar si una palabra es positiva o negativa

    word = st.sidebar.text_input("Ingresa una palabra para verificar su sentimiento")
    if word:
        word_scores = pd.merge(pd.DataFrame([word], columns=['Palabra']), afinn_lexicon, on='Palabra', how='inner')
        if word_scores.empty:
            st.sidebar.write("La palabra no está en el lexicon AFINN en español.")
        else:
            sentiment_score = word_scores['Puntuacion'].iloc[0]
            if sentiment_score > 0:
                st.sidebar.write(f"La palabra '{word}' es positiva.")
            elif sentiment_score < 0:
                st.sidebar.write(f"La palabra '{word}' es negativa.")
            else:
                st.sidebar.write(f"La palabra '{word}' es neutral.")
    
    
    def buscar_palabra_en_oracion(texto, palabra):
        # Tokenizar el texto en oraciones
        oraciones = sent_tokenize(texto)

        # Buscar la palabra en cada oración
        oraciones_con_palabra = [oracion for oracion in oraciones if palabra.lower() in oracion.lower()]

        return oraciones_con_palabra
            
   
    
    if word_to_search:
        oraciones_con_palabra = buscar_palabra_en_oracion(input_text, word_to_search)
        if oraciones_con_palabra:
            st.write(f"La palabra '{word_to_search}' se encuentra en las siguientes oraciones:")
            for oracion in oraciones_con_palabra:
                st.write(oracion)
        else:
            st.write(f"La palabra '{word_to_search}' no se encuentra en ninguna oración.")

  

if __name__ == "__main__":
    main()
