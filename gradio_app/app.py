from deep_translator import GoogleTranslator
from transformers import pipeline
import gradio as gr
from pytube import YouTube

pipe = pipeline(model="tlord/whisper")

LANGUAGES = {
    'afrikaans' : 'af',
    'albanian' : 'sq',
    'amharic' : 'am',
    'arabic' : 'ar',
    'armenian' : 'hy',
    'azerbaijani' : 'az',
    'basque' : 'eu',
    'belarusian' : 'be',
    'bengali' : 'bn',
    'bosnian' : 'bs',
    'bulgarian' : 'bg',
    'catalan' : 'ca',
    'cebuano' : 'ceb',
    'chichewa' : 'ny',
    'chinese (simplified)' : 'zh-c',
    'chinese (traditional)' : 'zh-t',
    'corsican' : 'co',
    'croatian' : 'hr',
    'czech' : 'cs',
    'danish' : 'da',
    'dutch' : 'nl',
    'english' : 'en',
    'esperanto' : 'eo',
    'estonian' : 'et',
    'filipino' : 'tl',
    'finnish' : 'fi',
    'french' : 'fr',
    'frisian' : 'fy',
    'galician' : 'gl',
    'georgian' : 'ka',
    'german' : 'de',
    'greek' : 'el',
    'gujarati' : 'gu',
    'haitian creole' : 'ht',
    'hausa' : 'ha',
    'hawaiian' : 'haw',
    'hebrew' : 'iw',
    'hebrew' : 'he',
    'hindi' : 'hi',
    'hmong' : 'hmn',
    'hungarian' : 'hu',
    'icelandic' : 'is',
    'igbo' : 'ig',
    'indonesian' : 'id',
    'irish' : 'ga',
    'italian' : 'it',
    'japanese' : 'ja',
    'javanese' : 'jw',
    'kannada' : 'kn',
    'kazakh' : 'kk',
    'khmer' : 'km',
    'korean' : 'ko',
    'kurdish (kurmanji)' : 'ku',
    'kyrgyz' : 'ky',
    'lao' : 'lo',
    'latin' : 'la',
    'latvian' : 'lv',
    'lithuanian' : 'lt',
    'luxembourgish' : 'lb',
    'macedonian' : 'mk',
    'malagasy' : 'mg',
    'malay' : 'ms',
    'malayalam' : 'ml',
    'maltese' : 'mt',
    'maori' : 'mi',
    'marathi' : 'mr',
    'mongolian' : 'mn',
    'myanmar (burmese)' : 'my',
    'nepali' : 'ne',
    'norwegian' : 'no',
    'odia' : 'or',
    'pashto' : 'ps',
    'persian' : 'fa',
    'polish' : 'pl',
    'portuguese' : 'pt',
    'punjabi' : 'pa',
    'romanian' : 'ro',
    'russian' : 'ru',
    'samoan' : 'sm',
    'scots gaelic' : 'gd',
    'serbian' : 'sr',
    'sesotho' : 'st',
    'shona' : 'sn',
    'sindhi' : 'sd',
    'sinhala' : 'si',
    'slovak' : 'sk',
    'slovenian' : 'sl',
    'somali' : 'so',
    'spanish' : 'es',
    'sundanese' : 'su',
    'swahili' : 'sw',
    'tajik' : 'tg',
    'tamil' : 'ta',
    'telugu' : 'te',
    'thai' : 'th',
    'turkish' : 'tr',
    'ukrainian' : 'uk',
    'urdu' : 'ur',
    'uyghur' : 'ug',
    'uzbek' : 'uz',
    'vietnamese' : 'vi',
    'welsh' : 'cy',
    'xhosa' : 'xh',
    'yiddish' : 'yi',
    'yoruba' : 'yo',
    'zulu' : 'zu',
}

def get_soundfile(link):
    yt = YouTube(link)
    audio = yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")

    return audio

def translate(message, lang):
    res = GoogleTranslator(source='sv', target=lang).translate(message)
    if res != None and res != "":
        return res
    else:
        return "Error, sorry!"

def transcribe(audio, lang, history, link):
    if link != "":
        audio = get_soundfile(link)
    if lang is None or lang == "":
        lang = 'english'
    history = history or []
    lang_code = LANGUAGES[lang]
    text = pipe(audio)["text"]
    history.append((text, translate(text, lang_code)))

    return history, history

with gr.Blocks() as demo:
    history = gr.State([])
    with gr.Row():
        gr.Markdown(
        """
        # ID2223 Lab 2 Demo

        We use the pre-trained Whisper-Small model, finetuned for Swedish.

        The demo shows how the transcriber can be used. We showcase two variants:

        - Recorded. Transcribes the recorded audio.
        - YouTube. Paste a link and get transcription of a YouTube video.

        The chat will answer with the transcribed text translated into your chosen language.

        """)
    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(label="Language", choices=list(LANGUAGES.keys()), value="english")
            recorded = gr.Audio(label="Recorded transcription", source="microphone", type="filepath")
            link = gr.Textbox(label="Put YouTube link here", value="")
            submit = gr.Button(value="Transcribe & Translate")
        with gr.Column():
            chatbot = gr.Chatbot().style(color_map=("green", "gray"))
    submit.click(transcribe, inputs=[recorded, language, history, link], outputs=[chatbot, history])
demo.launch()