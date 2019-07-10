# bert-as-service

The service uses BERT model as a sentence encoding service, i.e. mapping a variable-length sentence to a fixed-length vector.
The goal is to represent a variable length sentence into a fixed length vector, e.g. hello world to [0.1, 0.3, 0.9]. 
Each element of the vector should "encode" some semantics of the original sentence.

Service uses BERT - an NLP model developed by Google for pre-training language representations. It leverages an enormous amount of plain text data publicly available on the web and is trained in an unsupervised manner. Pre-training a BERT model is a an expensive, one-time procedure for each language. Google released several pre-trained models where you can download from https://github.com/google-research/bert#pre-trained-models.

## Installation

The actial server and client can be installed via pip. They can be installed separately or even on different machines:

pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`

* using ports 7001 and 7002 for port-in and port-out communications


## Dependencies
* Python >= 3.5
* Tensorflow >= 1.10

## Starting server

1. Download a Pre-trained BERT Model

Download a model, then uncompress the zip file into some folder, say /tmp/english_L-12_H-768_A-12/

2. Start the BERT service

After installing the server, you should be able to use bert-serving-start CLI as follows:

bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/

## More Info

More info can be found at https://github.com/hanxiao/bert-as-service

## License

Software is licensed under MIT license. Copy of the license is included

## Supported languages

Languages:

    Afrikaans
    Albanian
    Arabic
    Aragonese
    Armenian
    Asturian
    Azerbaijani
    Bashkir
    Basque
    Bavarian
    Belarusian
    Bengali
    Bishnupriya Manipuri
    Bosnian
    Breton
    Bulgarian
    Burmese
    Catalan
    Cebuano
    Chechen
    Chinese (Simplified)
    Chinese (Traditional)
    Chuvash
    Croatian
    Czech
    Danish
    Dutch
    English
    Estonian
    Finnish
    French
    Galician
    Georgian
    German
    Greek
    Gujarati
    Haitian
    Hebrew
    Hindi
    Hungarian
    Icelandic
    Ido
    Indonesian
    Irish
    Italian
    Japanese
    Javanese
    Kannada
    Kazakh
    Kirghiz
    Korean
    Latin
    Latvian
    Lithuanian
    Lombard
    Low Saxon
    Luxembourgish
    Macedonian
    Malagasy
    Malay
    Malayalam
    Marathi
    Minangkabau
    Nepali
    Newar
    Norwegian (Bokmal)
    Norwegian (Nynorsk)
    Occitan
    Persian (Farsi)
    Piedmontese
    Polish
    Portuguese
    Punjabi
    Romanian
    Russian
    Scots
    Serbian
    Serbo-Croatian
    Sicilian
    Slovak
    Slovenian
    South Azerbaijani
    Spanish
    Sundanese
    Swahili
    Swedish
    Tagalog
    Tajik
    Tamil
    Tatar
    Telugu
    Turkish
    Ukrainian
    Urdu
    Uzbek
    Vietnamese
    Volapük
    Waray-Waray
    Welsh
    West Frisian
    Western Punjabi
    Yoruba


## References

[1] Xiao, Han,
    [*bert-as-service*](https://github.com/hanxiao/bert-as-service),
    2018
