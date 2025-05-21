import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, unquote

# Dictionnaire des URLs classées par langue
urls = {
    "fr": [
        "https://fr.wikipedia.org/wiki/Charles_Perrault#Les_Contes",
        "https://fr.wikipedia.org/wiki/Peau_d%27Âne",
        "https://fr.wikipedia.org/wiki/La_Belle_au_bois_dormant",
        "https://fr.wikipedia.org/wiki/Le_Petit_Chaperon_rouge",
        "https://fr.wikipedia.org/wiki/La_Barbe_bleue",
        "https://fr.wikipedia.org/wiki/Le_Maître_chat_ou_le_Chat_botté"
    ],
    "en": [
        "https://en.wikipedia.org/wiki/Charles_Perrault",
        "https://en.wikipedia.org/wiki/Donkeyskin",
        "https://en.wikipedia.org/wiki/Sleeping_Beauty",
        "https://en.wikipedia.org/wiki/Little_Red_Riding_Hood",
        "https://en.wikipedia.org/wiki/Bluebeard",
        "https://en.wikipedia.org/wiki/Puss_in_Boots"
    ],
    "es": [
        "https://es.wikipedia.org/wiki/Charles_Perrault",
        "https://es.wikipedia.org/wiki/Piel_de_asno",
        "https://es.wikipedia.org/wiki/La_Bella_Durmiente",
        "https://es.wikipedia.org/wiki/Caperucita_Roja",
        "https://es.wikipedia.org/wiki/Barba_Azul",
        "https://es.wikipedia.org/wiki/El_Gato_con_Botas"
    ]
}

# Fonction de scraping
def scrape_wikipedia_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content_div = soup.find('div', {'id': 'mw-content-text'})
        paragraphs = content_div.find_all('p', recursive=True)

        text = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return text
    except Exception as e:
        return f"[Erreur] {url} : {str(e)}"

# Fonction principale pour tout enregistrer
def main():
    for lang, url_list in urls.items():
        lang_dir = os.path.join("data", lang)
        os.makedirs(lang_dir, exist_ok=True)

        for url in url_list:
            print(f"Scraping {url}")
            text = scrape_wikipedia_article(url)

            path = urlparse(url).path
            page_name = os.path.basename(path).split('#')[0]
            page_name = unquote(page_name).replace(' ', '_')

            file_path = os.path.join(lang_dir, f"{page_name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    print("\n Tous les articles ont été scrappés et enregistrés dans le dossier 'data/'.")

if __name__ == "__main__":
    main()
