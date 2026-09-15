import os

wiki_path = "data/wikipedia_xml/chrwiki-latest-pages-articles.xml"
if os.path.exists(wiki_path):
    with open(wiki_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("Page count:", content.count('<page>'))
else:
    print("File not found:", wiki_path)
