
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import requests

@st.cache_resource
def load_model():
    return SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

model = load_model()

def get_access_token(client_id, client_secret):
    url = "https://ops.epo.org/3.2/auth/accesstoken"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    r = requests.post(url, headers=headers, data=data, auth=(client_id, client_secret))
    return r.json().get("access_token")

def search_patents(query, access_token, max_results=5):
    base_url = "https://ops.epo.org/3.2/rest-services/published-data/search"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    params = {"q": query, "Range": f"1-{max_results}"}
    r = requests.get(base_url, headers=headers, params=params)
    return r.json()

def get_biblio(country, number, kind, access_token):
    url = f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{country}{number}{kind}/biblio"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    r = requests.get(url, headers=headers)
    return r.json() if r.status_code == 200 else {}

def semantic_filter(abstracts, domanda, top_k=3):
    embeddings = model.encode(abstracts, convert_to_tensor=True)
    domanda_emb = model.encode(domanda, convert_to_tensor=True)
    risultati = util.semantic_search(domanda_emb, embeddings, top_k=top_k)[0]
    return [(abstracts[r['corpus_id']], r['score']) for r in risultati]

st.title("Motore semantico brevettuale")
client_id = st.text_input("Client ID EPO", type="password")
client_secret = st.text_input("Client Secret EPO", type="password")
domanda = st.text_input("Scrivi la tua domanda (es. tecnologie per ridurre il consumo energetico)")

if st.button("Cerca") and client_id and client_secret and domanda:
    try:
        token = get_access_token(client_id, client_secret)
        risultati = search_patents(domanda, token)
        entries = risultati["ops:world-patent-data"]["ops:biblio-search"]["ops:search-result"]["ops:publication-reference"]

        dati = []
        for e in entries:
            doc = e["document-id"]
            country = doc["country"]["$"]
            number = doc["doc-number"]["$"]
            kind = doc["kind"]["$"]
            data = get_biblio(country, number, kind, token)
            try:
                title = data["ops:world-patent-data"]["exchange-documents"]["exchange-document"]["bibliographic-data"]["invention-title"][0]["$"]
            except:
                title = "(titolo non disponibile)"
            try:
                abstract = data["ops:world-patent-data"]["exchange-documents"]["exchange-document"]["abstract"]["p"][0]["$"]
            except:
                abstract = "(abstract non disponibile)"
            if abstract != "(abstract non disponibile)":
                dati.append({"numero": f"{country}{number}{kind}", "titolo": title, "abstract": abstract})

        if not dati:
            st.warning("Nessun abstract disponibile nei risultati.")
        else:
            abstract_list = [x["abstract"] for x in dati]
            risultati_finali = semantic_filter(abstract_list, domanda)
            df = pd.DataFrame([{
                "Brevetto": dati[i]["numero"],
                "Titolo": dati[i]["titolo"],
                "Abstract": risultati_finali[j][0],
                "Similarità": round(risultati_finali[j][1], 4)
            } for j, i in enumerate([abstract_list.index(r[0]) for r in risultati_finali])])
            st.dataframe(df)
    except Exception as e:
        st.error(f"Errore: {e}")
