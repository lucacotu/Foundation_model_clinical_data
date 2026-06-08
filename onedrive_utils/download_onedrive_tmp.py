import msal
import requests
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
#  MODIFICA QUESTE VARIABILI CON I TUOI DATI
# ============================================================

# Cartella REMOTA su OneDrive da scaricare
# Esempio: "Tesi" oppure "Documenti/Tesi/Capitoli"
CARTELLA_REMOTA = "Tesi/Foundation_model_clinical_data/"

# Cartella LOCALE dove salvare i file scaricati (verrà creata se non esiste)
CARTELLA_LOCALE = "../../Foundation_model_clinical_data/"

# Worker paralleli per il download (aumenta con cautela: troppi possono causare rate limiting)
MAX_WORKERS = 1

# ============================================================
#  NON MODIFICARE DA QUI IN POI
# ============================================================

CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
SCOPES = ["https://graph.microsoft.com/Files.ReadWrite.All"]
AUTHORITY = "https://login.microsoftonline.com/common"

_token_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg)


def graph_request(method, url, max_retry=5, **kwargs):
    """Esegue una richiesta HTTP verso Microsoft Graph con retry automatico su 429."""
    for tentativo in range(max_retry):
        response = requests.request(method, url, **kwargs)
        if response.status_code != 429:
            return response
        try:
            retry_after = int(response.json().get("error", {}).get("retryAfterSeconds", 60))
        except Exception:
            retry_after = 60
        log(f"Rate limit (429), attendo {retry_after}s prima di riprovare...")
        time.sleep(retry_after)
    raise Exception(f"Rate limit persistente dopo {max_retry} tentativi su {url}")


def autenticati():
    """Esegue il login tramite Device Code Flow. Restituisce (app, accounts)."""
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            print(" Login effettuato dalla cache.")
            return app, accounts

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise Exception("Errore nell'avvio del device flow: " + str(flow))

    print("\n" + "="*60)
    print("AUTENTICAZIONE RICHIESTA")
    print("="*60)
    print(flow["message"])
    print("="*60 + "\n")
    input("Premi Invio dopo aver completato il login nel browser...")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise Exception("Login fallito: " + result.get("error_description", "Errore sconosciuto"))

    print("Login effettuato con successo!\n")
    accounts = app.get_accounts()
    return app, accounts


def get_token(app, accounts):
    """Ottiene un token valido, rinnovandolo automaticamente via refresh token se scaduto."""
    with _token_lock:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if result and "access_token" in result:
        return result["access_token"]
    raise Exception("Impossibile rinnovare il token di accesso.")


def data_modifica_locale(percorso_locale):
    """Restituisce la data di ultima modifica del file locale (con timezone UTC)."""
    timestamp = os.path.getmtime(percorso_locale)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def lista_contenuto(app, accounts, percorso_remoto):
    """Restituisce il contenuto (file e cartelle) di una cartella su OneDrive."""
    tutti_gli_elementi = []
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/children"

    while url:
        token = get_token(app, accounts)
        response = graph_request("GET", url, headers={"Authorization": f"Bearer {token}"})
        if response.status_code == 404:
            print(f"La cartella remota '{percorso_remoto}' non esiste su OneDrive.")
            sys.exit(1)
        if response.status_code != 200:
            raise Exception(f"Errore nella lista: {response.status_code} — {response.text}")

        dati = response.json()
        tutti_gli_elementi.extend(dati.get("value", []))
        url = dati.get("@odata.nextLink")

    return tutti_gli_elementi

def scarica_file(app, accounts, elemento, percorso_locale):
    nome_file = elemento["name"]
    item_id = elemento["id"]

    data_str = elemento["lastModifiedDateTime"]
    data_remota = datetime.fromisoformat(data_str.replace("Z", "+00:00"))

    if os.path.exists(percorso_locale):
        data_locale = data_modifica_locale(percorso_locale)
        if data_remota <= data_locale:
            log(f"{nome_file} — saltato (locale più recente o uguale)")
            return "saltato"
        else:
            log(f"{nome_file} — aggiornamento (remoto più recente)")
    else:
        log(f"{nome_file} — nuovo file")

    # Ri-richiedi il downloadUrl fresco usando l'item ID
    token = get_token(app, accounts)
    url_meta = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}"
    response = graph_request("GET", url_meta, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        log(f"{nome_file} — errore nel recupero metadati: {response.status_code}")
        return "errore"
    download_url = response.json().get("@microsoft.graph.downloadUrl")
    if not download_url:
        log(f"{nome_file} — downloadUrl non disponibile")
        return "errore"

    cartella_dest = os.path.dirname(percorso_locale)
    os.makedirs(cartella_dest, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(dir=cartella_dest, delete=False) as tmp:
            percorso_tmp = tmp.name
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                dimensione_totale = int(r.headers.get("Content-Length", 0))
                scaricati = 0
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                    scaricati += len(chunk)
                    if dimensione_totale:
                        percentuale = int((scaricati / dimensione_totale) * 100)
                        log(f" {nome_file}: {percentuale}%")
        os.replace(percorso_tmp, percorso_locale)
        percorso_tmp = None
    finally:
        if percorso_tmp is not None and os.path.exists(percorso_tmp):
            try:
                os.remove(percorso_tmp)
            except OSError:
                pass

    log(f"{nome_file} — completato")
    return "scaricato"


def _raccogli_file(app, accounts, percorso_remoto, percorso_locale):
    """
    Traversa ricorsivamente la struttura remota e restituisce una lista di
    (elemento, dest_locale) per tutti i file trovati.
    Crea le directory locali necessarie durante la traversata.
    """
    tasks = []
    elementi = lista_contenuto(app, accounts, percorso_remoto)

    for elemento in elementi:
        nome = elemento["name"]
        dest_locale = os.path.join(percorso_locale, nome)

        if "folder" in elemento:
            os.makedirs(dest_locale, exist_ok=True)
            tasks.extend(_raccogli_file(app, accounts, f"{percorso_remoto}/{nome}", dest_locale))
        elif "file" in elemento:
            tasks.append((elemento, dest_locale))

    return tasks


def scarica_cartella(app, accounts, cartella_remota, cartella_locale):
    """Scarica ricorsivamente tutti i file di una cartella da OneDrive con download paralleli."""
    os.makedirs(cartella_locale, exist_ok=True)

    print(f"\n Download da OneDrive: {cartella_remota}")
    print(f"Destinazione locale: {cartella_locale}")
    print(f"Worker paralleli: {MAX_WORKERS}")
    print("\n Scansione struttura remota...")

    tasks = _raccogli_file(app, accounts, cartella_remota, cartella_locale)
    print(f"{len(tasks)} file trovati\n")

    scaricati = saltati = errori = 0
    contatori_lock = threading.Lock()

    def download_task(elemento, dest_locale):
        try:
            return scarica_file(app, accounts, elemento, dest_locale)
        except Exception as e:
            log(f"Errore su {elemento['name']}: {e}")
            return "errore"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_task, elemento, dest_locale): (elemento, dest_locale)
            for elemento, dest_locale in tasks
        }
        for future in as_completed(futures):
            esito = future.result()
            with contatori_lock:
                if esito == "scaricato":
                    scaricati += 1
                elif esito == "saltato":
                    saltati += 1
                else:
                    errori += 1

    print(f"\n{'='*50}")
    print(f"Nuovi/aggiornati: {scaricati}")
    print(f"Saltati (già aggiornati): {saltati}")
    print(f"Errori: {errori}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    app, accounts = autenticati()
    scarica_cartella(app, accounts, CARTELLA_REMOTA, CARTELLA_LOCALE)
