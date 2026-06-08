import msal
import requests
import os
import sys
import resource
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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

# Worker paralleli per la SCANSIONE della struttura remota.
# La scansione è I/O bound (molte richieste API leggere): puoi alzare fino a 16-32.
MAX_SCAN_WORKERS = 16

# Worker paralleli per il DOWNLOAD dei file.
# Il download è banda-bound: inizia con 8, alza se non hai 429.
MAX_DOWNLOAD_WORKERS = 8

# Mostra progresso ogni N file (riduci il rumore con 70k+ file)
LOG_OGNI_N_FILE = 100

# Timeout in secondi per ogni singola richiesta HTTP verso Graph API
REQUEST_TIMEOUT = 30

# Cartelle remote da NON scaricare (nomi esatti)
CARTELLE_ESCLUSE = {}


# ============================================================
#  NON MODIFICARE DA QUI IN POI
# ============================================================

CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
SCOPES = ["https://graph.microsoft.com/Files.ReadWrite.All"]
AUTHORITY = "https://login.microsoftonline.com/common"

_token_lock = threading.Lock()
_print_lock = threading.Lock()


@dataclass
class Contatori:
    """
    Stato del progresso incapsulato in un oggetto per evitare variabili globali
    mutabili, rendendo il codice rientrante e più facile da testare.
    """
    totale: int = 0
    processati: int = 0
    scaricati: int = 0
    saltati: int = 0
    errori: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def incrementa(self, esito: str) -> int:
        """Aggiorna i contatori in modo thread-safe e restituisce il valore corrente."""
        with self._lock:
            self.processati += 1
            if esito == "scaricato":
                self.scaricati += 1
            elif esito == "saltato":
                self.saltati += 1
            else:
                self.errori += 1
            return self.processati


def _alza_limite_fd():
    """Alza il limite di file aperti del processo al massimo consentito dal sistema."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        nuovo_soft = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (nuovo_soft, hard))
        print(f"Limite file aperti: {soft} → {nuovo_soft} (hard limit: {hard})")
    except Exception as e:
        print(f"Avviso: impossibile alzare il limite dei file aperti: {e}")


def log(msg: str):
    with _print_lock:
        print(msg)


def log_progresso(contatori: Contatori, esito: str):
    """Stampa il progresso solo ogni LOG_OGNI_N_FILE file per ridurre l'overhead di I/O."""
    processati = contatori.incrementa(esito)
    if processati % LOG_OGNI_N_FILE == 0 or processati == contatori.totale:
        with _print_lock:
            print(f"Progresso: {processati}/{contatori.totale} ({esito})")


def graph_request(method: str, url: str, max_retry: int = 5, **kwargs) -> requests.Response:
    """
    Esegue una richiesta HTTP verso Microsoft Graph con:
    - retry automatico su 429 (rate limit)
    - timeout fisso per evitare worker bloccati indefinitamente
    """
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)

    for tentativo in range(max_retry):
        response = requests.request(method, url, **kwargs)
        if response.status_code != 429:
            return response
        try:
            retry_after = int(response.json().get("error", {}).get("retryAfterSeconds", 60))
        except Exception:
            retry_after = 60
        log(f"Rate limit (429), attendo {retry_after}s prima di riprovare... "
            f"(tentativo {tentativo + 1}/{max_retry})")
        time.sleep(retry_after)

    raise Exception(f"Rate limit persistente dopo {max_retry} tentativi su {url}")


def autenticati():
    """Esegue il login tramite Device Code Flow. Restituisce (app, accounts)."""
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            print("Login effettuato dalla cache.")
            return app, accounts

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise Exception("Errore nell'avvio del device flow: " + str(flow))

    print("\n" + "=" * 60)
    print("AUTENTICAZIONE RICHIESTA")
    print("=" * 60)
    print(flow["message"])
    print("=" * 60 + "\n")
    input("Premi Invio dopo aver completato il login nel browser...")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise Exception("Login fallito: " + result.get("error_description", "Errore sconosciuto"))

    print("Login effettuato con successo!\n")
    accounts = app.get_accounts()
    return app, accounts


def get_token(app, accounts) -> str:
    """Ottiene un token valido, rinnovandolo automaticamente via refresh token se scaduto."""
    with _token_lock:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if result and "access_token" in result:
        return result["access_token"]
    raise Exception("Impossibile rinnovare il token di accesso.")


def data_modifica_locale(percorso_locale: str) -> datetime:
    """Restituisce la data di ultima modifica del file locale (con timezone UTC).
    Usa lstat per non seguire i symlink."""
    timestamp = os.lstat(percorso_locale).st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def lista_contenuto(app, accounts, percorso_remoto: str) -> list:
    """Restituisce il contenuto (file e cartelle) di una cartella su OneDrive,
    gestendo automaticamente la paginazione via @odata.nextLink."""
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


def _get_download_url(app, accounts, item_id: str, nome_file: str) -> str | None:
    """
    Richiede un downloadUrl fresco dall'API Graph usando l'item ID.
    Il pre-signed URL scade in ~1h, quindi va sempre riacquisito prima del download.
    Restituisce None in caso di errore.
    """
    token = get_token(app, accounts)
    url_meta = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}"
    response = graph_request("GET", url_meta, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        log(f"ERRORE metadati {nome_file}: {response.status_code}")
        return None
    download_url = response.json().get("@microsoft.graph.downloadUrl")
    if not download_url:
        log(f"ERRORE: downloadUrl non disponibile per {nome_file}")
    return download_url


def scarica_file(app, accounts, elemento: dict, percorso_locale: str) -> str:
    """
    Scarica un file da OneDrive solo se è più recente di quello locale.

    - downloadUrl riacquisito sempre fresco prima del download
    - Gestione corretta del 401 durante lo streaming: chiude la risposta
      precedente prima di aprirne una nuova (evita leak di connessioni)
    - NamedTemporaryFile + os.replace per atomicità (file mai corrotti)
    - Chunk da 1MB per ridurre l'overhead di sistema
    """
    nome_file = elemento["name"]
    item_id = elemento["id"]

    data_str = elemento["lastModifiedDateTime"]
    data_remota = datetime.fromisoformat(data_str.replace("Z", "+00:00"))

    if os.path.exists(percorso_locale):
        data_locale = data_modifica_locale(percorso_locale)
        if data_remota <= data_locale:
            return "saltato"

    cartella_dest = os.path.dirname(percorso_locale)
    os.makedirs(cartella_dest, exist_ok=True)

    download_url = _get_download_url(app, accounts, item_id, nome_file)
    if not download_url:
        return "errore"

    percorso_tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=cartella_dest, delete=False) as tmp:
            percorso_tmp = tmp.name

            with requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 401:
                    # downloadUrl scaduto durante il download:
                    # chiude esplicitamente la risposta corrente prima di riaprire
                    r.close()
                    download_url = _get_download_url(app, accounts, item_id, nome_file)
                    if not download_url:
                        return "errore"
                    with requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT) as r2:
                        r2.raise_for_status()
                        for chunk in r2.iter_content(chunk_size=1024 * 1024):
                            tmp.write(chunk)
                else:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        tmp.write(chunk)

        os.replace(percorso_tmp, percorso_locale)
        percorso_tmp = None

    finally:
        if percorso_tmp is not None and os.path.exists(percorso_tmp):
            try:
                os.remove(percorso_tmp)
            except OSError:
                pass

    return "scaricato"


def _raccogli_file_bfs(app, accounts, radice_remota: str, radice_locale: str) -> list:
    """
    Visita BFS della struttura remota con parallelismo continuo (work-stealing).

    Perché NON usiamo la ricorsione con un ThreadPoolExecutor:
    - Nella versione ricorsiva ogni worker chiama executor.submit() per i figli
      e poi si blocca su as_completed() in attesa dei risultati.
    - Con una struttura densa, tutti i worker si bloccano contemporaneamente
      in attesa di subtask che non possono partire → deadlock o starvation.

    Soluzione: BFS iterativo con coda esplicita e contatore di task in volo.

    Struttura dati:
    - `coda`         : deque di (percorso_remoto, percorso_locale) da esplorare.
                       Modificata sia dal thread principale sia dai callback dei
                       future, quindi protetta da `coda_lock`.
    - `in_volo`      : numero di task attualmente in esecuzione nel pool,
                       protetto dallo stesso `coda_lock`.
    - `coda_non_vuota`: Condition legata a `coda_lock`, usata per segnalare al
                        loop principale che ci sono nuovi item da schedulare o
                        che la scansione è terminata.

    Ciclo principale (thread principale, non occupa worker):
    1. Finché la coda non è vuota, preleva tutti gli item disponibili e
       sottomette un future per ciascuno.
    2. Se la coda è vuota ma ci sono task in volo, aspetta sulla Condition
       finché un future non la notifica (potrebbe aggiungere nuove cartelle).
    3. Quando coda è vuota E in_volo == 0, la scansione è completa.

    Callback dei future (eseguito nel thread del worker al completamento):
    - Aggiunge le sottocartelle trovate alla coda.
    - Decrementa `in_volo`.
    - Notifica il loop principale via `coda_non_vuota.notify_all()`.

    Garanzie:
    - Nessun worker si blocca mai in attesa di altri worker.
    - Il pool è saturo finché ci sono cartelle da esplorare.
    - Nessun deadlock possibile: il loop principale non occupa worker.
    - Thread-safe: coda e in_volo modificati solo sotto `coda_lock`.
    """
    tasks_file = []
    tasks_lock = threading.Lock()

    coda_lock = threading.Lock()
    coda_non_vuota = threading.Condition(coda_lock)
    coda = deque()
    coda.append((radice_remota.rstrip("/"), radice_locale))
    in_volo = [0]  # lista per mutabilità in closure Python < 3.10

    def on_cartella_completata(future, percorso_remoto: str):
        """
        Callback eseguito nel thread del worker al completamento del future.
        Aggiunge le sottocartelle trovate alla coda e notifica il loop principale.
        """
        try:
            elementi = future.result()
        except Exception as e:
            log(f"  ERRORE scansione {percorso_remoto}: {e}")
            with coda_non_vuota:
                in_volo[0] -= 1
                coda_non_vuota.notify_all()
            return

        base = percorso_remoto.rstrip("/")
        file_trovati = 0
        nuove_cartelle = []

        for elemento in elementi:
            nome = elemento["name"]
            # Calcola il percorso locale corrispondente:
            # toglie il prefisso della radice remota e ricostruisce sotto radice_locale
            rel = percorso_remoto[len(radice_remota.rstrip("/")):]
            dest_locale = os.path.join(radice_locale, rel.lstrip("/"), nome)

            if "folder" in elemento:
                if nome in CARTELLE_ESCLUSE:
                    log(f"  Saltata (esclusa): {base}/{nome}")
                    continue
                os.makedirs(dest_locale, exist_ok=True)
                nuove_cartelle.append((f"{base}/{nome}", dest_locale))
            elif "file" in elemento:
                with tasks_lock:
                    tasks_file.append((elemento, dest_locale))
                file_trovati += 1

        log(f"  {base}: {file_trovati} file, {len(nuove_cartelle)} sottocartelle")

        with coda_non_vuota:
            coda.extend(nuove_cartelle)
            in_volo[0] -= 1
            coda_non_vuota.notify_all()

    with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as executor:
        while True:
            with coda_non_vuota:
                # Sottometti tutti gli item disponibili in coda
                while coda:
                    percorso_rem, percorso_loc = coda.popleft()
                    future = executor.submit(lista_contenuto, app, accounts, percorso_rem)
                    in_volo[0] += 1
                    future.add_done_callback(
                        lambda f, rem=percorso_rem: on_cartella_completata(f, rem)
                    )

                # Scansione terminata: coda vuota e nessun task in esecuzione
                if in_volo[0] == 0:
                    break

                # Coda vuota ma ci sono task in volo: aspetta nuove notifiche
                coda_non_vuota.wait()

    return tasks_file


def scarica_cartella(app, accounts, cartella_remota: str, cartella_locale: str):
    """
    Scarica ricorsivamente tutti i file di una cartella da OneDrive.

    Pipeline in due fasi con executor separati:
    1. Scansione BFS parallela della struttura remota  (MAX_SCAN_WORKERS)
    2. Download parallelo dei file                     (MAX_DOWNLOAD_WORKERS)
    """
    os.makedirs(cartella_locale, exist_ok=True)
    contatori = Contatori()

    print(f"\nDownload da OneDrive: {cartella_remota}")
    print(f"Destinazione locale:  {cartella_locale}")
    print(f"Worker scansione:     {MAX_SCAN_WORKERS}")
    print(f"Worker download:      {MAX_DOWNLOAD_WORKERS}")

    # ── Fase 1: scansione BFS ────────────────────────────────────────────────
    print("\nScansione struttura remota (BFS parallelo)...")
    tasks = _raccogli_file_bfs(app, accounts, cartella_remota, cartella_locale)

    contatori.totale = len(tasks)
    print(f"{contatori.totale} file trovati\n")

    # ── Fase 2: download ─────────────────────────────────────────────────────
    def download_task(elemento, dest_locale) -> str:
        try:
            esito = scarica_file(app, accounts, elemento, dest_locale)
            log_progresso(contatori, esito)
            return esito
        except Exception as e:
            log(f"Errore su {elemento['name']}: {e}")
            log_progresso(contatori, "errore")
            return "errore"

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as dl_executor:
        futures = {
            dl_executor.submit(download_task, elemento, dest_locale): (elemento, dest_locale)
            for elemento, dest_locale in tasks
        }
        for future in as_completed(futures):
            future.result()

    print(f"\n{'=' * 50}")
    print(f"Nuovi/aggiornati:         {contatori.scaricati}")
    print(f"Saltati (già aggiornati): {contatori.saltati}")
    print(f"Errori:                   {contatori.errori}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    _alza_limite_fd()
    app, accounts = autenticati()
    scarica_cartella(app, accounts, CARTELLA_REMOTA, CARTELLA_LOCALE)
