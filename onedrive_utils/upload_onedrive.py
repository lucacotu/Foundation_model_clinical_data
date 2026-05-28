import msal
import requests
import os
import sys
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
#  MODIFICA QUESTE VARIABILI CON I TUOI DATI
# ============================================================

# Cartella LOCALE che vuoi caricare (percorso assoluto o relativo)
CARTELLA_LOCALE = "../../Foundation_model_clinical_data"

# Cartella REMOTA su OneDrive dove caricare i file
# Esempio: "Tesi" oppure "Documenti/Tesi/Capitoli"
CARTELLA_REMOTA = "Tesi/Foundation_model_clinical_data"

# Worker paralleli per l'upload (aumenta con cautela: troppi possono causare rate limiting)
MAX_WORKERS = 4

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
        log(f"  ⏳ Rate limit (429), attendo {retry_after}s prima di riprovare...")
        time.sleep(retry_after)
    raise Exception(f"Rate limit persistente dopo {max_retry} tentativi su {url}")


def autenticati():
    """Esegue il login tramite Device Code Flow. Restituisce (app, accounts)."""
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            print("✅ Login effettuato dalla cache.")
            return app, accounts

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise Exception("Errore nell'avvio del device flow: " + str(flow))

    print("\n" + "="*60)
    print("🔐 AUTENTICAZIONE RICHIESTA")
    print("="*60)
    print(flow["message"])
    print("="*60 + "\n")
    input("Premi Invio dopo aver completato il login nel browser...")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise Exception("Login fallito: " + result.get("error_description", "Errore sconosciuto"))

    print("✅ Login effettuato con successo!\n")
    accounts = app.get_accounts()
    return app, accounts


def get_token(app, accounts):
    """Ottiene un token valido, rinnovandolo automaticamente via refresh token se scaduto."""
    with _token_lock:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if result and "access_token" in result:
        return result["access_token"]
    raise Exception("Impossibile rinnovare il token di accesso.")


def data_modifica_remota(app, accounts, percorso_remoto):
    """Restituisce la data di ultima modifica del file su OneDrive, o None se non esiste."""
    token = get_token(app, accounts)
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}"
    response = graph_request("GET", url, headers=headers)

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise Exception(f"Errore nel controllo remoto: {response.status_code} — {response.text}")

    data_str = response.json()["lastModifiedDateTime"]
    return datetime.fromisoformat(data_str.replace("Z", "+00:00"))


def data_modifica_locale(percorso_locale):
    """Restituisce la data di ultima modifica del file locale (con timezone UTC)."""
    timestamp = os.path.getmtime(percorso_locale)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def carica_file(app, accounts, percorso_locale, percorso_remoto):
    """
    Carica un file su OneDrive solo se è più recente di quello remoto.
    Gestisce automaticamente file grandi (> 4MB) e rinnova il token se necessario.
    """
    nome_file = os.path.basename(percorso_locale)
    dimensione = os.path.getsize(percorso_locale)

    data_locale = data_modifica_locale(percorso_locale)
    data_remota = data_modifica_remota(app, accounts, percorso_remoto)

    if data_remota is not None:
        if data_locale <= data_remota:
            log(f"  ⏭️  {nome_file} — saltato (remoto più recente o uguale)")
            return "saltato"
        else:
            log(f"  🔄 {nome_file} — aggiornamento (locale più recente)")
    else:
        log(f"  🆕 {nome_file} — nuovo file")

    if dimensione <= 4 * 1024 * 1024:
        token = get_token(app, accounts)
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/content"
        with open(percorso_locale, "rb") as f:
            response = graph_request(
                "PUT", url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
                data=f
            )
        if response.status_code in (200, 201):
            log(f"  ✅ {nome_file}")
            return "caricato"
        else:
            log(f"  ❌ {nome_file} — Errore {response.status_code}: {response.text}")
            return "errore"

    else:
        # Per file grandi, il token serve solo per creare la sessione;
        # i chunk usano l'uploadUrl che non richiede Authorization.
        token = get_token(app, accounts)
        log(f"  📦 {nome_file} ({dimensione // (1024*1024)} MB) — upload in sessione...")
        url_sessione = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/createUploadSession"
        sessione = graph_request(
            "POST", url_sessione,
            headers={"Authorization": f"Bearer {token}"},
            json={"item": {"@microsoft.graph.conflictBehavior": "replace"}}
        )

        if sessione.status_code != 200:
            log(f"  ❌ Impossibile creare la sessione per {nome_file}: {sessione.text}")
            return "errore"

        upload_url = sessione.json()["uploadUrl"]
        chunk_size = 10 * 1024 * 1024  # 10MB per chunk
        inviati = 0
        MAX_RETRY = 3
        completato = False

        try:
            with open(percorso_locale, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    fine = inviati + len(chunk) - 1
                    chunk_headers = {
                        "Content-Range": f"bytes {inviati}-{fine}/{dimensione}",
                        "Content-Length": str(len(chunk))
                    }

                    for tentativo in range(1, MAX_RETRY + 1):
                        r = requests.put(upload_url, headers=chunk_headers, data=chunk)
                        if r.status_code in (200, 201, 202):
                            break
                        if tentativo < MAX_RETRY:
                            log(f"\n  ⚠️  Chunk {inviati}-{fine} fallito ({r.status_code}), retry {tentativo}/{MAX_RETRY-1}...")
                            time.sleep(2 ** tentativo)
                    else:
                        log(f"\n  ❌ {nome_file} — chunk {inviati}-{fine} fallito dopo {MAX_RETRY} tentativi.")
                        return "errore"

                    inviati += len(chunk)
                    percentuale = int((inviati / dimensione) * 100)
                    log(f"  ⬆️  {nome_file}: {percentuale}%")

            completato = True
        finally:
            if not completato:
                requests.delete(upload_url)
                log(f"\n  ❌ {nome_file} — sessione annullata per errore imprevisto.")

        log(f"  ✅ {nome_file}")
        return "caricato"


def carica_cartella(app, accounts, cartella_locale, cartella_remota):
    """Carica ricorsivamente una cartella locale su OneDrive con upload paralleli."""
    if not os.path.isdir(cartella_locale):
        print(f"❌ La cartella locale '{cartella_locale}' non esiste.")
        sys.exit(1)

    print(f"\n📂 Avvio upload da: {cartella_locale}")
    print(f"   → Destinazione OneDrive: {cartella_remota}")
    print(f"   → Worker paralleli: {MAX_WORKERS}\n")

    # Raccoglie tutti i file prima di avviare i worker
    tasks = []
    for root, dirs, files in os.walk(cartella_locale):
        relativo = os.path.relpath(root, cartella_locale)
        if relativo == ".":
            remoto_corrente = cartella_remota
        else:
            relativo = relativo.replace(os.sep, "/")
            remoto_corrente = f"{cartella_remota}/{relativo}"
        for file in files:
            percorso_locale = os.path.join(root, file)
            percorso_remoto = f"{remoto_corrente}/{file}"
            tasks.append((percorso_locale, percorso_remoto))

    caricati = saltati = errori = 0
    file_con_errore = []

    def upload_task(percorso_locale, percorso_remoto):
        try:
            return carica_file(app, accounts, percorso_locale, percorso_remoto)
        except Exception as e:
            log(f"  ❌ Errore su {os.path.basename(percorso_locale)}: {e}")
            return "errore"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_task, p_loc, p_rem): (p_loc, p_rem)
            for p_loc, p_rem in tasks
        }
        for future in as_completed(futures):
            p_loc, p_rem = futures[future]
            esito = future.result()
            if esito == "caricato":
                caricati += 1
            elif esito == "saltato":
                saltati += 1
            else:
                errori += 1
                file_con_errore.append(p_loc)

    print(f"\n{'='*50}")
    print(f"🆕 Nuovi/aggiornati: {caricati}")
    print(f"⏭️  Saltati (già aggiornati): {saltati}")
    print(f"❌ Errori: {errori}")
    if file_con_errore:
        print("\nFile non caricati:")
        for path in file_con_errore:
            print(f"  - {os.path.basename(path)}  ({path})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    app, accounts = autenticati()
    carica_cartella(app, accounts, CARTELLA_LOCALE, CARTELLA_REMOTA)

