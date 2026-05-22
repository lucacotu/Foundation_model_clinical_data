import msal
import requests
import os
import sys
import time
from datetime import datetime, timezone

# ============================================================
#  MODIFICA QUESTE VARIABILI CON I TUOI DATI
# ============================================================

# Cartella LOCALE che vuoi caricare (percorso assoluto o relativo)
CARTELLA_LOCALE = "../../Foundation_model_clinical_data/"

# Cartella REMOTA su OneDrive dove caricare i file
# Esempio: "Tesi" oppure "Documenti/Tesi/Capitoli"
CARTELLA_REMOTA = "Tesi/Foundation_model_clinical_data"

# ============================================================
#  NON MODIFICARE DA QUI IN POI
# ============================================================

CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
SCOPES = ["https://graph.microsoft.com/Files.ReadWrite.All"]
AUTHORITY = "https://login.microsoftonline.com/common"


def autenticati():
    """Esegue il login tramite Device Code Flow."""
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            print("✅ Login effettuato dalla cache.")
            return result["access_token"]

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
    return result["access_token"]


def data_modifica_remota(token, percorso_remoto):
    """
    Restituisce la data di ultima modifica del file su OneDrive.
    Ritorna None se il file non esiste.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}"
    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        return None  # File non esiste in remoto
    if response.status_code != 200:
        raise Exception(f"Errore nel controllo remoto: {response.status_code} — {response.text}")

    data_str = response.json()["lastModifiedDateTime"]
    return datetime.fromisoformat(data_str.replace("Z", "+00:00"))


def data_modifica_locale(percorso_locale):
    """Restituisce la data di ultima modifica del file locale (con timezone UTC)."""
    timestamp = os.path.getmtime(percorso_locale)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def carica_file(token, percorso_locale, percorso_remoto):
    """
    Carica un file su OneDrive solo se è più recente di quello remoto.
    Gestisce automaticamente file grandi (> 4MB).
    """
    headers = {"Authorization": f"Bearer {token}"}
    nome_file = os.path.basename(percorso_locale)
    dimensione = os.path.getsize(percorso_locale)

    # --- Confronto date ---
    data_locale = data_modifica_locale(percorso_locale)
    data_remota = data_modifica_remota(token, percorso_remoto)

    if data_remota is not None:
        if data_locale <= data_remota:
            print(f"  ⏭️  {nome_file} — saltato (remoto più recente o uguale)")
            return "saltato"
        else:
            print(f"  🔄 {nome_file} — aggiornamento (locale più recente)")
    else:
        print(f"  🆕 {nome_file} — nuovo file")

    # --- Upload ---
    if dimensione <= 4 * 1024 * 1024:
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/content"
        with open(percorso_locale, "rb") as f:
            response = requests.put(
                url,
                headers={**headers, "Content-Type": "application/octet-stream"},
                data=f
            )
        if response.status_code in (200, 201):
            print(f"  ✅ {nome_file}")
            return "caricato"
        else:
            print(f"  ❌ {nome_file} — Errore {response.status_code}: {response.text}")
            return "errore"

    else:
        print(f"  📦 {nome_file} ({dimensione // (1024*1024)} MB) — upload in sessione...")
        url_sessione = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/createUploadSession"
        sessione = requests.post(url_sessione, headers=headers, json={
            "item": {"@microsoft.graph.conflictBehavior": "replace"}
        })

        if sessione.status_code != 200:
            print(f"  ❌ Impossibile creare la sessione per {nome_file}: {sessione.text}")
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
                        # 202 = chunk intermedio accettato, 200/201 = ultimo chunk completato
                        if r.status_code in (200, 201, 202):
                            break
                        if tentativo < MAX_RETRY:
                            print(f"\n  ⚠️  Chunk {inviati}-{fine} fallito ({r.status_code}), retry {tentativo}/{MAX_RETRY-1}...")
                            time.sleep(2 ** tentativo)  # backoff esponenziale: 2s, 4s
                    else:
                        print(f"\n  ❌ {nome_file} — chunk {inviati}-{fine} fallito dopo {MAX_RETRY} tentativi.")
                        return "errore"

                    inviati += len(chunk)
                    percentuale = int((inviati / dimensione) * 100)
                    print(f"  ⬆️  {nome_file}: {percentuale}%", end="\r")

            completato = True
        finally:
            # Garantisce la cancellazione della sessione in caso di eccezione imprevista
            if not completato:
                requests.delete(upload_url)
                print(f"\n  ❌ {nome_file} — sessione annullata per errore imprevisto.")

        print(f"\n  ✅ {nome_file}")
        return "caricato"


def carica_cartella(token, cartella_locale, cartella_remota):
    """Carica ricorsivamente una cartella locale su OneDrive."""
    if not os.path.isdir(cartella_locale):
        print(f"❌ La cartella locale '{cartella_locale}' non esiste.")
        sys.exit(1)

    print(f"\n📂 Avvio upload da: {cartella_locale}")
    print(f"   → Destinazione OneDrive: {cartella_remota}\n")

    caricati = saltati = errori = 0

    for root, dirs, files in os.walk(cartella_locale):
        relativo = os.path.relpath(root, cartella_locale)
        if relativo == ".":
            remoto_corrente = cartella_remota
        else:
            relativo = relativo.replace(os.sep, "/")
            remoto_corrente = f"{cartella_remota}/{relativo}"

        if files:
            print(f"\n📁 {remoto_corrente}/")

        for file in files:
            percorso_locale = os.path.join(root, file)
            percorso_remoto = f"{remoto_corrente}/{file}"
            try:
                esito = carica_file(token, percorso_locale, percorso_remoto)
                if esito == "caricato":
                    caricati += 1
                elif esito == "saltato":
                    saltati += 1
                else:
                    errori += 1
            except Exception as e:
                print(f"  ❌ Errore su {file}: {e}")
                errori += 1

    print(f"\n{'='*50}")
    print(f"🆕 Nuovi/aggiornati: {caricati}")
    print(f"⏭️  Saltati (già aggiornati): {saltati}")
    print(f"❌ Errori: {errori}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    token = autenticati()
    carica_cartella(token, CARTELLA_LOCALE, CARTELLA_REMOTA)
