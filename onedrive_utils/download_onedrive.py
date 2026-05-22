import msal
import requests
import os
import sys
import tempfile
from datetime import datetime, timezone

# ============================================================
#  MODIFICA QUESTE VARIABILI CON I TUOI DATI
# ============================================================

# Cartella REMOTA su OneDrive da scaricare
# Esempio: "Tesi" oppure "Documenti/Tesi/Capitoli"
CARTELLA_REMOTA = "Tesi/checkpoints/"

# Cartella LOCALE dove salvare i file scaricati (verrà creata se non esiste)
CARTELLA_LOCALE = "../checkpoints/"

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


def data_modifica_locale(percorso_locale):
    """Restituisce la data di ultima modifica del file locale (con timezone UTC)."""
    timestamp = os.path.getmtime(percorso_locale)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def lista_contenuto(token, percorso_remoto):
    """Restituisce il contenuto (file e cartelle) di una cartella su OneDrive."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{percorso_remoto}:/children"

    tutti_gli_elementi = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            print(f"❌ La cartella remota '{percorso_remoto}' non esiste su OneDrive.")
            sys.exit(1)
        if response.status_code != 200:
            raise Exception(f"Errore nella lista: {response.status_code} — {response.text}")

        dati = response.json()
        tutti_gli_elementi.extend(dati.get("value", []))
        url = dati.get("@odata.nextLink")

    return tutti_gli_elementi


def scarica_file(token, elemento, percorso_locale):
    """
    Scarica un file da OneDrive solo se è più recente di quello locale.
    """
    nome_file = elemento["name"]
    download_url = elemento["@microsoft.graph.downloadUrl"]

    # Data di modifica remota (dal metadato già disponibile nell'elemento)
    data_str = elemento["lastModifiedDateTime"]
    data_remota = datetime.fromisoformat(data_str.replace("Z", "+00:00"))

    # Confronto date se il file esiste già in locale
    if os.path.exists(percorso_locale):
        data_locale = data_modifica_locale(percorso_locale)
        if data_remota <= data_locale:
            print(f"  ⏭️  {nome_file} — saltato (locale più recente o uguale)")
            return "saltato"
        else:
            print(f"  🔄 {nome_file} — aggiornamento (remoto più recente)")
    else:
        print(f"  🆕 {nome_file} — nuovo file")

    # --- Download ---
    cartella_dest = os.path.dirname(percorso_locale)
    os.makedirs(cartella_dest, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}

    # Scarica su file temporaneo; lo sposta sulla destinazione solo se completo.
    fd, percorso_tmp = tempfile.mkstemp(dir=cartella_dest)
    try:
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            dimensione_totale = int(r.headers.get("Content-Length", 0))
            scaricati = 0

            with os.fdopen(fd, "wb") as f:
                fd = None  # os.fdopen si è preso la proprietà del descrittore
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    scaricati += len(chunk)
                    if dimensione_totale:
                        percentuale = int((scaricati / dimensione_totale) * 100)
                        print(f"  ⬇️  {nome_file}: {percentuale}%", end="\r")

        os.replace(percorso_tmp, percorso_locale)
        percorso_tmp = None  # segnala che il file è già stato spostato
    finally:
        if percorso_tmp and os.path.exists(percorso_tmp):
            os.remove(percorso_tmp)  # pulizia in caso di errore
        elif fd is not None:
            os.close(fd)

    print(f"  ✅ {nome_file}          ")
    return "scaricato"


def scarica_cartella(token, cartella_remota, cartella_locale):
    """Scarica ricorsivamente tutti i file di una cartella da OneDrive."""
    os.makedirs(cartella_locale, exist_ok=True)

    print(f"\n📂 Download da OneDrive: {cartella_remota}")
    print(f"   → Destinazione locale: {cartella_locale}\n")

    scaricati = saltati = errori = 0

    def _scarica_ricorsivo(percorso_remoto, percorso_locale):
        nonlocal scaricati, saltati, errori
        elementi = lista_contenuto(token, percorso_remoto)

        for elemento in elementi:
            nome = elemento["name"]
            dest_locale = os.path.join(percorso_locale, nome)

            if "folder" in elemento:
                print(f"\n📁 Sottocartella: {percorso_remoto}/{nome}/")
                os.makedirs(dest_locale, exist_ok=True)
                _scarica_ricorsivo(f"{percorso_remoto}/{nome}", dest_locale)

            elif "file" in elemento:
                try:
                    esito = scarica_file(token, elemento, dest_locale)
                    if esito == "scaricato":
                        scaricati += 1
                    elif esito == "saltato":
                        saltati += 1
                    else:
                        errori += 1
                except Exception as e:
                    print(f"  ❌ Errore su {nome}: {e}")
                    errori += 1

    _scarica_ricorsivo(cartella_remota, cartella_locale)

    print(f"\n{'='*50}")
    print(f"🆕 Nuovi/aggiornati: {scaricati}")
    print(f"⏭️  Saltati (già aggiornati): {saltati}")
    print(f"❌ Errori: {errori}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    token = autenticati()
    scarica_cartella(token, CARTELLA_REMOTA, CARTELLA_LOCALE)
