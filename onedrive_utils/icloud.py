#!/usr/bin/env python3
"""
icloud_sync.py — Sincronizza una cartella tra iCloud Drive e il filesystem locale
usando pyicloud, con gestione del 2FA e salto dei file già aggiornati (confronto
per dimensione e data di modifica).

Uso:
    pip install pyicloud

    export ICLOUD_EMAIL="tua@email.com"
    export ICLOUD_PASSWORD="password"      # opzionale: se assente viene chiesta

    # Scarica da iCloud in locale
    python icloud_sync.py download --remote "MiaCartella" --local "./MiaCartella"

    # Carica dal locale su iCloud
    python icloud_sync.py upload --remote "MiaCartella" --local "./MiaCartella"

    # Prova a vuoto, senza trasferire nulla
    python icloud_sync.py download --remote "MiaCartella" --local "./MiaCartella" --dry-run

Note:
  - pyicloud usa un'API iCloud NON ufficiale: può rompersi se Apple cambia qualcosa.
  - Il 2FA di Apple non è aggirabile. Su una macchina headless prevedi di dover
    reinserire un codice ogni tanto, quando la sessione fidata scade.
  - La cartella dei cookie (--cookies) conserva la sessione tra le esecuzioni,
    così non devi rifare il 2FA a ogni avvio finché la sessione resta valida.
"""

import argparse
import getpass
import logging
import os
import sys
import time
from datetime import timezone
from pathlib import Path

try:
    from pyicloud import PyiCloudService
    from pyicloud.exceptions import PyiCloudFailedLoginException
except ImportError:
    sys.exit("pyicloud non installato. Esegui: pip install pyicloud")

log = logging.getLogger("icloud_sync")


# ----------------------------------------------------------------------------
# Autenticazione
# ----------------------------------------------------------------------------
def autentica(email, password, cookie_dir):
    """Login su iCloud con gestione 2FA/2SA. Ritorna l'istanza PyiCloudService."""
    try:
        api = PyiCloudService(email, password, cookie_directory=cookie_dir)
    except PyiCloudFailedLoginException as e:
        sys.exit(f"Login fallito: {e}")

    # Autenticazione a due fattori (metodo moderno)
    if api.requires_2fa:
        log.info("Richiesto codice 2FA.")
        code = input("Inserisci il codice 2FA ricevuto sui dispositivi Apple: ").strip()
        if not api.validate_2fa_code(code):
            sys.exit("Codice 2FA non valido.")
        if not api.is_trusted_session:
            log.info("Registro la sessione come fidata...")
            api.trust_session()

    # Verifica in due passaggi (metodo più vecchio, presente su alcuni account)
    elif api.requires_2sa:
        log.info("Richiesta verifica in due passaggi (2SA).")
        devices = api.trusted_devices
        for i, d in enumerate(devices):
            etichetta = d.get("deviceName") or ("SMS a " + d.get("phoneNumber", "?"))
            print(f"  {i}: {etichetta}")
        idx = int(input("Scegli il dispositivo per ricevere il codice: ").strip())
        device = devices[idx]
        if not api.send_verification_code(device):
            sys.exit("Invio del codice fallito.")
        code = input("Inserisci il codice ricevuto: ").strip()
        if not api.validate_verification_code(device, code):
            sys.exit("Codice di verifica non valido.")

    return api


# ----------------------------------------------------------------------------
# Utilità di navigazione e metadati
# ----------------------------------------------------------------------------
def naviga(api, percorso):
    """Naviga fino al nodo iCloud indicato da un percorso tipo 'A/B/C'."""
    nodo = api.drive
    for parte in [p for p in percorso.strip("/").split("/") if p]:
        nodo = nodo[parte]
    return nodo


def _figlio_dopo_mkdir(nodo, nome, tentativi=5, attesa=1.0):
    """Dopo un mkdir, restituisce il nodo della cartella appena creata.

    get_children(force=True) rilegge i dati DAL SERVER e aggiorna nodo.data: è il
    punto cruciale, perché il semplice azzeramento della cache non basterebbe —
    pyicloud ricostruirebbe l'elenco dei figli dai dati grezzi vecchi, ancora
    privi della cartella nuova. La piccola ripetizione assorbe eventuali ritardi
    di propagazione lato iCloud."""
    for _ in range(tentativi):
        nodo.get_children(force=True)
        try:
            return nodo[nome]
        except KeyError:
            time.sleep(attesa)
    raise RuntimeError(f"Cartella remota '{nome}' non trovata dopo la creazione.")


def _remote_mtime(nodo):
    """Timestamp UNIX della data di modifica del nodo remoto, o None."""
    dm = getattr(nodo, "date_modified", None)
    if dm is None:
        return None
    if dm.tzinfo is None:            # pyicloud restituisce di solito UTC "naive"
        dm = dm.replace(tzinfo=timezone.utc)
    return dm.timestamp()


# ----------------------------------------------------------------------------
# DOWNLOAD: iCloud -> locale
# ----------------------------------------------------------------------------
def _serve_scaricare(nodo, locale):
    """True se il file remoto va scaricato (assente o diverso in locale)."""
    if not locale.exists():
        return True
    st = locale.stat()
    if nodo.size is not None and st.st_size != nodo.size:
        return True
    rmt = _remote_mtime(nodo)
    if rmt is not None and st.st_mtime + 2 < rmt:    # tolleranza di 2 secondi
        return True
    return False


def scarica_cartella(nodo, locale, dry_run, stats):
    locale = Path(locale)
    locale.mkdir(parents=True, exist_ok=True)
    for nome in (nodo.dir() or []):
        item = nodo[nome]
        dest = locale / nome
        if item.type == "folder":
            scarica_cartella(item, dest, dry_run, stats)
        else:
            if not _serve_scaricare(item, dest):
                log.info("=  salto (aggiornato): %s", dest)
                stats["saltati"] += 1
                continue
            log.info("v  scarico: %s", dest)
            stats["trasferiti"] += 1
            if dry_run:
                continue
            # iter_content rispetta il Content-Encoding (es. gzip sui JSON/testo):
            # leggere resp.raw direttamente salterebbe la decodifica e salverebbe
            # file vuoti o corrotti per i contenuti compressi dal server.
            with item.open(stream=True) as resp:
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            # allinea la data locale a quella remota: rende stabili i confronti futuri
            rmt = _remote_mtime(item)
            if rmt is not None:
                os.utime(dest, (rmt, rmt))


# ----------------------------------------------------------------------------
# UPLOAD: locale -> iCloud
# ----------------------------------------------------------------------------
def _serve_caricare(locale, figlio_remoto):
    """True se il file locale va caricato. Euristica: carica se il file remoto
    manca o ha dimensione diversa. pyicloud non offre un confronto per data
    affidabile in upload, quindi ci basiamo sulla dimensione."""
    if figlio_remoto is None or figlio_remoto.type == "folder":
        return True
    if figlio_remoto.size is not None and figlio_remoto.size == locale.stat().st_size:
        return False
    return True


def carica_cartella(locale, nodo, dry_run, stats):
    locale = Path(locale)
    esistenti = {}
    for nome in (nodo.dir() or []):
        esistenti[nome] = nodo[nome]

    for entry in sorted(locale.iterdir()):
        if entry.is_dir():
            if entry.name in esistenti:
                carica_cartella(entry, nodo[entry.name], dry_run, stats)
            else:
                log.info("+  creo cartella remota: %s", entry.name)
                if dry_run:
                    n = sum(1 for f in entry.rglob("*") if f.is_file())
                    log.info("   (dry-run) %d file dentro '%s' verrebbero caricati", n, entry.name)
                    stats["trasferiti"] += n
                else:
                    nodo.mkdir(entry.name)
                    figlio = _figlio_dopo_mkdir(nodo, entry.name)
                    carica_cartella(entry, figlio, dry_run, stats)
        else:
            figlio = esistenti.get(entry.name)
            if not _serve_caricare(entry, figlio):
                log.info("=  salto (aggiornato): %s", entry)
                stats["saltati"] += 1
                continue
            log.info("^  carico: %s", entry)
            stats["trasferiti"] += 1
            if dry_run:
                continue
            with open(entry, "rb") as fh:
                nodo.upload(fh)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Sincronizza una cartella tra iCloud Drive e il filesystem locale."
    )
    p.add_argument("mode", choices=["download", "upload"], help="Direzione della sincronizzazione.")
    p.add_argument("--remote", required=True, help="Cartella su iCloud, es. 'MiaCartella' o 'A/B'.")
    p.add_argument("--local", required=True, help="Cartella locale.")
    p.add_argument("--cookies", default=os.path.expanduser("~/.pyicloud"),
                   help="Cartella in cui conservare la sessione (default: ~/.pyicloud).")
    p.add_argument("--dry-run", action="store_true", help="Mostra cosa farebbe senza trasferire nulla.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    email = os.environ.get("ICLOUD_EMAIL") or input("Apple ID (email): ").strip()
    password = os.environ.get("ICLOUD_PASSWORD") or getpass.getpass("Password: ")

    Path(args.cookies).mkdir(parents=True, exist_ok=True)
    api = autentica(email, password, args.cookies)

    stats = {"trasferiti": 0, "saltati": 0}

    if args.mode == "download":
        try:
            nodo = naviga(api, args.remote)
        except (KeyError, IndexError):
            sys.exit(f"Cartella remota inesistente: {args.remote}")
        scarica_cartella(nodo, args.local, args.dry_run, stats)
    else:  # upload
        if not Path(args.local).is_dir():
            sys.exit(f"Cartella locale inesistente: {args.local}")
        try:
            nodo = naviga(api, args.remote)
        except (KeyError, IndexError):
            sys.exit(f"Cartella remota inesistente: {args.remote} (creala prima da iCloud).")
        carica_cartella(args.local, nodo, args.dry_run, stats)

    verbo = "sarebbero trasferiti" if args.dry_run else "trasferiti"
    log.info("\nFatto. File %s: %d  |  saltati (già aggiornati): %d",
             verbo, stats["trasferiti"], stats["saltati"])


if __name__ == "__main__":
    main()
