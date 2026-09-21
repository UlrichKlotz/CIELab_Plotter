# CIELab Plotter – Streamlit Web-Version

Diese Version stellt den CIELab–sRGB Gamut Plotter als Web-Anwendung mit
Streamlit bereit – ohne lokale Python-Installation für die Nutzer.

**Autor:** Prof. Dr. Ulrich E. Klotz · Hochschule München University of Applied Sciences

**Empfohlene Zitierung:**

> Klotz, U. E. (2026). CIELab–sRGB Gamut Plotter [Computer software].
> Hochschule München University of Applied Sciences.
> https://doi.org/10.5281/zenodo.22811675

---

## Variante A: Auf Streamlit Community Cloud veröffentlichen (empfohlen)

So wird die Seite öffentlich im Browser erreichbar, ganz ohne dass die Nutzer
etwas installieren müssen.

1. **GitHub-Repository vorbereiten.** Lege die drei Dateien `app.py`,
   `requirements.txt` und diese `README_streamlit.md` in ein **öffentliches**
   GitHub-Repository (das bestehende Repository kann verwendet werden).
2. **Bei Streamlit anmelden.** Öffne <https://share.streamlit.io> und melde
   dich mit deinem GitHub-Konto an.
3. **Neue App anlegen.** Klicke auf **„Create app“** → **„Deploy a public app
   from GitHub“**.
4. **Repository, Branch und Datei wählen.**
   - Repository: dein GitHub-Repository
   - Branch: `main` (oder der gewünschte Branch)
   - Main file path: `app.py`
5. **Deploy** klicken. Streamlit installiert automatisch die Pakete aus
   `requirements.txt` und startet die App. Nach kurzer Zeit erhältst du eine
   dauerhafte öffentliche URL (z. B. `https://<name>.streamlit.app`), die du
   teilen oder verlinken kannst.

Änderungen am Code werden nach jedem `git push` automatisch übernommen und die
App neu gestartet.

## Variante B: Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Danach die im Terminal angezeigte lokale URL (standardmäßig
<http://localhost:8501>) im Browser öffnen.

---

## Funktionen

- Messdaten per Copy/Paste importieren
- CSV/TSV/TXT-Dateien hochladen
- Tab-, Semikolon- und Komma-getrennte Daten
- Daten direkt in einer Tabelle bearbeiten
- L*-Ebene und a*/b*-Bereiche einstellen
- Datenpunkte ein-/ausblenden
- Phasenbezeichnungen ein-/ausblenden
- PNG-Export mit weißem, publikationsgeeignetem Hintergrund
- DOI-/Quellenangabe dauerhaft im exportierten Diagramm (unter der Legende)

Die Web-Version bietet bewusst keine Session-Speicherung. Diese Funktion bleibt
der Desktop-/Python-Version vorbehalten.

## Leistung / Reaktionsverhalten

Diagramme werden über `st.cache_data` zwischengespeichert. Solange sich Daten
und Einstellungen nicht ändern, wird die Grafik nicht neu berechnet, wodurch die
Oberfläche bei Interaktionen deutlich flüssiger reagiert.

## Quelle

CIELab–sRGB Gamut Plotter
https://github.com/UlrichKlotz/CIELab_Plotter

DOI: https://doi.org/10.5281/zenodo.22811675

Lizenz: GPL-3.0-or-later
