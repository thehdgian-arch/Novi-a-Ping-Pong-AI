# 🎮 NOVI – Pong Neural Network Trainer (CH/DE)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Private-lightgrey.svg)]()

> Ein interaktiver Pong-Trainer mit neuronalem Netzwerk (PyTorch) – perfekt zum Experimentieren mit Reinforcement Learning und KI-Visualisierung.

![NOVI Demo](saves/demo_screenshot.png) <!-- Optional: Füge einen Screenshot ein -->

---

## ✨ Features

| Feature | Beschreibung |
|---------|-------------|
| 🧠 **Training** | Neues NN trainieren oder bestehendes Modell weitertrainieren |
| 🎮 **Spielmodus** | Trainiertes Netzwerk spielen lassen (ohne Training) |
| 🖥️ **Multi-View** | 1 oder 2 Spielfelder parallel für direkte Vergleiche |
| 📊 **Live-Graphen** | Echtzeit-Visualisierung von Trefferquote & Scores |
| 💾 **Auto-Save** | Modelle und Diagramme werden automatisch gespeichert |

---

## 🚀 Schnellstart

### Voraussetzungen

- **Python 3.9** oder neuer
- **tkinter** (meist bei Python dabei)

### Installation

```bash
# Abhängigkeiten installieren
pip install pygame torch matplotlib
🍎 macOS-Hinweis: Falls tkinter fehlt, installiere Python über python.org, nicht über Homebrew.
Starten
bash
Copy
python Novi.py
(Der Dateiname kann bei dir leicht abweichen)
🎯 Bedienung
Start-Menü
Table
Option	Beschreibung
1️⃣ Anzahl Spielfelder	1 Spiel → NN vs. Bot | 2 Spiele → Zwei NNs parallel
🆕 Neu lernen	Neues neuronales Netz von Grund auf trainieren
🔁 Weiter lernen	Bestehende .pth-Datei laden und weitertrainieren
🎮 Nur spielen	Training deaktiviert – NN spielt nur
Steuerung
❌ Keine Tastatur nötig – alles läuft automatisch
✅ Beenden: Fenster schließen
🧠 Wie es funktioniert
Spiellogik
Table
Element	Beschreibung
Links	Neuronales Netz (NN)
Rechts	Bot (absichtlich etwas schlechter)
NN-Input (was das Netz "sieht")
🏐 Ballposition
⚡ Ballgeschwindigkeit
🏓 Eigene Paddle-Position
NN-Output (Entscheidungen)
⬆️ Hoch
⬇️ Runter
⏹️ Stehen bleiben
Belohnungssystem
Table
Ereignis	Bewertung
✅ Ball getroffen	Gut
❌ Ball vorbeigeflogen	Schlecht
📊 Graphen & Visualisierung
Table
Graph	Bedeutung
Trefferquote	Wie gut das NN den Ball trifft
Scores	Punkte: NN vs. Bot
Vergleich (nur bei 2 Feldern)	Direkter Vergleich NN1 vs. NN2
💾 Speicherstruktur
Beim Start wird automatisch ein Zeitstempel-Ordner erstellt:
plain
Copy
saves/
└── 2026-01-30_14-22-08/
    ├── nn1.pth              # Trainiertes Modell Feld 1
    ├── f1_rate.png          # Trefferquote Feld 1
    ├── f1_scores.png        # Punktestand Feld 1
    ├── nn2.pth              # Trainiertes Modell Feld 2 (nur bei 2 Feldern)
    ├── f2_rate.png          # Trefferquote Feld 2
    ├── f2_scores.png        # Punktestand Feld 2
    └── nn_compare.png       # Vergleich beider NNs
Table
Dateiendung	Bedeutung
.pth	Trainiertes PyTorch-Modell
*_rate.png	Trefferquote-Diagramm
*_scores.png	Punktestand-Diagramm
nn_compare.png	Vergleichsdiagramm
🎨 Farben anpassen
Die Farben pro Spielfeld kannst du direkt im Code ändern:
Python
Copy
colors_f1 = {
    "nn": (0, 255, 100),      # NN-Farbe (RGB)
    "bot": (150, 150, 150),   # Bot-Farbe
    "x": (255, 255, 255)      # Zusätzliche Elemente
}
RGB-Werte von 0–255
🛠️ Fehlerbehebung
<details>
<summary><b>❌ ModuleNotFoundError: pygame</b></summary>
bash
Copy
pip install pygame
</details>
<details>
<summary><b>❌ Tkinter fehlt (macOS)</b></summary>
Python von python.org installieren (nicht Homebrew)
</details>
<details>
<summary><b>❌ Modell lädt nicht</b></summary>
Datei muss eine .pth-Datei sein
</details>
🎯 Über das Projekt
Idee & Zweck
NOVI ist ein Lern- und Experimentierprojekt für:
🧠 Neuronale Netze verstehen
🔄 Reinforcement-ähnliches Lernen
🎮 Game + KI Kombination
📈 Training-Visualisierung
⚠️ Perfekt zum Verstehen, nicht für maximale Performance.
📄 Lizenz
plain
Copy
Privates Lernprojekt.
Keine Garantie, keine Haftung.
Viel Spaß beim Trainieren! 🚀
PS: Diese README ist auch als PDF im NOVI-Ordner verfügbar.
plain
Copy

---

## Was ich verbessert habe:

| Aspekt | Änderung |
|--------|----------|
| **Struktur** | Klare Hierarchie mit Überschriften (H1, H2, H3) |
| **Badges** | Python-Version, PyTorch, Lizenz als Shield-Badges |
| **Tabellen** | Bessere Lesbarkeit für Features, Steuerung, Dateien |
| **Code-Blöcke** | Syntax-Highlighting für Installation & Code-Beispiele |
| **Emojis** | Visuelle Orientierung (nicht überladen) |
| **Details/Summary** | Kollapsible Fehlerbehebung für Übersichtlichkeit |
| **Tree-View** | Ordnerstruktur als Code-Block mit Kommentaren |
| **Callouts** | Wichtige Hinweise mit >-Blockquotes hervorgehoben |
