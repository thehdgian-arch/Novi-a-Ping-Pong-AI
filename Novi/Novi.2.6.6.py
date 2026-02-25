import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import math

# ---------------- HILFS-FARBE 0-255 → 0-1 -----------------
def c(*rgb): return tuple(v/255 for v in rgb)

# --------------- ORDNER-LOGIK ------------------------------------
def get_save_path():
    base = "saves"
    os.makedirs(base, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path)
    return path

SAVE_DIR = get_save_path()
print("Speichere in", SAVE_DIR)

# --------------- START-MENÜ --------------------------------------
def menu_choice():
    root = tk.Tk()
    root.title("Pong Setup")
    root.geometry("350x400") # Etwas größer für mehr Optionen
    root.resizable(False, False)
    
    # Defaults
    choice = {"action": None, "file": None, "fields": 1, "mode": "bot"}

    var_fields = tk.IntVar(value=1)
    var_mode = tk.StringVar(value="bot")

    def set_vals():
        choice["fields"] = var_fields.get()
        choice["mode"] = var_mode.get()

    def neu():
        set_vals()
        choice["action"] = "neu"
        root.destroy()

    def weiter():
        set_vals()
        f = filedialog.askopenfilename(
            title="Modell wählen (NN .pth)",
            filetypes=[("PyTorch", "*.pth")])
        if f:
            choice["action"] = "weiter"
            choice["file"] = f
            root.destroy()
        else:
            messagebox.showwarning("Keine Datei", "Bitte eine .pth-Datei wählen.")

    def spielen():
        set_vals()
        f = filedialog.askopenfilename(
            title="Modell wählen (nur spielen)",
            filetypes=[("PyTorch", "*.pth")])
        if f:
            choice["action"] = "spielen"
            choice["file"] = f
            root.destroy()
        else:
            messagebox.showwarning("Keine Datei", "Bitte eine .pth-Datei wählen.")
   
    # --- GUI Elemente ---
    tk.Label(root, text="--- Einstellungen ---", font=("Arial", 12, "bold")).pack(pady=5)
    
    # 1. Anzahl Felder
    tk.Label(root, text="Anzahl der Spielfelder:").pack(pady=2)
    frame_fields = tk.Frame(root)
    frame_fields.pack()
    tk.Radiobutton(frame_fields, text="1 Spiel", variable=var_fields, value=1).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_fields, text="2 Spiele", variable=var_fields, value=2).pack(side=tk.LEFT, padx=10)

    # 2. Spielmodus (NEU)
    tk.Label(root, text="Gegner-Modus:").pack(pady=2)
    frame_mode = tk.Frame(root)
    frame_mode.pack()
    tk.Radiobutton(frame_mode, text="NN vs. Bot (Standard)", variable=var_mode, value="bot").pack(anchor=tk.W)
    tk.Radiobutton(frame_mode, text="NN vs. NN (Training für beide)", variable=var_mode, value="nn").pack(anchor=tk.W)

    tk.Label(root, text="-----------------------").pack(pady=5)

    # Buttons
    tk.Button(root, text="Neu lernen (Reset)", width=30, command=neu).pack(pady=5)
    tk.Button(root, text="Weiter lernen (Load)", width=30, command=weiter).pack(pady=5)
    tk.Button(root, text="Nur spielen (Kein Training)", width=30, command=spielen).pack(pady=5)

    root.mainloop()
    return choice

CHOICE = menu_choice()
if CHOICE["action"] is None:
    print("Abbruch – nichts gewählt.")
    exit()

NUM_FIELDS = CHOICE["fields"]
GAME_MODE = CHOICE["mode"] # "bot" oder "nn"

# --------------- PYGAME-SETUP ------------------------------------
pygame.init()
FIELD_W, FIELD_H = 900, 300
MARGIN = 20
SCREEN_W = FIELD_W + 350
SCREEN_H = NUM_FIELDS * FIELD_H + (NUM_FIELDS + 1) * MARGIN
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption(f"Pong: {GAME_MODE.upper()}-Modus | {NUM_FIELDS} Feld(er)")

font = pygame.font.SysFont("Arial", 18, bold=False)

clock = pygame.time.Clock()

# --------------- HILFS-FUNKTIONEN --------------------------------
def new_ball():
    speed = random.uniform(4.5, 6.5)
    angle = random.uniform(-0.9, 0.9)
    direction = random.choice([-1, 1])
    vx = math.cos(angle) * speed * direction
    vy = math.sin(angle) * speed
    return FIELD_W // 2, FIELD_H // 2, vx, vy

def clamp(v, mi, ma):
    return max(mi, min(ma, v))

# --------------- NEURAL NET --------------------------------------
class PongNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# --------------- GAME INSTANCE -----------------------------------
PAD_W, PAD_H = 10, 80
BALL_W = BALL_H = 16
GRAPH_W, GRAPH_H = 250, 90

class GameInstance:
    def __init__(self, y_offset, net_file=None, train_enable=True, mode="bot",
                 paddle_nn_color=(0, 255, 10),
                 paddle_bot_color=(0, 180, 255),
                 x_color=(255, 255, 255)):
        self.y0 = y_offset
        self.train_enable = train_enable
        self.mode = mode  # "bot" oder "nn"

        # Spiellogik
        self.p_y = FIELD_H // 2 - PAD_H // 2
        self.ai_y = FIELD_H // 2 - PAD_H // 2
        self.b_x, self.b_y, self.b_vx, self.b_vy = new_ball()
        self.p_score = 0
        self.ai_score = 0
        self.hit = 0
        self.miss = 0

        # Historie
        self.rate_hist = [0.0]
        self.nn_sc_hist = [0]
        self.ai_sc_hist = [0]

        # NN Setup
        random.seed(None)
        torch.manual_seed(random.randint(0, 1_000_000))

        # --- Linkes NN (NN1) ---
        self.net = PongNet()
        self.opt = optim.Adam(self.net.parameters(), lr=0.005)
        
        # --- Rechtes NN (NN2) - nur wenn Mode == 'nn' ---
        self.net2 = None
        self.opt2 = None
        if self.mode == "nn":
            self.net2 = PongNet()
            self.opt2 = optim.Adam(self.net2.parameters(), lr=0.005)

        # Laden
        if net_file and os.path.exists(net_file):
            try:
                state = torch.load(net_file, map_location="cpu")
                self.net.load_state_dict(state)
                print(f"Modell 1 geladen: {net_file}")
                # Im NN vs NN Modus starten beide mit dem gleichen Netz (Klonen)
                if self.mode == "nn":
                    self.net2.load_state_dict(state)
                    print(f"Modell 2 (Kopie) geladen.")
            except Exception as e:
                print("Fehler beim Laden:", e)

        # Farben
        self.paddle_nn_color = paddle_nn_color
        self.paddle_bot_color = paddle_bot_color # Wird für NN2 genutzt im NN-Modus
        self.x_color = x_color

    # -------------------- SPIEL-LOGIK -----------------
    def step(self):
        
        # x
        self.b_x += self.b_vx
        self.b_y += self.b_vy

        if self.b_y <= 0 or self.b_y + BALL_H >= FIELD_H:
            self.b_vy *= -1
            self.b_vy += random.uniform(-0.8, 0.8)

        # Links (NN 1)
        if self.b_x <= PAD_W:
            if self.p_y < self.b_y + BALL_H // 2 < self.p_y + PAD_H:
                angle_noise = random.uniform(-0.5, 0.5)
                self.b_vx = abs(self.b_vx) * 1.05
                self.b_vy += angle_noise
                self.hit += 1
            else:
                self.ai_score += 1
                self.miss += 1
                self.b_x, self.b_y, self.b_vx, self.b_vy = new_ball()
                self.b_vx = 5

        # Rechts (Bot ODER NN 2)
        if self.b_x + BALL_W >= FIELD_W - PAD_W:
            if self.ai_y < self.b_y + BALL_H // 2 < self.ai_y + PAD_H:
                angle_noise = random.uniform(-0.5, 0.5)
                self.b_vx = -abs(self.b_vx)
                self.b_vy += angle_noise
            else:
                self.p_score += 1
                self.b_x, self.b_y, self.b_vx, self.b_vy = new_ball()
                self.b_vx = -5

        # --- Steuerung NN 1 (Links) ---
        st = torch.tensor([self.b_x / FIELD_W,
                           self.b_y / FIELD_H,
                           self.b_vx / 10,
                           self.b_vy / 10,
                           self.p_y / FIELD_H], dtype=torch.float32)
        mv = self.net(st)

        if self.train_enable:
            tgt = 0.0
            if (self.b_y + BALL_H // 2) < (self.p_y + PAD_H // 2) - 10: tgt = -1.0
            if (self.b_y + BALL_H // 2) > (self.p_y + PAD_H // 2) + 10: tgt = 1.0
            self.opt.zero_grad()
            nn.MSELoss()(mv, torch.tensor([tgt])).backward()
            self.opt.step()

        self.p_y = clamp(self.p_y + mv.item() * 12, 0, FIELD_H - PAD_H)

        # --- Steuerung RECHTS: Modus-Entscheidung ---
        if self.mode == "bot":
            # ALTER BOT-CODE
            speed = 2.2 # Langsamer Bot
            if random.random() < 0.10: speed *= -1
            if self.b_y + BALL_H // 2 > self.ai_y + PAD_H // 2:
                self.ai_y += speed
            else:
                self.ai_y -= speed
        
        elif self.mode == "nn":
            # NEUER NN VS NN CODE (Rechtes Netz)
            # Wir füttern die Daten. Für das rechte Netz ist es nützlich,
            # die Position auch normalisiert zu haben.
            st2 = torch.tensor([self.b_x / FIELD_W,
                                self.b_y / FIELD_H,
                                self.b_vx / 10,
                                self.b_vy / 10,
                                self.ai_y / FIELD_H], dtype=torch.float32)
            mv2 = self.net2(st2)

            if self.train_enable:
                # Zielwert berechnen (Training nach Ballposition)
                tgt2 = 0.0
                if (self.b_y + BALL_H // 2) < (self.ai_y + PAD_H // 2) - 10: tgt2 = -1.0
                if (self.b_y + BALL_H // 2) > (self.ai_y + PAD_H // 2) + 10: tgt2 = 1.0
                self.opt2.zero_grad()
                nn.MSELoss()(mv2, torch.tensor([tgt2])).backward()
                self.opt2.step()
            
            # Bewegung ausführen
            self.ai_y = clamp(self.ai_y + mv2.item() * 12, 0, FIELD_H - PAD_H)

    # -------------------- HISTORIE -----------------
    def record_history(self):
        total = self.hit + self.miss
        self.rate_hist.append(self.hit / total if total else 0)
        self.nn_sc_hist.append(self.p_score)
        self.ai_sc_hist.append(self.ai_score)

    # -------------------- ZEICHNEN (angepasst) ------------
    def draw(self, surf):
        # Feld-Hintergrund
        pygame.draw.rect(surf, (20, 20, 20), (0, self.y0, FIELD_W, FIELD_H))
        # Mittellinie
        pygame.draw.line(surf, (120, 120, 120),
                         (FIELD_W // 2, self.y0),
                         (FIELD_W // 2, self.y0 + FIELD_H), 2)
        # Schläger
        pygame.draw.rect(surf, self.paddle_nn_color,
                         (0, int(self.y0 + self.p_y), PAD_W, PAD_H))
        pygame.draw.rect(surf, self.paddle_bot_color,
                         (FIELD_W - PAD_W, int(self.y0 + self.ai_y), PAD_W, PAD_H))
        # BALL
        pygame.draw.ellipse(
                        surf,
                        self.x_color,   
                        (int(self.b_x), int(self.y0 + self.b_y), BALL_W, BALL_H)
        )
        
        # TEXT ANPASSUNG JE NACH MODUS
        label_left = "NN 1 "
        label_right = "Bot " if self.mode == "bot" else "NN 2 "

        # Scores
        surf.blit(font.render(f"{label_left}: {self.p_score}", True, self.paddle_nn_color),
                  (10, self.y0 + 10))
        surf.blit(font.render(f"{label_right}: {self.ai_score}", True, self.paddle_bot_color),
                  (FIELD_W - 100, self.y0 + 10))

# --------------- GRAPHEN ZEICHEN -------------------------------
# --------------- GRAPHEN ZEICHEN -------------------------------
def draw_graphs(surface, instances):
    x0 = FIELD_W + MARGIN
    y = MARGIN
    colors_rate = [(0, 0, 255), (255, 0, 0)]          # Treffer-Raten
    colors_score = [(0, 255, 0), (150, 150, 15),     # Feld 1: NN1, NN2
                    (255, 15, 255), (0, 10, 100)]    # Feld 2: NN1, NN2

    for idx, gi in enumerate(instances):
        label_opp = "Bot" if gi.mode == "bot" else "NN2"
        
        def draw(data_lists, colors, title, yy):
            pygame.draw.rect(surface, (40, 40, 40), (x0, yy, GRAPH_W, GRAPH_H))
            pygame.draw.rect(surface, (100, 100, 100), (x0, yy, GRAPH_W, GRAPH_H), 1)
            surface.blit(font.render(title, True, (200, 200, 200)), (x0, yy - 20))
            all_vals = [v for sub in data_lists for v in sub[-60:]]
            max_v = max(all_vals) if all_vals and max(all_vals) > 0 else 1
            for d_idx, data in enumerate(data_lists):
                pts = []
                for i, v in enumerate(data[-60:]):
                    px = x0 + i * (GRAPH_W / (len(data[-60:]) - 1)) if len(data[-60:]) > 1 else x0
                    py = yy + GRAPH_H - v / max_v * GRAPH_H
                    pts.append((px, py))
                if len(pts) > 1:
                    pygame.draw.lines(surface, colors[d_idx], False, pts, 2)

        draw([gi.rate_hist], [colors_rate[idx]], f"Feld{idx+1} Treffer (Links)", y)
        y += GRAPH_H + MARGIN
        draw([gi.nn_sc_hist, gi.ai_sc_hist], [colors_score[idx*2], colors_score[idx*2+1]], f"Feld{idx+1}: NN1 vs {label_opp}", y)
        y += GRAPH_H + MARGIN

    # Vergleichsgraph immer
    # Farben für bis zu 4 Kurven
    cmp_colors = [(0,255,0),(150,150,15),(255,15,255),(0,10,100)]
    cmp_data = []
    cmp_labels = []

    for idx, gi in enumerate(instances):
        cmp_data.append(gi.nn_sc_hist)       # Linkes NN
        cmp_labels.append(f"F{idx+1} NN1")
        if gi.mode == "nn" and gi.net2:
            cmp_data.append(gi.ai_sc_hist)   # Rechtes NN nur im NN-vs-NN
            cmp_labels.append(f"F{idx+1} NN2")

    draw(cmp_data, cmp_colors[:len(cmp_data)], "Vergleich alle Netze", y)


# --------------- INSTANZEN ERSTELLEN -----------------------------
colors_f1 = {
    "nn":   (0, 255, 100),
    "bot":  (150, 150, 150),
    "x": (255, 255, 255)
}
colors_f2 = {
    "nn":   (255, 255, 0),
    "bot":  (100, 100, 100),
    "x": (0, 255, 255)
}

instances = []
y_off = MARGIN

# Feld 1
# Parameter 'mode' hinzugefügt
if CHOICE["action"] == "neu":
    gi = GameInstance(y_off, train_enable=True, mode=GAME_MODE,
                      paddle_nn_color=colors_f1["nn"],
                      paddle_bot_color=colors_f1["bot"],
                      x_color=colors_f1["x"])
elif CHOICE["action"] == "weiter":
    gi = GameInstance(y_off, net_file=CHOICE["file"], train_enable=True, mode=GAME_MODE,
                      paddle_nn_color=colors_f1["nn"],
                      paddle_bot_color=colors_f1["bot"],
                      x_color=colors_f1["x"])
elif CHOICE["action"] == "spielen":
    gi = GameInstance(y_off, net_file=CHOICE["file"], train_enable=False, mode=GAME_MODE,
                      paddle_nn_color=colors_f1["nn"],
                      paddle_bot_color=colors_f1["bot"],
                      x_color=colors_f1["x"])
instances.append(gi)

# Feld 2
if NUM_FIELDS == 2:
    y_off = 2 * MARGIN + FIELD_H
    if CHOICE["action"] == "neu":
        gi = GameInstance(y_off, train_enable=True, mode=GAME_MODE,
                          paddle_nn_color=colors_f2["nn"],
                          paddle_bot_color=colors_f2["bot"],
                          x_color=colors_f2["x"])
    elif CHOICE["action"] == "weiter":
        gi = GameInstance(y_off, net_file=CHOICE["file"], train_enable=True, mode=GAME_MODE,
                          paddle_nn_color=colors_f2["nn"],
                          paddle_bot_color=colors_f2["bot"],
                          x_color=colors_f2["x"])
    elif CHOICE["action"] == "spielen":
        gi = GameInstance(y_off, net_file=CHOICE["file"], train_enable=False, mode=GAME_MODE,
                          paddle_nn_color=colors_f2["nn"],
                          paddle_bot_color=colors_f2["bot"],
                          x_color=colors_f2["x"])
    instances.append(gi)

# --------------- HAUPTSCHLEIFE -----------------------------------
frame_count = 0
running = True
while running:
    clock.tick(60)
    frame_count += 1
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for gi in instances:
        gi.step()
    if frame_count % 60 == 0:
        for gi in instances:
            gi.record_history()

    for gi in instances:
        gi.draw(screen)
    draw_graphs(screen, instances)
    pygame.display.flip()

# --------------- SPEICHERN + DEBUG ---------------------------------------
pygame.quit()

print("SAVE_DIR existiert:", os.path.exists(SAVE_DIR))
print("SAVE_DIR ist:", SAVE_DIR)

def c(*rgb): return tuple(v/255 for v in rgb)
colors_rate = [c(0, 255, 0), c(255, 255, 180)]
colors_score = [c(50, 255, 50), c(0, 180, 255), c(255, 0, 0), c(0, 0, 0)]

for idx, gi in enumerate(instances):
    # --- Rate ---
    plt.figure(figsize=(8, 5))
    plt.plot(gi.rate_hist, color=colors_rate[idx])
    plt.title(f"Treffe Feld{idx+1} (Links)")
    plt.grid(True)
    fname_rate = os.path.join(SAVE_DIR, f"f{idx+1}_rate.png")
    plt.savefig(fname_rate, format='png', dpi=150, facecolor='white')
    plt.close()

    # --- Scores ---
    plt.figure(figsize=(8, 5))
    label_opp = "Bot" if gi.mode == "bot" else "NN2"
    plt.plot(gi.nn_sc_hist, label="NN1", color=colors_score[idx*2])
    plt.plot(gi.ai_sc_hist, label=label_opp, color=colors_score[idx*2+1])
    plt.title(f"Scores Feld{idx+1}")
    plt.legend(); plt.grid(True)
    fname_score = os.path.join(SAVE_DIR, f"f{idx+1}_scores.png")
    plt.savefig(fname_score, format='png', dpi=150, facecolor='white')
    plt.close()

    # --- Modell ---
    # Speichere das Linke Modell
    fname_model = os.path.join(SAVE_DIR, f"nn{idx+1}_left.pth")
    torch.save(gi.net.state_dict(), fname_model)
    print("Links gespeichert:", fname_model)
    
    # Speichere das Rechte Modell (falls existent)
    if gi.mode == "nn" and gi.net2:
        fname_model2 = os.path.join(SAVE_DIR, f"nn{idx+1}_right.pth")
        torch.save(gi.net2.state_dict(), fname_model2)
        print("Rechts gespeichert:", fname_model2)


# ---------------- VERGLEICHSGRAPH SPEICHERN --------------------
# Vergleich aller Netze (NN1 und NN2 falls NN-vs-NN)
if len(instances) > 1:
    plt.figure(figsize=(8,5))
    cmp_colors = ['green','darkgreen','magenta','purple']
    cmp_labels = []
    cmp_data = []

    for idx, gi in enumerate(instances):
        cmp_data.append(gi.nn_sc_hist)
        cmp_labels.append(f"F{idx+1} NN1")
        if gi.mode == "nn" and gi.net2:
            cmp_data.append(gi.ai_sc_hist)
            cmp_labels.append(f"F{idx+1} NN2")

    for d, lbl, col in zip(cmp_data, cmp_labels, cmp_colors):
        plt.plot(d, label=lbl, color=col)

    plt.title("Vergleich aller Netze")
    plt.legend(); plt.grid(True)
    fname_cmp = os.path.join(SAVE_DIR, "vergleich_alle_netze.png")
    plt.savefig(fname_cmp, format='png', dpi=150, facecolor='white')
    plt.close()
    print("Vergleichsgraf gespeichert:", fname_cmp)


print("Fertig.")