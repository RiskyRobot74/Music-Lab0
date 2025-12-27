import math
import pygame
import numpy as np

# -----------------------------
# Settings
# -----------------------------
SAMPLE_RATE = 44100
CHANNELS = 2  # stereo
BUFFER = 512

WINDOW_W, WINDOW_H = 980, 520
FPS = 60

# Key mapping: a row for white keys, top row for black keys
# A S D F G H J K = C D E F G A B C
# W E   T Y U     = C# D#   F# G# A#
KEYBOARD_MAP = [
    # (pygame_key, semitone_offset_from_C4, is_black, label)
    (pygame.K_a, 0,  False, "A"),
    (pygame.K_w, 1,  True,  "W"),
    (pygame.K_s, 2,  False, "S"),
    (pygame.K_e, 3,  True,  "E"),
    (pygame.K_d, 4,  False, "D"),
    (pygame.K_f, 5,  False, "F"),
    (pygame.K_t, 6,  True,  "T"),
    (pygame.K_g, 7,  False, "G"),
    (pygame.K_y, 8,  True,  "Y"),
    (pygame.K_h, 9,  False, "H"),
    (pygame.K_u, 10, True,  "U"),
    (pygame.K_j, 11, False, "J"),
    (pygame.K_k, 12, False, "K"),
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Base note: C4
C4_FREQ = 261.625565

def semitone_to_freq(semitone_offset: int, octave_shift: int) -> float:
    total = semitone_offset + octave_shift * 12
    return C4_FREQ * (2 ** (total / 12))

# -----------------------------
# Simple synth instruments
# -----------------------------
def adsr_envelope(n: int, attack=0.01, decay=0.08, sustain=0.6, release=0.15) -> np.ndarray:
    a = int(attack * SAMPLE_RATE)
    d = int(decay * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    s = max(0, n - (a + d + r))

    env = np.zeros(n, dtype=np.float32)
    idx = 0

    if a > 0:
        env[idx:idx+a] = np.linspace(0.0, 1.0, a, endpoint=False)
        idx += a
    if d > 0:
        env[idx:idx+d] = np.linspace(1.0, sustain, d, endpoint=False)
        idx += d
    if s > 0:
        env[idx:idx+s] = sustain
        idx += s
    if r > 0:
        start = env[idx-1] if idx > 0 else sustain
        env[idx:idx+r] = np.linspace(start, 0.0, r, endpoint=True)
        idx += r

    if idx < n:
        env[idx:] = 0.0
    return env

def wave_sine(phase):     return np.sin(phase)
def wave_square(phase):   return np.sign(np.sin(phase))
def wave_saw(phase):      return 2.0 * (phase / (2*np.pi) - np.floor(0.5 + phase / (2*np.pi)))
def wave_triangle(phase): return 2.0 * np.abs(wave_saw(phase)) - 1.0

def synth_note(freq: float, duration: float, instrument: str, volume: float) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * freq * t

    if instrument == "Sine":
        sig = wave_sine(phase)
        sig *= adsr_envelope(n, attack=0.01, decay=0.06, sustain=0.75, release=0.18)

    elif instrument == "Square":
        sig = wave_square(phase)
        sig *= adsr_envelope(n, attack=0.005, decay=0.05, sustain=0.55, release=0.12)

    elif instrument == "Triangle":
        sig = wave_triangle(phase)
        sig *= adsr_envelope(n, attack=0.01, decay=0.08, sustain=0.65, release=0.15)

    elif instrument == "Saw":
        sig = wave_saw(phase)
        sig *= adsr_envelope(n, attack=0.01, decay=0.10, sustain=0.5, release=0.18)

    elif instrument == "Organ":
        sig = (0.70 * wave_sine(phase) +
               0.20 * wave_sine(2*phase) +
               0.10 * wave_sine(3*phase))
        sig *= adsr_envelope(n, attack=0.02, decay=0.06, sustain=0.95, release=0.22)

    elif instrument == "Bell":
        sig = (0.70 * wave_sine(phase) +
               0.35 * wave_sine(2.71*phase) +
               0.20 * wave_sine(5.18*phase))
        sig *= adsr_envelope(n, attack=0.002, decay=0.20, sustain=0.12, release=0.35)

    elif instrument == "Piano-ish":
        sig = (0.55 * wave_sine(phase) +
               0.28 * wave_sine(2*phase) +
               0.12 * wave_sine(3*phase) +
               0.05 * wave_sine(4*phase))
        sig *= adsr_envelope(n, attack=0.003, decay=0.22, sustain=0.15, release=0.35)

        click_len = min(int(0.008 * SAMPLE_RATE), n)
        sig[:click_len] += (np.random.uniform(-0.15, 0.15, click_len).astype(np.float32)
                            * np.linspace(1, 0, click_len))

    elif instrument == "Chiptune":
        wobble = 1.0 + 0.003 * np.sin(2*np.pi*6.0*t)
        sig = np.sign(np.sin(2*np.pi*freq*wobble*t))
        sig *= adsr_envelope(n, attack=0.002, decay=0.04, sustain=0.6, release=0.08)

    else:
        sig = wave_sine(phase) * adsr_envelope(n)

    sig = np.clip(sig, -1.0, 1.0)
    sig *= float(volume)

    mono = (sig * 32767).astype(np.int16)
    stereo = np.column_stack((mono, mono))
    return stereo

# -----------------------------
# UI helpers
# -----------------------------
def draw_rounded_rect(surf, rect, color, radius=10):
    pygame.draw.rect(surf, color, rect, border_radius=radius)

# -----------------------------
# App
# -----------------------------
def main():
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=CHANNELS, buffer=BUFFER)

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Keyboard Music Lab")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("segoeui", 18)
    font_big = pygame.font.SysFont("segoeui", 28, bold=True)
    font_mono = pygame.font.SysFont("consolas", 16)

    instruments = ["Piano-ish", "Organ", "Bell", "Sine", "Triangle", "Saw", "Square", "Chiptune"]
    instrument = "Piano-ish"
    octave_shift = 0
    volume = 0.45
    note_duration = 6.0  # long; stops on key-up

    sound_cache = {}
    active_channels = {}

    white_keys = [k for k in KEYBOARD_MAP if not k[2]]
    black_keys = [k for k in KEYBOARD_MAP if k[2]]

    # Layout
    top_bar = pygame.Rect(0, 0, WINDOW_W, 80)

    panel_area = pygame.Rect(40, 95, WINDOW_W - 80, 110)  # a bit taller
    piano_area = pygame.Rect(40, 220, WINDOW_W - 80, 240)

    # Reserve a fixed right-side controls column so text/buttons never overlap instruments
    CTRL_COL_W = 300
    CTRL_PAD = 14
    controls_area = pygame.Rect(panel_area.right - CTRL_COL_W, panel_area.y, CTRL_COL_W, panel_area.height)
    instruments_area = pygame.Rect(panel_area.x, panel_area.y, panel_area.width - CTRL_COL_W - 10, panel_area.height)

    # Instrument buttons (wrapped inside instruments_area only)
    btns = []
    x = instruments_area.x + 10
    y = instruments_area.y + 14
    row_h = 42
    gap = 10

    for name in instruments:
        w = 110 if len(name) <= 6 else 140
        rect = pygame.Rect(x, y, w, 34)
        # wrap if next button would cross instruments_area
        if rect.right > instruments_area.right - 10:
            x = instruments_area.x + 10
            y += row_h
            rect = pygame.Rect(x, y, w, 34)
        btns.append((name, rect))
        x += w + gap

    # Controls buttons in controls_area
    # (positions are anchored inside the reserved column)
    oct_down = pygame.Rect(controls_area.x + CTRL_PAD + 170, controls_area.y + 16, 50, 34)
    oct_up   = pygame.Rect(controls_area.x + CTRL_PAD + 230, controls_area.y + 16, 50, 34)

    vol_down = pygame.Rect(controls_area.x + CTRL_PAD + 170, controls_area.y + 62, 50, 34)
    vol_up   = pygame.Rect(controls_area.x + CTRL_PAD + 230, controls_area.y + 62, 50, 34)

    def get_sound(inst, octv, semitone):
        key = (inst, octv, semitone, round(volume, 3))
        if key in sound_cache:
            return sound_cache[key]

        freq = semitone_to_freq(semitone, octv)
        audio = synth_note(freq, note_duration, inst, volume)
        snd = pygame.sndarray.make_sound(audio)
        sound_cache[key] = snd
        return snd

    def current_note_name(semitone_offset):
        total = semitone_offset + octave_shift * 12
        name = NOTE_NAMES[(total % 12 + 12) % 12]
        octv = 4 + math.floor(total / 12)
        return f"{name}{octv}"

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_LEFTBRACKET:
                    octave_shift = max(-2, octave_shift - 1)
                    sound_cache.clear()
                if event.key == pygame.K_RIGHTBRACKET:
                    octave_shift = min(2, octave_shift + 1)
                    sound_cache.clear()

                if event.key == pygame.K_MINUS:
                    volume = max(0.05, volume - 0.05)
                    sound_cache.clear()
                if event.key == pygame.K_EQUALS:
                    volume = min(1.0, volume + 0.05)
                    sound_cache.clear()

                for pk, semi, is_black, label in KEYBOARD_MAP:
                    if event.key == pk and pk not in active_channels:
                        snd = get_sound(instrument, octave_shift, semi)
                        ch = snd.play()
                        if ch is not None:
                            active_channels[pk] = ch

            elif event.type == pygame.KEYUP:
                if event.key in active_channels:
                    active_channels[event.key].fadeout(80)
                    del active_channels[event.key]

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                for name, rect in btns:
                    if rect.collidepoint((mx, my)):
                        instrument = name
                        sound_cache.clear()

                if oct_down.collidepoint((mx, my)):
                    octave_shift = max(-2, octave_shift - 1)
                    sound_cache.clear()
                if oct_up.collidepoint((mx, my)):
                    octave_shift = min(2, octave_shift + 1)
                    sound_cache.clear()

                if vol_down.collidepoint((mx, my)):
                    volume = max(0.05, volume - 0.05)
                    sound_cache.clear()
                if vol_up.collidepoint((mx, my)):
                    volume = min(1.0, volume + 0.05)
                    sound_cache.clear()

        # -----------------------------
        # Draw
        # -----------------------------
        screen.fill((14, 14, 18))

        draw_rounded_rect(screen, top_bar, (22, 22, 30), 0)
        screen.blit(font_big.render("Keyboard Music Lab", True, (245, 245, 255)), (28, 22))
        help_text = "Play: A S D F G H J K | W E T Y U | Octave: [ ] | Volume: - = | ESC quits"
        screen.blit(font.render(help_text, True, (190, 190, 205)), (28, 52))

        # Panel
        draw_rounded_rect(screen, panel_area, (20, 20, 26), 16)

        # Labels
        screen.blit(font.render("Instruments:", True, (220, 220, 235)), (panel_area.x + 10, panel_area.y - 22))

        # Instrument buttons
        for name, rect in btns:
            is_active = (name == instrument)
            bg = (65, 45, 120) if is_active else (35, 35, 46)
            fg = (255, 255, 255) if is_active else (220, 220, 230)
            draw_rounded_rect(screen, rect, bg, 10)
            screen.blit(font.render(name, True, fg), (rect.x + 10, rect.y + 7))

        # Right controls column background (subtle)
        inner = pygame.Rect(controls_area.x + 6, controls_area.y + 10, controls_area.width - 12, controls_area.height - 20)
        draw_rounded_rect(screen, inner, (18, 18, 24), 14)

        # Controls text
        tx = controls_area.x + CTRL_PAD + 14
        screen.blit(font.render(f"Octave: {octave_shift:+d}", True, (230, 230, 240)), (tx, controls_area.y + 22))
        screen.blit(font.render(f"Volume: {volume:.2f}", True, (230, 230, 240)), (tx, controls_area.y + 68))

        # Controls buttons
        for rect, text in [(oct_down, "−"), (oct_up, "+"), (vol_down, "−"), (vol_up, "+")]:
            draw_rounded_rect(screen, rect, (38, 38, 50), 10)
            screen.blit(font_big.render(text, True, (245, 245, 255)), (rect.x + 16, rect.y + 3))

        # Piano
        draw_rounded_rect(screen, piano_area, (18, 18, 24), 18)

        key_w = (piano_area.width - 20) // len(white_keys)
        key_h = piano_area.height - 30
        base_x = piano_area.x + 10
        base_y = piano_area.y + 15

        for i, (pk, semi, is_black, label) in enumerate(white_keys):
            r = pygame.Rect(base_x + i * key_w, base_y, key_w - 4, key_h)
            pressed = pk in active_channels
            color = (240, 240, 245) if not pressed else (200, 220, 255)
            pygame.draw.rect(screen, color, r, border_radius=10)
            pygame.draw.rect(screen, (70, 70, 85), r, width=2, border_radius=10)

            note_name = current_note_name(semi)
            screen.blit(font_mono.render(note_name, True, (40, 40, 55)), (r.x + 8, r.bottom - 44))
            screen.blit(font_mono.render(label, True, (40, 40, 55)), (r.x + 8, r.bottom - 24))

        black_positions = {1: 0.70, 3: 1.70, 6: 3.70, 8: 4.70, 10: 5.70}
        black_w = int(key_w * 0.62)
        black_h = int(key_h * 0.62)

        for pk, semi, is_black, label in black_keys:
            if semi not in black_positions:
                continue
            pos = black_positions[semi]
            x = int(base_x + pos * key_w - black_w / 2)
            r = pygame.Rect(x, base_y, black_w, black_h)
            pressed = pk in active_channels
            color = (40, 40, 52) if not pressed else (120, 110, 170)
            pygame.draw.rect(screen, color, r, border_radius=10)
            pygame.draw.rect(screen, (10, 10, 14), r, width=2, border_radius=10)

            note_name = current_note_name(semi)
            screen.blit(font_mono.render(note_name, True, (235, 235, 245)), (r.x + 8, r.bottom - 42))
            screen.blit(font_mono.render(label, True, (235, 235, 245)), (r.x + 8, r.bottom - 22))

        status = f"Instrument: {instrument}  |  Octave: {octave_shift:+d}  |  Volume: {volume:.2f}"
        screen.blit(font.render(status, True, (200, 200, 215)), (40, 480))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
