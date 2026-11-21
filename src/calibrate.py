"""
Kalibrierung am fertig gebauten Hund (Plan 5.5, Schritt 1).

Für jedes Gelenk einzeln: Nullpunkt, Drehrichtung und Grenzen messen. Danach
weiß servos.py, welcher Servo-Winkel zu welchem Gelenkwinkel gehört.

Vorher: Hund aufbocken (Beine hängen frei), jemand steht am Not-Aus bzw. am
XT60-Stecker, Servostrom an. Es bewegt sich immer nur das eine Gelenk, das
gerade dran ist, in Schritten von höchstens 10°, langsam (1° pro 20 ms).

    python src/calibrate.py --hardware                  # alle 12 Gelenke nacheinander
    python src/calibrate.py --hardware --joint FR_lower  # nur eins

Befehle während der Kalibrierung:
    +2 / -5 / +0.5   so viele Grad drehen (höchstens 10)
    ref              das Gelenk steht jetzt genau in der Bezugslage (steht dabei)
    dir              Richtungstest: dreht 5° und fragt, wohin es ging
    min / max        hier ist ein Anschlag (kurz bevor etwas anstößt), je einmal pro Seite
    ok               fertig mit diesem Gelenk
    q                abbrechen

Gespeichert wird nach jedem Gelenk in ~/nova-kalibrierung.json. Diese Datei kommt
als calibration.json ins Repo, dann liest servos.py sie überall (Pi, PC, Training).

Noch nicht auf der Hardware getestet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

from gait import STAND
from kinematics import body_ik
from servos import JOINT_ORDER, JOINTS, Joint

SIM_MODE = os.environ.get("NOVA_SIM") == "1"
OUT = Path.home() / "nova-kalibrierung.json"

# Gelenk -> (Kinematik-Winkel der Bezugslage, wie die Bezugslage aussieht, wohin "+" geht)
REFERENCE = {
    "shoulder": (0.0, "Bein hängt von vorne gesehen senkrecht nach unten (Hüftstück waagrecht)",
                 "das Bein nach außen (weg vom Körper)"),
    "upper": (0.0, "Oberschenkel zeigt von der Seite gesehen senkrecht nach unten",
              "der Oberschenkel nach vorne"),
    "lower": (90.0, "Knie im rechten Winkel: Unterschenkel steht 90° zum Oberschenkel (Geodreieck)",
              "das Knie weiter zu (mehr gebeugt)"),
}
START = 90.0           # Grad roh: so wurden die Servos zentriert und die Hörner aufgesteckt
MAX_MOVE = 10.0        # Grad pro Befehl höchstens
STEP_DELAY = 0.02      # s pro Grad beim Fahren
DIRECTION_TEST = 5.0   # Grad für den Richtungstest
MARGIN = 3.0           # Grad Abstand zu den gemessenen Anschlägen
MIN_RANGE = 10.0       # Grad: noch engere Grenzen sind wohl ein Messfehler

_NUMBER = re.compile(r"^[+-]\d+(\.\d+)?$")


class Session:
    """Ein Gelenk nach dem anderen kalibrieren. ask/say/sleep austauschbar (Tests)."""

    def __init__(self, servos, ask=input, say=print, sleep=time.sleep) -> None:
        self.servos, self.ask, self.say, self.sleep = servos, ask, say, sleep
        self.position: dict[int, float] = {}

    def _go(self, channel: int, target: float) -> None:
        """Langsam (1° pro Schritt) auf `target` fahren."""
        current = self.position.get(channel)
        if current is None:
            self.position[channel] = self.servos.set_raw(channel, target)
            return
        steps = max(1, math.ceil(abs(target - current)))
        for i in range(1, steps + 1):
            self.position[channel] = self.servos.set_raw(channel, current + (target - current) * i / steps)
            self.sleep(STEP_DELAY)

    def calibrate_joint(self, name: str) -> Joint | None:
        """Ein Gelenk kalibrieren. None = abgebrochen. Danach ist der Servo locker."""
        channel = JOINTS[name].channel
        kind = name[3:]
        ref_angle, ref_text, plus_text = REFERENCE[kind]
        self.say(f"\n=== {name} (Kanal {channel}) ===")
        self.say(f"Bezugslage: {ref_text}.")
        self.say(f"Plus heißt hier: {plus_text}.")
        self.say("Befehle: +2/-5 drehen (höchstens 10°), ref, dir, min, max, ok, q")
        self._go(channel, START)
        ref: float | None = None
        positive: bool | None = None
        limits: dict[str, float] = {}
        while True:
            pos = self.position[channel]
            cmd = self.ask(f"{name} bei {pos:.1f}° > ").strip().lower()
            if cmd in ("q", "quit", "ende"):
                self.servos.release(channel)
                return None
            if _NUMBER.match(cmd):
                delta = float(cmd)
                if abs(delta) > MAX_MOVE:
                    self.say(f"Bitte höchstens {MAX_MOVE:.0f}° auf einmal.")
                    continue
                self._go(channel, pos + delta)
            elif cmd == "ref":
                ref = pos
                self.say(f"Bezugslage bei {pos:.1f}° gespeichert.")
            elif cmd == "dir":
                self._go(channel, pos + DIRECTION_TEST)
                answer = self.ask(f"Ist gerade {plus_text} gegangen? (j/n) ").strip().lower()
                positive = answer.startswith(("j", "y"))
                self._go(channel, pos)
            elif cmd in ("min", "max"):
                limits[cmd] = pos
                self.say(f"Anschlag ({cmd}) bei {pos:.1f}° gespeichert.")
            elif cmd in ("ok", "fertig"):
                missing = [text for text, have in (("Bezugslage (ref)", ref is not None),
                                                   ("Richtung (dir)", positive is not None),
                                                   ("Anschlag min", "min" in limits),
                                                   ("Anschlag max", "max" in limits)) if not have]
                if missing:
                    self.say(f"Es fehlt noch: {', '.join(missing)}")
                    continue
                joint = self._result(name, channel, ref, positive, limits)
                if joint is None:
                    limits.clear()
                    continue
                self.servos.release(channel)
                self.say(f"{name}: Nullpunkt {joint.zero:.1f}°, Grenzen {joint.min_angle:.1f}..{joint.max_angle:.1f}°, "
                         f"Stehpose {joint.neutral:.1f}°, {'gespiegelt' if joint.inverted else 'normal'} eingebaut.")
                return joint
            else:
                self.say("Unbekannter Befehl. Möglich: +2, -5, ref, dir, min, max, ok, q")

    def _result(self, name: str, channel: int, ref: float, positive: bool, limits: dict[str, float]) -> Joint | None:
        """Messwerte (roh) -> Joint für servos.py. None, wenn die Grenzen nicht passen."""
        inverted = not positive  # gespiegelt: mehr Servo-Winkel dreht das Gelenk in die Minus-Richtung

        def logical(raw: float) -> float:  # servos.set_joint() rechnet bei inverted 180 - Winkel
            return 180.0 - raw if inverted else raw

        ref_angle = REFERENCE[name[3:]][0]
        zero = logical(ref) - ref_angle
        low, high = sorted((logical(limits["min"]), logical(limits["max"])))
        low, high = low + MARGIN, high - MARGIN
        if high - low < MIN_RANGE:
            self.say(f"Die Anschläge liegen zu eng beieinander ({high - low:.1f}° nach Abzug). "
                     "Bitte min und max noch einmal messen.")
            return None
        stand = getattr(body_ik(STAND.feet, STAND.body)[name[:2]], name[3:])
        neutral = zero + stand
        if not low <= neutral <= high:
            self.say(f"Achtung: Die Stehpose ({neutral:.1f}°) liegt außerhalb der Grenzen. Vermutlich sitzt das "
                     "Horn um einen Zahn versetzt. Horn umstecken und dieses Gelenk noch einmal kalibrieren.")
            neutral = min(high, max(low, neutral))
        return Joint(channel, round(low, 1), round(high, 1), round(neutral, 1), round(zero, 1), inverted)


def save(path, joints: dict[str, Joint]) -> None:
    """Gelenke in die Kalibrier-Datei schreiben, schon gespeicherte andere Gelenke bleiben."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stored = data.get("joints", {})
    for name, j in joints.items():
        stored[name] = {"min": j.min_angle, "max": j.max_angle, "neutral": j.neutral, "zero": j.zero,
                        "inverted": j.inverted}
    data = {"datum": date.today().isoformat(),
            "hinweis": "Aus src/calibrate.py. Als calibration.json ins Repo, dann gilt sie in servos.py.",
            "joints": stored}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gelenke von Nova kalibrieren (am fertigen Hund, aufgebockt)")
    parser.add_argument("--joint", help="nur dieses Gelenk, z. B. FR_lower (Standard: alle 12)")
    parser.add_argument("--out", default=str(OUT), help="Kalibrier-Datei (Standard %(default)s)")
    parser.add_argument("--hardware", action="store_true", help="echte Servos bewegen")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    names = [args.joint] if args.joint else list(JOINT_ORDER)
    unknown = [n for n in names if n not in JOINTS]
    if unknown:
        print(f"Unbekanntes Gelenk: {', '.join(unknown)}. Möglich: {', '.join(JOINT_ORDER)}")
        return 2
    if not SIM_MODE and not args.hardware:
        print("Achtung: Das bewegt die echten Servos, ohne Gelenkgrenzen (die werden ja gerade gemessen).\n"
              "Hund aufbocken, jemand an den XT60-Stecker, dann mit --hardware starten.")
        return 2
    from servos import Servos

    servos = Servos()
    session = Session(servos)
    print("Kalibrierung: Hund aufgebockt? Beine hängen frei? Jemand am Stecker? Dann los.")
    done = []
    for name in names:
        joint = session.calibrate_joint(name)
        if joint is None:
            print("Abgebrochen.")
            break
        save(args.out, {name: joint})
        done.append(name)
    servos.relax()
    if done:
        print(f"\nGespeichert ({len(done)} Gelenke): {args.out}\n"
              "Auf den Laptop kopieren und als calibration.json ins Repo legen, dann pushen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
