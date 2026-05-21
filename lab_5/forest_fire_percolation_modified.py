#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratorium: perkolacja na przykładzie pożaru lasu
- model Drossel–Schwabl (wzrost drzew p, pioruny f) + powiązanie z perkolacją,
- dodatkowo: klasyczny eksperyment perkolacyjny "czy pożar przebija na drugą stronę?" (site percolation).

Struktura programu jest celowo podobna do przykładowego epidemia.cpp (SIR): Stan/Koordynaty/Parametry/RNG/Populacja/Statystyka/main.

Wymagania:
    pip install numpy matplotlib pillow
Opcjonalnie do MP4:
    ffmpeg (np. apt-get install ffmpeg lub conda install -c conda-forge ffmpeg)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from collections import deque

# =========================
# 1) Stany i koordynaty
# =========================

class Stan(IntEnum):
    PUSTO = 0
    DRZEWO = 1
    OGIEŃ_ZOLTY = 2
    OGIEŃ_POMAR = 3
    OGIEŃ_CZERW = 4


@dataclass(frozen=True)
class Koordynaty:
    x: int
    y: int

    def sasiad_lewy(self) -> "Koordynaty": return Koordynaty(self.x - 1, self.y)
    def sasiad_prawy(self) -> "Koordynaty": return Koordynaty(self.x + 1, self.y)
    def sasiad_gorny(self) -> "Koordynaty": return Koordynaty(self.x, self.y - 1)
    def sasiad_dolny(self) -> "Koordynaty": return Koordynaty(self.x, self.y + 1)


@dataclass
class ParametryPozaru:
    """
    Parametry modelu:
      p: prawdopodobieństwo wzrostu drzewa na pustym polu w jednym kroku,
      f: prawdopodobieństwo uderzenia pioruna w drzewo w jednym kroku,
      q: prawdopodobieństwo przeniesienia ognia z sąsiada (q=1 -> deterministycznie).
    """
    p: float = 0.01
    f: float = 1e-5
    q: float = 1.0


# =========================
# 2) RNG
# =========================

class RNG:
    def __init__(self, seed: int = 132123) -> None:
        self._rng = np.random.default_rng(seed)

    def losuj_od_0_do_1(self, size=None) -> np.ndarray:
        return self._rng.random(size=size)

    def losuj_int(self, low: int, high: int, size=None) -> np.ndarray:
        return self._rng.integers(low, high, size=size, endpoint=False)


# =========================
# 3) Populacja/Las
# =========================

class Las:
    """
    Las jako LxL siatka stanów. Aktualizacja jest synchroniczna (reguły wykonywane równolegle),
    jak w automatach komórkowych.
    """
    def __init__(self, bok_mapy: int, seed: int = 132123) -> None:
        self.L = int(bok_mapy)
        self.rng = RNG(seed=seed)
        self.reset()

    def reset(self) -> None:
        self.siatka = np.zeros((self.L, self.L), dtype=np.uint8)

    def inicjalizuj_losowo(self, gestosc_drzew: float) -> None:
        self.reset()
        mask = self.rng.losuj_od_0_do_1(size=(self.L, self.L)) < float(gestosc_drzew)
        self.siatka[mask] = Stan.DRZEWO

    def gestosc_drzew(self) -> float:
        return float(np.mean(self.siatka == Stan.DRZEWO))

    def liczba_drzew(self) -> int:
        return int(np.sum(self.siatka == Stan.DRZEWO))

    def liczba_ognia(self) -> int:
        return int(np.sum(self.siatka >= Stan.OGIEŃ_ZOLTY))

    def _sasiedzi_ognia(self) -> np.ndarray:
        """True tam, gdzie komórka ma palącego się sąsiada (4-sąsiedztwo, bez zawijania)."""
        fire = (self.siatka >= Stan.OGIEŃ_ZOLTY)
        nb = np.zeros_like(fire, dtype=bool)
        nb[1:, :]  |= fire[:-1, :]
        nb[:-1, :] |= fire[1:, :]
        nb[:, 1:]  |= fire[:, :-1]
        nb[:, :-1] |= fire[:, 1:]
        return nb

    def _rozmiary_klastrow_drzew(self, start_mask: np.ndarray, tree_mask: np.ndarray) -> List[int]:
        """
        Dla komórek True w start_mask (zapalonych przez piorun) wyznacza rozmiary spójnych klastrów drzew
        (4-sąsiedztwo), licząc tylko po drzewach z tree_mask. Każdy klaster liczony jest maks. raz.
        To przybliża "ile drzew docelowo spłonie", gdy ogień rozchodzi się deterministycznie.
        """
        visited = np.zeros_like(start_mask, dtype=bool)
        sizes: List[int] = []
        L = self.L

        starts = np.argwhere(start_mask)
        for (x0, y0) in starts:
            x0, y0 = int(x0), int(y0)
            if visited[x0, y0] or (not tree_mask[x0, y0]):
                continue

            q = deque([(x0, y0)])
            visited[x0, y0] = True
            size = 0

            while q:
                x, y = q.popleft()
                if not tree_mask[x, y]:
                    continue
                size += 1

                if x > 0 and (not visited[x-1, y]):
                    visited[x-1, y] = True
                    q.append((x-1, y))
                if x < L-1 and (not visited[x+1, y]):
                    visited[x+1, y] = True
                    q.append((x+1, y))
                if y > 0 and (not visited[x, y-1]):
                    visited[x, y-1] = True
                    q.append((x, y-1))
                if y < L-1 and (not visited[x, y+1]):
                    visited[x, y+1] = True
                    q.append((x, y+1))

            if size > 0:
                sizes.append(size)

        return sizes

    def krok(self, par: ParametryPozaru) -> Dict[str, Any]:
        """
        Jeden krok czasowy (automat komórkowy, reguły Drossel–Schwabl):
          1) Płonące -> pusto (tu: 3 fazy koloru: żółty->pomarańczowy->czerwony->pusto),
          2) Drzewo pali się, jeśli sąsiad płonie (z prawdopodobieństwem q),
          3) Drzewo zapala się samo z prawdopodobieństwem f (piorun),
          4) Pusto zarasta drzewem z prawdopodobieństwem p.
        Zwraca zdarzenia/statystyki z tego kroku.
        """
        p, f, qprob = float(par.p), float(par.f), float(par.q)

        s = self.siatka

        # 4) Wzrost drzew
        empty = (s == Stan.PUSTO)
        grow = empty & (self.rng.losuj_od_0_do_1(size=s.shape) < p)

        # 2) i 3) Zapłon drzew
        trees = (s == Stan.DRZEWO)
        near_fire = self._sasiedzi_ognia()
        ignite_from_neighbors = trees & near_fire & (self.rng.losuj_od_0_do_1(size=s.shape) < qprob)
        ignite_from_lightning = trees & (~near_fire) & (self.rng.losuj_od_0_do_1(size=s.shape) < f)

        # --- "Wykres": rozmiary klastrów zapalonych przez piorun
        rozmiary_klastrow = self._rozmiary_klastrow_drzew(ignite_from_lightning, trees)

        nowy_ogien = ignite_from_neighbors | ignite_from_lightning
        ile_nowo_zapalonych = int(np.sum(nowy_ogien))

        # 1) Ewolucja ognia
        s2 = s.copy()
        s2[s == Stan.OGIEŃ_ZOLTY] = Stan.OGIEŃ_POMAR
        s2[s == Stan.OGIEŃ_POMAR] = Stan.OGIEŃ_CZERW
        s2[s == Stan.OGIEŃ_CZERW] = Stan.PUSTO

        # Zastosuj zapłon i wzrost (na s2)
        s2[grow] = Stan.DRZEWO
        s2[nowy_ogien] = Stan.OGIEŃ_ZOLTY

        self.siatka = s2

        return {
            "nowe_drzewa": int(np.sum(grow)),
            "nowy_ogien": ile_nowo_zapalonych,
            "ognia": self.liczba_ognia(),
            "drzew": self.liczba_drzew(),
            "rozmiary_klastrow": rozmiary_klastrow,
        }

    def zapisz_do_pliku(self, nazwa: str | Path) -> None:
        nazwa = Path(nazwa)
        np.savetxt(nazwa, self.siatka.astype(int), fmt="%d", delimiter="\t")


# =========================
# 4) Statystyka
# =========================

class Statystyka:
    def __init__(self) -> None:
        self.drzewa: List[int] = []
        self.ognia: List[int] = []
        self.gestosc: List[float] = []
        self.rozmiary_pozarow: List[int] = []

    def dodaj(self, las: Las, zdarzenia: Dict[str, Any]) -> None:
        self.drzewa.append(int(zdarzenia["drzew"]))
        self.ognia.append(int(zdarzenia["ognia"]))
        self.gestosc.append(float(las.gestosc_drzew()))
        for s in (zdarzenia.get("rozmiary_klastrow", []) or []):
            if int(s) > 0:
                self.rozmiary_pozarow.append(int(s))

    def maksimum_ognia(self) -> int:
        return int(max(self.ognia) if self.ognia else 0)

    def kiedy_maksimum_ognia(self) -> int:
        if not self.ognia:
            return 0
        return int(np.argmax(np.array(self.ognia)))

    def zapisz_do_pliku(self, nazwa: str | Path) -> None:
        nazwa = Path(nazwa)
        dane = np.column_stack([
            np.arange(len(self.drzewa)),
            np.array(self.drzewa, dtype=int),
            np.array(self.ognia, dtype=int),
            np.array(self.gestosc, dtype=float),
        ])
        header = "dzien\tliczba_drzew\tliczba_ognia\tgestosc_drzew"
        np.savetxt(nazwa, dane, fmt=["%d", "%d", "%d", "%.6f"], delimiter="\t", header=header)


# =========================
# 5) Wizualizacja
# =========================

def _kolormap_lasu() -> ListedColormap:
    """
    Kolory:
      - pusto: czarny
      - drzewo: zielony
      - ogień: żółty, pomarańczowy, czerwony (wymóg zadania)
    """
    return ListedColormap([
        (0.0, 0.0, 0.0),     # PUSTO
        (0.0, 0.4, 0.0),     # DRZEWO
        (1.0, 1.0, 0.0),     # OGIEŃ_ZOLTY
        (1.0, 0.55, 0.0),    # OGIEŃ_POMAR
        (1.0, 0.0, 0.0),     # OGIEŃ_CZERW
    ])


def nagraj_animacje(
    las: Las,
    par: ParametryPozaru,
    ile_krokow: int,
    plik_wyj: str | Path = "pozar.gif",
    fps: int = 20,
    co_ile_krokow: int = 1,
    tytul: str = "Pożar lasu (model Drossel–Schwabl)",
) -> Statystyka:
    """
    Nagrywa animację przebiegu pożaru i jednocześnie zbiera statystyki.
    Uwaga: zapis MP4 wymaga ffmpeg. GIF zapisuje się przez pillow.
    """
    plik_wyj = Path(plik_wyj)
    stat = Statystyka()
    cmap = _kolormap_lasu()

    fig, ax = plt.subplots()
    im = ax.imshow(las.siatka, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(tytul)
    ax.set_xticks([])
    ax.set_yticks([])

    # ZMIANA: Dodano color="white" oraz opcjonalnie pogrubienie (weight="bold"),
    # aby tekst był jeszcze lepiej widoczny na zielonym/żółtym tle.
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", color="white", weight="bold")

    def update(frame: int):
        for _ in range(co_ile_krokow):
            ev = las.krok(par)
            stat.dodaj(las, ev)
        im.set_data(las.siatka)
        txt.set_text(
            f"krok={len(stat.drzewa)}\n"
            f"drzewa={stat.drzewa[-1] if stat.drzewa else 0}\n"
            f"gestosc={stat.gestosc[-1]*100 if stat.gestosc else 0:.1f}%\n"
            f"ogien={stat.ognia[-1] if stat.ognia else 0}"
        )
        return [im, txt]

    frames = int(np.ceil(ile_krokow / max(co_ile_krokow, 1)))
    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)

    if plik_wyj.suffix.lower() == ".mp4":
        anim.save(plik_wyj, fps=fps)  # wymaga ffmpeg
    else:
        anim.save(plik_wyj, writer="pillow", fps=fps)

    plt.close(fig)
    return stat


def wykres_rozmiarow_pozarow(stat: Statystyka, plik: str | Path = "rozmiary_pozarow_loglog.png") -> None:
    """Wykres: rozkład rozmiarów pożarów (spalone drzewa / piorun), w skali log-log."""
    plik = Path(plik)
    sizes = np.array(stat.rozmiary_pozarow, dtype=int)
    if sizes.size == 0:
        print("Brak rozmiarów pożarów (pioruny nie trafiły w drzewa?) – zwiększ liczbę kroków lub f.")
        return

    min_s = max(1, int(sizes.min()))
    max_s = int(sizes.max())
    bins = np.unique(np.logspace(np.log10(min_s), np.log10(max_s), num=30).astype(int))
    if bins.size < 5:
        bins = np.arange(min_s, max_s + 2)

    hist, edges = np.histogram(sizes, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots()
    ax.plot(centers, hist, marker="o", linestyle="-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rozmiar pożaru (spalone drzewa / piorun)")
    ax.set_ylabel("Liczba zdarzeń")
    ax.set_title("Rozkład rozmiarów pożarów (log-log)")
    fig.tight_layout()
    fig.savefig(plik, dpi=160)
    plt.close(fig)


def wykres_gestosci_w_czasie(stat: Statystyka, plik: str | Path = "gestosc_w_czasie.png") -> None:
    plik = Path(plik)
    t = np.arange(len(stat.gestosc))
    fig, ax = plt.subplots()
    ax.plot(t, np.array(stat.gestosc) * 100.0)
    ax.set_xlabel("Krok czasowy")
    ax.set_ylabel("Gęstość drzew [%]")
    ax.set_title("Gęstość drzew w czasie")
    fig.tight_layout()
    fig.savefig(plik, dpi=160)
    plt.close(fig)


# =========================
# 6) Eksperyment perkolacyjny (statyczny)
# =========================

def czy_przebija_na_prawo(L: int, gestosc: float, seed: int) -> bool:
    """
    Klasyczna ilustracja perkolacji (site percolation na siatce kwadratowej):
      - drzewa występują na polu z prawdopodobieństwem gestosc=ρ,
      - podpalamy drzewa w lewej kolumnie,
      - ogień rozchodzi się deterministycznie po sąsiadach,
      - sprawdzamy, czy ogień dotarł do prawej kolumny.
    """
    rng = np.random.default_rng(seed)
    trees = rng.random((L, L)) < gestosc

    burning = np.zeros((L, L), dtype=bool)
    burning[:, 0] = trees[:, 0]
    burned = np.zeros((L, L), dtype=bool)

    while burning.any():
        burned |= burning
        nb = np.zeros_like(burning, dtype=bool)
        nb[1:, :]  |= burning[:-1, :]
        nb[:-1, :] |= burning[1:, :]
        nb[:, 1:]  |= burning[:, :-1]
        nb[:, :-1] |= burning[:, 1:]
        burning = nb & trees & (~burned)

    return bool(burned[:, -1].any())


def krzywa_perkolacji(
    L: int = 128,
    gestosci: Optional[np.ndarray] = None,
    proby_na_punkt: int = 100,
    seed0: int = 1,
    plik: str | Path = "krzywa_perkolacji.png",
) -> Tuple[np.ndarray, np.ndarray]:
    """Empirycznie wyznacza P(przebicia) w funkcji gęstości drzew ρ."""
    if gestosci is None:
        gestosci = np.linspace(0.45, 0.75, 25)
    gestosci = np.array(gestosci, dtype=float)

    probs = np.zeros_like(gestosci)
    for i, rho in enumerate(gestosci):
        hits = 0
        for k in range(proby_na_punkt):
            if czy_przebija_na_prawo(L, float(rho), seed=seed0 + i * 10_000 + k):
                hits += 1
        probs[i] = hits / proby_na_punkt

    fig, ax = plt.subplots()
    ax.plot(gestosci, probs, marker="o", linestyle="-")
    ax.set_xlabel("Gęstość drzew ρ")
    ax.set_ylabel("P(przebicia ognia na prawą krawędź)")
    ax.set_title("Krzywa perkolacji: przejście pożaru przez las")
    fig.tight_layout()
    fig.savefig(Path(plik), dpi=160)
    plt.close(fig)
    return gestosci, probs


def wykres_porownawczy_q(slownik_statystyk: Dict[float, Statystyka], plik: str | Path = "porownanie_q_w_czasie.png") -> None:
    """Tworzy wspólny wykres gęstości drzew w czasie dla różnych wartości q."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for q_val, stat in slownik_statystyk.items():
        t = np.arange(len(stat.gestosc))
        ax.plot(t, np.array(stat.gestosc) * 100.0, label=f"q = {q_val}")

    ax.set_xlabel("Krok czasowy")
    ax.set_ylabel("Gęstość drzew [%]")
    ax.set_title("Wpływ parametru q na gęstość lasu w czasie")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(Path(plik), dpi=160)
    plt.close(fig)