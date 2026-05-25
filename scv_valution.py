import argparse
import numpy as np


def run_simulation(iterations, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 1. GENEROWANIE ZMIENNYCH LOSOWYCH (100 000 iteracji)
    # Ryzyko projektu pilotażowego: rozkład jednostajny (uniform) -> sukces jeśli wynik <= 0.60
    pilot_success = np.random.uniform(0, 1, iterations) <= 0.60

    # Nakłady inwestycyjne (CAPEX): rozkład trójkątny (900, 1000, 1100)
    capex_sim = np.random.triangular(900, 1000, 1100, iterations)

    # Cena jednostkowa: rozkład normalny (średnia 10, odchylenie 10% z 10 = 1.0) dla 5 lat
    # Generujemy macierz o wymiarach (iterations, 5 lat)
    prices_sim = np.random.normal(10.0, 1.0, size=(iterations, 5))

    # 2. PARAMETRY STAŁE MODELU FINANSOWEGO
    quantity = 100  # Stały wolumen sprzedaży
    var_cost_per_unit = 5.0  # Zmienny koszt jednostkowy
    fixed_costs = 50  # Stałe koszty operacyjne (600 mln operacyjnych - 50 mln deprecjacji)
    tax_rate = 0.19  # Podatek dochodowy 19%
    wacc = 0.10  # Koszt kapitału 10%
    pilot_capex = 100.0  # Koszt projektu pilotażowego (100 mln EUR)

    # Czynniki dyskontujące dla lat 1 do 5
    discount_factors = np.array([1 / (1 + wacc) ** t for t in range(1, 6)])

    # Kontener na wyniki NPV projektu rozszerzonego (extension)
    npv_ext_all = np.zeros(iterations)

    # 3. SYMULACJA PRZEPŁYWÓW PIENIĘŻNYCH (Wektoryzacja)
    for i in range(iterations):
        capex = capex_sim[i]
        depreciation = capex * 0.10  # Amortyzacja 10% rocznie
        residual_value = capex * 0.20  # Sprzedaż aktywów w 5. roku (20%)

        # Obliczanie EBIT i podatku dla każdego roku (rok 1 do 5)
        revenues = prices_sim[i] * quantity
        variable_costs = var_cost_per_unit * quantity
        ebit = revenues - variable_costs - fixed_costs - depreciation
        tax = ebit * tax_rate

        # Free Cash Flows (FCF) dla lat 1-4
        fcf = ebit - tax + depreciation

        # FCF dla roku 5 (dodajemy wartość końcową netto)
        fcf[-1] += residual_value

        # Bieżąca wartość przepływów (PV) zdyskontowana na rok t=0
        pv_fcf = np.sum(fcf * discount_factors)

        # NPV projektu rozszerzonego (PV przepływów minus początkowy CAPEX stochastyczny)
        npv_ext_all[i] = pv_fcf - capex

    # 4. IMPLEMENTACJA METOD SCV ORAZ 2MC
    # Inwestycja bazowa (pilot program): strata 100 mln w przypadku porażki, 0 w przypadku sukcesu rynkowego
    # (w ujęciu opcji, projekt bazowy sam w sobie bez opcji rozwoju generuje stratę nakładów)
    npv_base = np.where(pilot_success, -pilot_capex, -pilot_capex)

    # Model 2MC: Wynik finansowy przy automatycznym (sztywnym) wykonaniu w przypadku sukcesu pilotażu
    # Jeśli pilot się udał -> bierzemy wynik rozszerzenia minus koszt pilota. Jeśli nie -> tracimy koszt pilota.
    economic_result_2mc = np.where(
        pilot_success, -pilot_capex + npv_ext_all, -pilot_capex
    )

    # Model SCV: Warunkowe porównanie (opcja aktywowana TYLKO gdy NPV_ext > 0 w przypadku sukcesu pilotażu)
    # ROV = E[MAX((NPV_ext + WynikPilota) - NPV_base, 0)]
    # Po uproszczeniu matematycznym: jeśli sukces i NPV_ext > 0 -> wartość to NPV_ext. W innych wypadkach 0.
    rov_iterations = np.where((pilot_success) & (npv_ext_all > 0), npv_ext_all, 0)

    # 5. STATYSTYKI KOŃCOWE
    rov_mean = np.mean(rov_iterations)
    rov_std = np.std(rov_iterations)
    standard_error = rov_std / np.sqrt(iterations)

    mc2_mean = np.mean(economic_result_2mc)
    mc2_std = np.std(economic_result_2mc)

    # Obliczanie prawdopodobieństwa odrzucenia/porażki (wartość bliska zero w SCV)
    zero_hits_pct = (np.sum(rov_iterations == 0) / iterations) * 100

    print("-" * 50)
    print(f"WYNIKI SYMULACJI MONTE CARLO ({iterations:,} iteracji)")
    print("-" * 50)
    print(f"Wycena Opcji Realnej (SCV):      {rov_mean:.2f} mln EUR")
    print(f"Błąd standardowy estymatora SCV: {standard_error:.4f} mln EUR")
    print(f"Odchylenie standardowe SCV:      {rov_std:.2f} mln EUR")
    print(f"Scenariusze równe zero (odcięte): {zero_hits_pct:.1f}%")
    print("-" * 50)
    print(f"Elastyczność Decyzyjna (2MC):    {mc2_mean:.2f} mln EUR")
    print(f"Odchylenie standardowe 2MC:      {mc2_std:.2f} mln EUR")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wycena opcji realnych metodą SCV i 2MC przy użyciu symulacji Monte Carlo."
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100000,
        help="Liczba iteracji symulacji (domyślnie: 100000)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Ziarno generatora liczb losowych (RNG) dla replikowalności",
    )

    args = parser.parse_args()

    run_simulation(iterations=args.iterations, seed=args.seed)