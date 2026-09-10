# Ewaluacja MVP — raport do analizy pracy dyplomowej

Model osiąga mAP **63.70%** i NDS **68.85%** na zbiorze walidacyjnym nuScenes.

## Protokół eksperymentu

- Model: MVP CenterPoint VoxelNet.
- Fuzja: Fuzja punktowa na wejściu: LiDAR + punkty wirtualne wygenerowane z kamer.
- Zbiór: nuScenes validation, 6019 ramek, po 10 kolejnych skanów LiDAR.
- Dopasowanie AP: odległość środków 3D przy progach 0.5, 1.0, 2.0, 4.0 m; maksymalnie 500 detekcji na ramkę.
- Konfiguracja: `nusc_centerpoint_voxelnet_0075voxel_bevfusion_virtual_ft.py`.
- Checkpoint: `mvp_centerpoint_epoch_6.pth`.
- Punkty wirtualne wygenerowano wcześniej z obrazów; ewaluator otrzymuje je jako rozszerzoną chmurę punktów.

## Metryki główne

| Metryka | Wynik [%] |
| --- | ---: |
| mAP | 63.70 |
| NDS | 68.85 |

## Błędy True Positive nuScenes

| Metryka | Wynik | Jednostka | Interpretacja |
| --- | ---: | --- | --- |
| mATE | 0.2737 | m | średni błąd translacji; mniej = lepiej |
| mASE | 0.2560 | 1 - IoU | średni błąd skali; mniej = lepiej |
| mAOE | 0.2873 | rad | średni błąd orientacji; mniej = lepiej |
| mAVE | 0.2814 | m/s | średni błąd prędkości; mniej = lepiej |
| mAAE | 0.2022 | 1 - accuracy | średni błąd atrybutu; mniej = lepiej |

## AP według klasy i tolerancji odległości

Wartości podano w procentach. Średnie AP klasy jest średnią z czterech progów odległości.

| Klasa | AP@0.5 m | AP@1.0 m | AP@2.0 m | AP@4.0 m | Średnie AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| car | 76.28 | 86.28 | 89.55 | 90.72 | 85.71 |
| truck | 43.60 | 61.49 | 69.70 | 73.63 | 62.11 |
| bus | 44.86 | 68.97 | 78.32 | 81.20 | 68.34 |
| trailer | 7.74 | 23.85 | 33.49 | 46.38 | 27.87 |
| construction_vehicle | 2.42 | 14.70 | 27.57 | 40.49 | 21.29 |
| pedestrian | 86.39 | 88.23 | 89.54 | 90.32 | 88.62 |
| motorcycle | 64.46 | 75.20 | 76.67 | 77.11 | 73.36 |
| bicycle | 61.13 | 63.59 | 64.41 | 65.07 | 63.55 |
| traffic_cone | 76.01 | 76.93 | 78.71 | 81.20 | 78.21 |
| barrier | 56.85 | 68.20 | 72.37 | 74.47 | 67.97 |
| **Makrośrednia klas** | **51.97** | **62.74** | **68.03** | **72.06** | **63.70** |

## Porównanie z wynikami referencyjnymi

| Model | mAP [%] | NDS [%] | ΔmAP względem referencji [p.p.] | ΔNDS [p.p.] |
| --- | ---: | ---: | ---: | ---: |
| **Ten checkpoint** | **63.70** | **68.85** | — | — |
| CenterPoint-Voxel (publikowany baseline) | 59.50 | 66.70 | 4.20 | 2.15 |
| MVP CenterPoint-Voxel (wynik publikowany) | 66.00 | 69.90 | -2.30 | -1.05 |

Wartości referencyjne pochodzą z tabeli wyników walidacyjnych publikacji i repozytorium MVP.

## Wydajność inferencji

| GPU | Ramki pomiarowe / wszystkie | Latencja [ms/ramkę/GPU] | Przepustowość [FPS/GPU] | Pełna pętla [s] | Peak VRAM allocated [MiB/GPU] |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA GeForce RTX 5060 Ti | 2006 / 6019 | 131.56 | 7.60 | 764.04 | 354 |

Pomiar latencji obejmuje środkową 1/3 zbioru po rozgrzaniu GPU. Pełna pętla obejmuje ładowanie danych i inferencję CenterPoint, ale nie obejmuje generowania punktów wirtualnych z kamer ani obliczania metryk nuScenes.

## Wnioski do analizy

- Najwyższe średnie AP uzyskano dla: pedestrian (88.62%), car (85.71%), traffic_cone (78.21%).
- Najniższe średnie AP uzyskano dla: truck (62.11%), trailer (27.87%), construction_vehicle (21.29%); te klasy są głównym obszarem dalszej poprawy.
- Względem publikowanego baseline'u CenterPoint-Voxel checkpoint poprawia mAP o **4.20 p.p.** i NDS o **2.15 p.p.**
- Względem publikowanego wyniku MVP rezultat jest niższy o **2.30 p.p. mAP** i **1.05 p.p. NDS**.

## Ograniczenia interpretacji

- To pojedynczy checkpoint i pojedyncza ewaluacja; bez powtórzeń treningu nie można podać odchylenia standardowego ani przedziału ufności między seedami.
- Porównania z tabelą publikacji są referencyjne, nie są kontrolowanym badaniem ablacyjnym wykonanym w identycznym środowisku.
- Koszt generowania punktów wirtualnych przez model 2D nie wchodzi do podanej latencji; liczby nie reprezentują pełnego potoku kamera → MVP → detekcja 3D.
- `use_camera=false` w metadanych nuScenes oznacza, że ewaluator widzi tylko wejście punktowe. Nie oznacza to braku informacji z kamer, ponieważ jest ona zakodowana we wcześniej wygenerowanych punktach wirtualnych.

## Reprodukcja

```bash
conda activate mvp-clean
python scripts/test_mvp_model.py
```

Do ponownego wygenerowania raportu bez inferencji:

```bash
python scripts/test_mvp_model.py --metrics-only results/reference/metrics_summary.json
```

Dane maszynowe do dalszych wykresów i obliczeń znajdują się w `thesis_metrics.csv`; pełny raport źródłowy znajduje się w `mvp_test_report.json`.
