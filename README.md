# MVP — trening, walidacja i testy

Komplet lokalnego projektu Multimodal Virtual Point 3D Detection: CenterNet2 generuje punkty wirtualne z kamer, a CenterPoint VoxelNet wykonuje detekcję 3D na LiDAR + punktach wirtualnych. Repozytorium docelowe: **prywatne `mmr0z/mvp-training`**.

## Zawartość

- `requirements.txt`: zależności z wersjami z lokalnego środowiska Python 3.10; instalacja PyTorch CUDA w `scripts/setup.sh`.
- `requirements-environment-snapshot.txt`: pełny informacyjny spis środowiska źródłowego, także pakietów niezwiązanych z MVP.
- `third_party/`: źródła CenterPoint, CenterNet2 i Detectron2 wraz z lokalnymi zmianami oraz licencjami. Pochodzenie i rewizje: `SOURCE_MANIFEST.json`.
- `models/manifest.json`: rozmiary i SHA256 dwóch checkpointów. Wagi są lokalnie w `models/`; publikacja umieszcza je w prywatnym wydaniu `models-v1`.
- `scripts/run.py`: przygotowanie danych, trening, walidacja, test jakości i inferencja.
- `scripts/test_mvp_model.py`: metryki nuScenes i raporty JSON/CSV/Markdown.
- `tests/`: testy obsługi metryk i programów uruchomieniowych, bez GPU.
- `results/reference/`: wcześniej zapisane wyniki walidacji checkpointu; mAP **0.637021**, NDS **0.688459**. Nie są wynikiem nowej ewaluacji podczas przygotowania repozytorium.

## Instalacja

Linux, Python **3.10**, GPU NVIDIA, sterownik i CUDA toolkit zgodne z PyTorch CUDA 12.8, kompilator C++ oraz `nvcc` w PATH. Skrypt odtwarza wersje z lokalnego środowiska; czysta instalacja i pełny przebieg GPU wymagają weryfikacji na maszynie docelowej. Nie uruchamiać starego `scripts/setup_env.sh` — pozostaje wyłącznie jako archiwalny skrypt projektu źródłowego.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
source env.sh
```

Ustaw `CUDA_HOME` na katalog toolkitu, jeżeli kompilator nie wykrywa go automatycznie. Główny model ma `dcn_head=False`; instalator buduje wymagane operacje IoU/NMS. Inne architektury w źródłach mogą wymagać dodatkowych rozszerzeń.

Po sklonowaniu prywatnego repozytorium pobierz wagi (GitHub CLI zalogowany do uprawnionego konta):

```bash
bash scripts/download_models.sh
python scripts/verify_models.py
```

## Dane i punkty wirtualne

Pobierz nuScenes trainval z kamerami, LiDAR, sweeps i metadanymi. Dataset pozostaje poza repozytorium. Powiąż go z obiema częściami projektu:

```bash
python scripts/link_data.py /sciezka/do/nuscenes
source env.sh
python scripts/run.py prepare
python virtual_gen.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl MODEL.WEIGHTS models/centernet2_checkpoint.pth
python virtual_gen.py --info_path data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl MODEL.WEIGHTS models/centernet2_checkpoint.pth
python scripts/check_virtual_points.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl --info_path data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl
```

Program `prepare` tworzy informacje o próbkach bez bazy GT, ponieważ aktywna konfiguracja ma `db_sampler=None`. Przy przenoszeniu datasetu wygeneruj pliki infos ponownie, jeśli zawierają stare ścieżki.

## Trening — jedna karta GPU

Domyślnie nowy trening inicjalizuje **wagi** z checkpointu epoki 6, rozpoczynając nowy harmonogram i optymalizator. Nie wznawia zakończonego harmonogramu poprzedniego eksperymentu.

```bash
python scripts/run.py train --epochs 6
# Alternatywnie trening od losowej inicjalizacji:
python scripts/run.py train --from-scratch --epochs 20
```

Domyślny seed: 42. Konfiguracja: `third_party/CenterPoint/configs/mvp/nusc_centerpoint_voxelnet_0075voxel_bevfusion_virtual_ft.py`. Konfiguracja wykonania i checkpointy trafiają do `results/generated/train/`. Parametr `--checkpoint` pozwala wskazać inne wagi inicjalizujące.

## Walidacja, test jakości i inferencja

```bash
python scripts/run.py validate
python scripts/run.py test
python scripts/run.py infer
# Test świeżo wytrenowanego checkpointu:
python scripts/run.py test --checkpoint results/generated/train/latest.pth
# Sam test istniejącego raportu, bez inferencji i GPU:
python scripts/test_mvp_model.py --metrics-only results/reference/metrics_summary.json
# Testy programów bez datasetu i GPU:
python -m unittest discover -s tests -v
```

`validate` oblicza metryki bez minimalnego progu jakości. `test` wymaga mAP >= 0.60 i NDS >= 0.65 na **zbiorze walidacyjnym**. `infer` zapisuje predykcje dla tego samego zbioru oraz uruchamia jego ewaluację. Parametr `--dry-run` pokazuje komendę bez wykonywania obliczeń. Wszystkie te polecenia domyślnie korzystają z załączonego checkpointu epoki 6, nie z ostatniego nowego treningu.

Oficjalny nuScenes test nie udostępnia etykiet. Po podłączeniu datasetu obejmującego `v1.0-test`:

```bash
python scripts/run.py prepare --version v1.0-test
python virtual_gen.py --info_path data/nuScenes/infos_test_10sweeps_withvelo.pkl MODEL.WEIGHTS models/centernet2_checkpoint.pth
python scripts/run.py testset
```

Wynik to predykcje do zgłoszenia na serwer nuScenes, bez lokalnego mAP/NDS dla testu.

## Publikacja na GitHub

Lokalne repozytorium zawiera commit z kodem. Po zainstalowaniu GitHub CLI:

```bash
gh auth login
bash scripts/publish_private.sh
```

Skrypt wymaga konta `mmr0z`, tworzy `mvp-training` jako **prywatne**, ponownie sprawdza widoczność, wysyła kod i oba checkpointy jako zasoby wydania `models-v1`. Nie nadpisuje historii zdalnego repozytorium. Pliki wag są ignorowane przez Git; po klonowaniu pobiera je `download_models.sh`.

## Pochodzenie i licencje

Oryginalny projekt i jego opis: `README.upstream.md`, licencja MIT w `LICENSE`. Każda skopiowana zależność zachowuje własną licencję w `third_party/`. Stan roboczy i lokalne poprawki zachowano w kopii źródeł; aktywna konfiguracja eksportu ma `resume_from=None`, aby umożliwić nowy trening. Dostęp do datasetu nuScenes trzeba zapewnić oddzielnie.
