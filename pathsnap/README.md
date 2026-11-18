# PathSnap

PathSnap is a GPS map-matching project. It ranks candidate directed road arcs
for noisy GPS observations and is intended to reconstruct routes with a
classifier and an HMM/Viterbi decoder.

## Setup

Use the same Python version on both laptops and create a virtual environment:

```bash
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, LightGBM may need the OpenMP runtime:

```bash
brew install libomp
```

From `pathsnap/`, run the real-data LightGBM tuning after candidate and label
tables are available:

```powershell
python -m src.models.tune --n-trials 20 --n-splits 3
```

Optuna minimizes mean candidate-row log loss across `GroupKFold` splits grouped
by `trip_id`. It only fits on training trips; student-test trips are used once
for the final tuned-versus-baseline comparison. The study database and trial
log are written under `results/optuna/`, the comparison is
`results/optuna/tuning_comparison.csv`, and the tuned bundle/config are written
under `artifacts/`.

# How to run the API

Do not add raw datasets, evaluator answers, virtual environments, or local
environment files to Git.

## Server: Laptop 1

For live GPS observations, set the raw dataset root to a directory containing
split-specific trip folders:

```text
<data-root>/
  train/<trip-id>/<trip-id>.track
  train/<trip-id>/<trip-id>.nodes
  train/<trip-id>/<trip-id>.arcs
  train/<trip-id>/<trip-id>.route
  student_test/<trip-id>/<trip-id>.track
  ...
```

Set `PATHSNAP_DATA_DIR` to the directory that contains `train/` and
`student_test/`. The reference `.route` is only loaded by the internal loader
when explicitly requested; API prediction endpoints never request or return it.

Inference uses the committed Member 2 bundle at
`artifacts/model_bundle.joblib` by default. It also needs Member 1's
per-trip features, road geometry, and CRS audit. Override their locations with
`PATHSNAP_FEATURES_DIR`, `PATHSNAP_ROAD_GEOMETRY_DIR`, and
`PATHSNAP_CRS_AUDIT` if needed. These inputs are used only for the permitted
`student_test` trip IDs; test labels and reference routes are not loaded.

Raw `.track` files are optional for prediction: without them, the API can still
return classifier/HMM routes from the precomputed features, but the UI cannot
show the original GPS layer. If you train a replacement bundle, set
`PATHSNAP_MODEL_PATH` to its location. For example:

```bash
python -m tools.train_api_bundle
export PATHSNAP_MODEL_PATH="$PWD/artifacts/classifier_bundle.joblib"
```

The generated `.joblib` stays local and is ignored by Git.

```bash
export PATHSNAP_DATA_DIR="$PWD/data"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Check the service from Laptop 1:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/trips
```

`/health` reports the status of raw GPS data, precomputed candidate inputs,
the classifier, and the HMM decoder separately. `/trips` lists only IDs in
the `student_test` split. Missing prediction inputs return HTTP 503 rather
than mock predictions.

## Client: Laptop 2

Find Laptop 1's Wi-Fi IP on macOS:

```bash
ipconfig getifaddr en0
```

If that returns no address, inspect `ifconfig` and use the active Wi-Fi
interface. Both laptops must be on the same network, and Laptop 1's firewall
must allow inbound TCP port 8000.

Test from Laptop 2, replacing the address with Laptop 1's LAN IP:

```bash
curl --connect-timeout 5 http://<laptop-1-ip>:8000/health
```

Start the Streamlit client with that URL:

```bash
PATHSNAP_API_URL="http://<laptop-1-ip>:8000" \
  streamlit run ui/app.py
```

The API URL can also be edited in the Streamlit sidebar. The UI calls FastAPI
over HTTP and reports connection failures and request timeouts.

## API

- `GET /` — service status
- `GET /health` — readiness of runtime dependencies
- `GET /trips` — permitted student-test trip IDs
- `GET /trips/{trip_id}` — GPS observations only; never returns the reference
  route
- `POST /evaluate/{trip_id}` — classifier/HMM evaluation when the required
  artifacts and integrations are available

Interactive API documentation is served at `/docs`.

## Tests

Run from this directory so `src` imports resolve:

```bash
python -m pytest -q
```

## Current integration status

The repository includes Member 1 feature and road-geometry outputs, Member 2's
trained model bundle, and Member 3's HMM decoder. Raw trip files are still
needed for the GPS observation layer; the API does not infer or fabricate
missing GPS coordinates or test metrics.
