# GPU benchmark on Google Colab (free)

Runs the multi-street solver on a free NVIDIA GPU to confirm the minutes/board
projection (see `docs/multistreet_spike.md`). Your Mac can't run it — CuPy needs
CUDA — so Colab (or any NVIDIA box) is the path.

## Files

- `poker_gpu_benchmark.ipynb` — the notebook. Open it in
  [Colab](https://colab.research.google.com) (File → Upload notebook).
- `poker_solver_gpu.zip` — the code bundle to upload when the notebook asks.
  Rebuild it any time from the repo root with:

  ```bash
  zip -r poker_solver_gpu.zip src bench -x '*/__pycache__/*' '*.pyc'
  ```

## Steps

1. Upload `poker_gpu_benchmark.ipynb` to Colab.
2. **Runtime → Change runtime type → GPU → Save.**
3. Run the cells top to bottom; upload `poker_solver_gpu.zip` when prompted.
4. The notebook prints a GPU vs CPU `ms/iter` table and a correctness check
   (GPU EV must equal CPU EV exactly).

Paste the two tables back to interpret: `ms/iter × 600 / 1000` ≈ seconds/board,
`× 12 / 60` ≈ minutes for a 12-board library.

## No Colab? Any NVIDIA box works

```bash
pip install cupy-cuda12x            # match the box's CUDA (nvidia-smi shows it)
PYTHONPATH=src python bench/gpu_bench.py 3 80 160 250
POKER_BACKEND=numpy PYTHONPATH=src python bench/gpu_bench.py 3 40 80   # CPU baseline
```
