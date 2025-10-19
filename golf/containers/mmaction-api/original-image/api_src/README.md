Running the ST-GCN test subprocess with GPU

This folder contains the API server and helper scripts used to run mmaction2 ST-GCN tests in a subprocess.

How to enable GPU for the subprocess

- Set CUDA_VISIBLE_DEVICES before starting the API server or export it in the environment used by the API. Example (Linux):

  export CUDA_VISIBLE_DEVICES=0

- Alternatively, set the environment variable `MMACTION_DEVICE` to a device string (e.g. `cuda`, `cuda:0`, or `cpu`) and the tester will forward it to the subprocess.

- The tester also accepts `--device` and `--cuda-devices` flags directly when calling `stgcn_subproc.py` (internal use by the API).

Notes

- The subprocess logs torch.cuda availability and the chosen device to stderr.
- If CUDA is unavailable, the subprocess will fall back to CPU.
