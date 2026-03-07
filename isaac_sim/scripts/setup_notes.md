# Isaac Sim / Isaac Lab Setup Notes

## Environment
- conda env: `isaac_sim` (Python 3.11)
- Isaac Sim: 5.1.0 (pip install)
- Isaac Lab: 0.54.3 (source install from ~/projects/IsaacLab)
- PyTorch: 2.12.0.dev (nightly cu128 — required for RTX 5060 Blackwell sm_120)

## Install Commands
```bash
conda create -n isaac_sim python=3.11 -y
conda activate isaac_sim
pip install --upgrade pip
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

cd ~/projects
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
```

## Franka USD Path (verified working)
```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
```

## Known Issues
- RTX 5060 Blackwell: use `--renderer PathTracing` to avoid blurry rendering bug
- Always launch with `os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'`
