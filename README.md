# Ultrasound Line Scanning Agent (ULSA)

## Setup

```bash
git submodule update --init --recursive
pip install -e zea
cp .env.example .env
touch users.yaml # edit!
```

## Fonts

```bash
cp styles/times.ttf /usr/local/share/fonts
fc-cache -fv
rm -fr ~/.cache/matplotlib
```
