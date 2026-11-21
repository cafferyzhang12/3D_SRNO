#!/bin/bash
# Download the upstream code and pretrained weights. No MRI inference or training here.
set -euo pipefail
name="${1:?Specify swinir or res_srdiff}"
root="${MRI_EXTERNAL_ROOT:-$PWD/references}"
py="${GSBM_PYTHON:-$HOME/.conda/envs/fundiff/bin/python}"
mkdir -p "$root/weights"
case "$name" in
    swinir)
        repo=JingyunLiang/SwinIR
        revision=6545850fbf8df298df73d81f3e8cba638787c8bd
        ;;
    res_srdiff)
        repo=mosaf/Res-SRDiff
        revision=2a0e0a8c6de924f61f496f1eaad9557038971c99
        ;;
    *) echo "Expected swinir or res_srdiff" >&2; exit 2 ;;
esac
checkout="$root/${repo#*/}"
if [[ ! -e "$checkout" ]]; then
    git clone "https://github.com/$repo.git" "$checkout"
    git -C "$checkout" checkout --detach "$revision"
fi
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]]
if [[ ! -e "$root/weights/$name.pth" ]]; then
    if [[ "$name" == swinir ]]; then
        curl --fail --location --retry 3 --continue-at - -o "$root/weights/$name.pth.partial" \
            https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
    else
        "$py" -m gdown --continue 1_lcg1r3vqluiBwmgFABMqBZP5B07THnf -O "$root/weights/$name.pth.partial"
    fi
    mv -- "$root/weights/$name.pth.partial" "$root/weights/$name.pth"
fi
