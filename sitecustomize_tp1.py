# TP=1 single-B70 build: install the k8v4 MLP W4A8 prefill hook at interpreter start.
# Upstream installs it from backend.py, which only loads with the TP2-only K8/V4 KV backend.
try:
    from k8v4_v030.w4a8_prefill import install_if_requested
    install_if_requested()
except Exception as e:  # fail loud: a silent fallback would make W4A8 numbers meaningless
    import sys
    sys.stderr.write(f"k8v4_w4a8: installer FAILED: {e!r}\n")
    raise
