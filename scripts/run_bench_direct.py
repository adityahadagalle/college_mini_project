import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

try:
    print("STARTING BENCHMARK SCRIPT", flush=True)
    import scripts.benchmark_phase6 as b
    import asyncio
    print("CALLING b.main()", flush=True)
    asyncio.run(b.main())
    print("FINISHED b.main()", flush=True)
except Exception as e:
    print(f"CAUGHT TOP-LEVEL EXCEPTION: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
