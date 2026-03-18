import requests
import httpx
import sys
import pytest

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"