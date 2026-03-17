import requests
import httpx
import sys
import pytest

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

def test_root():    
    r = requests.get(f"{BASE_URL}/", timeout=5)

    assert r.status_code == 200
    assert "Salut, Fitness Assistant ruleaza!" in r.text

@pytest.mark.asyncio
async def test_chat():    
    payload = {"message": "Arata-mi 3 exercitii pentru picioare, nivel incepator, fara echipament."}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}/chat/", json=payload)

    assert response.status_code == 200
    assert "picioare" in response.text.lower()
    assert "obiectiv de antrenament" in response.text.lower()

@pytest.mark.asyncio
async def test_chat_with_integer():    
    payload = {"message": 3}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}/chat/", json=payload)

    assert response.status_code == 422
