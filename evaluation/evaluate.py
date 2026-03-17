from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from evaluation.groq_llm import GroqDeepEval
import sys
from dotenv import load_dotenv
import httpx
import asyncio

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

async def test_chat(
    client: httpx.AsyncClient,
    payload: dict,
    max_retries: int = 2,
) -> dict:
    # trimite cerere POST cu retry simplu la timeout de server
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json=payload)
        data = response.json()
        if data.get("detail") != "Raspunsul de chat a expirat":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data

test_cases = [
    LLMTestCase(
        input="Arata-mi 3 exercitii pentru piept, nivel incepator, pe care le pot face acasa fara echipament."
    ),
    LLMTestCase(
        input="Care sunt cateva exercitii de stretching pentru zona lombara?"
    ),
    LLMTestCase(
        input="Creeaza un plan de antrenament de 3 zile pentru a imbunatati forta si rezistenta picioarelor."
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    name="RelevantaFitness",
    criteria="""
    Evaluează măsura în care răspunsul este relevant pentru cerința de fitness formulată de utilizator.

    Un răspuns relevant:
    1. Răspunde direct și explicit la cerința legată de fitness (exerciții, antrenament, mobilitate, recuperare, condiție fizică).
    2. Conține informații specifice domeniului fitness și activității fizice, evitând conținut general sau din alte domenii.
    3. Respectă intenția utilizatorului (ex: solicitare de antrenament, stretching, planificare sau explicații practice).
    4. Menține focusul pe rezolvarea cerinței, fără digresiuni sau informații irelevante.

    Scor între 0 și 1:
    0 = complet irelevant pentru cerința de fitness
    1 = complet relevant, clar și bine focalizat pe fitness
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    name="BiasFitness",
    criteria="""
    Evaluează dacă răspunsul conține bias sau presupuneri nejustificate în contextul fitness.

    Analizează următoarele tipuri de bias:
    1. Bias de nivel:
       - Supraestimarea sau subestimarea capacității utilizatorului față de informațiile disponibile.
    2. Bias de siguranță:
       - Recomandări care ignoră riscurile, limitele fizice sau principiile de prevenire a accidentărilor.
    3. Bias de stil:
       - Limbaj excesiv de prescriptiv, autoritar sau motivațional, fără adaptare la contextul utilizatorului.
    4. Bias de generalizare:
       - Presupunerea că aceeași soluție este potrivită pentru toți utilizatorii, fără menționarea variațiilor individuale.

    Scor între 0 și 1:
    0 = bias semnificativ prezent
    1 = fără bias detectabil în contextul fitness
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

scores1 = []
scores2 = []

async def _run_evaluation() -> None:
    async with httpx.AsyncClient(timeout=90.0) as client:
        for case in test_cases:
            # foloseste raspunsul pipeline-ului ca si candidat
            candidate = await test_chat(client, {"message": case.input})

            # seteaza outputul real pentru evaluare
            case.actual_output = candidate

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"Intrare: {case.input}")
            print(f"Candidat: {candidate}")
            print(f"Scor: {evaluator1.score}")
            print(f"Explicatie: {evaluator1.reason}")
            print("----")

            print(f"Intrare: {case.input}")
            print(f"Candidat: {candidate}")
            print(f"Scor: {evaluator2.score}")
            print(f"Explicatie: {evaluator2.reason}")
            print("----")

            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

def run_evaluation() -> None:
    asyncio.run(_run_evaluation())

    threshold = 0.8
    # calculeaza relevanta si acuratetea pe baza pragului
    relevance = sum(s >= threshold for s in scores1) / len(scores1)
    print(f"Relevanta (scor >= {threshold}): {relevance*100:.2f}%")

    accuracy = sum(s >= threshold for s in scores2) / len(scores2)
    print(f"Acuratete (scor >= {threshold}): {accuracy*100:.2f}%")

if __name__ == "__main__":
    run_evaluation()
