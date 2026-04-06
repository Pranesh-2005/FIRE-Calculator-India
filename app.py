from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import redis
import os
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq
import random
import json
import numpy as np

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------ CONFIG ------------------

GROQ_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
    ssl=True
)

client = Groq(api_key=GROQ_API_KEY)

# ------------------ MODELS ------------------

class FireInput(BaseModel):
    age: int
    monthly_expenses: float
    savings: float
    monthly_investment: float
    expected_return: float = 12
    inflation: float = 6

class ChatInput(BaseModel):
    age: int
    monthly_expenses: float
    savings: float
    question: str

# ------------------ FIRE ENGINE ------------------

def future_expense(expense, inflation, years):
    return expense * ((1 + inflation/100) ** years)

def calculate_fire(data: FireInput):
    years = 0
    corpus = data.savings

    while years < 60:
        corpus += data.monthly_investment * 12
        corpus *= (1 + data.expected_return / 100)

        inflated_expense = future_expense(
            data.monthly_expenses * 12,
            data.inflation,
            years
        )

        target = inflated_expense * 30  # safer India

        if corpus >= target:
            break

        years += 1

    return {
        "years_to_fire": years,
        "fire_age": data.age + years,
        "corpus": round(corpus, 2),
        "target": round(target, 2)
    }

# ------------------ MONTE CARLO ------------------

def monte_carlo_fire(data: FireInput, simulations=1000):
    success = 0
    results = []

    for _ in range(simulations):
        years = 0
        corpus = data.savings

        while years < 60:
            yearly_return = random.normalvariate(data.expected_return, 3)
            yearly_return = max(-5, min(20, yearly_return))

            inflation = random.normalvariate(data.inflation, 1)
            inflation = max(3, min(10, inflation))

            corpus += data.monthly_investment * 12
            corpus *= (1 + yearly_return / 100)

            inflated_expense = (
                data.monthly_expenses * 12 *
                ((1 + inflation/100) ** years)
            )

            target = inflated_expense * 30

            if corpus >= target:
                success += 1
                results.append(years)
                break

            years += 1

    success_rate = (success / simulations) * 100
    results.sort()

    return {
        "success_rate": round(success_rate, 2),
        "median_years": results[len(results)//2] if results else None,
        "best_case": min(results) if results else None,
        "worst_case": max(results) if results else None
    }

# ------------------ GRAPH: GROWTH ------------------

def generate_growth_chart(data: FireInput, years=40):
    age_list, corpus_list, target_list = [], [], []
    corpus = data.savings

    for year in range(years):
        age = data.age + year

        corpus += data.monthly_investment * 12
        corpus *= (1 + data.expected_return / 100)

        inflated_expense = (
            data.monthly_expenses * 12 *
            ((1 + data.inflation / 100) ** year)
        )

        target = inflated_expense * 30

        age_list.append(age)
        corpus_list.append(round(corpus, 2))
        target_list.append(round(target, 2))

    return {
        "age": age_list,
        "corpus": corpus_list,
        "target": target_list
    }

# ------------------ GRAPH: MONTE ------------------

def monte_carlo_distribution(data: FireInput, simulations=1000):
    results = []

    for _ in range(simulations):
        years = 0
        corpus = data.savings

        while years < 60:
            yearly_return = random.normalvariate(data.expected_return, 3)
            yearly_return = max(-5, min(20, yearly_return))

            inflation = random.normalvariate(data.inflation, 1)
            inflation = max(3, min(10, inflation))

            corpus += data.monthly_investment * 12
            corpus *= (1 + yearly_return / 100)

            inflated_expense = (
                data.monthly_expenses * 12 *
                ((1 + inflation/100) ** years)
            )

            target = inflated_expense * 30

            if corpus >= target:
                results.append(data.age + years)
                break

            years += 1

    hist, bin_edges = np.histogram(results, bins=10)

    return {
        "bins": bin_edges.tolist(),
        "frequency": hist.tolist()
    }

# ------------------ GRAPH: SCENARIO ------------------

def scenario_comparison(data: FireInput):
    scenarios = [0.8, 1.0, 1.2, 1.5]
    result = {}

    for factor in scenarios:
        modified = FireInput(
            age=data.age,
            monthly_expenses=data.monthly_expenses,
            savings=data.savings,
            monthly_investment=data.monthly_investment * factor,
            expected_return=data.expected_return,
            inflation=data.inflation
        )

        result[str(int(factor * 100)) + "%"] = calculate_fire(modified)

    return result

# ------------------ TAX ------------------

def equity_tax(gains):
    if gains <= 100000:
        return 0
    return (gains - 100000) * 0.10

# ------------------ MARKET ------------------

def get_nifty_data():
    cache_key = "nifty_yahoo_v3"

    # ✅ Try cache
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}
    except:
        pass

    try:
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="3mo", interval="1d")

        # ❌ Handle empty data
        if hist.empty:
            return {"error": "No data from Yahoo Finance"}

        prices = hist["Close"]

        # ✅ Core data
        dates = hist.index.strftime("%Y-%m-%d").tolist()
        prices_list = [round(x, 2) for x in prices.tolist()]

        returns = [
            round(((prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]) * 100, 2)
            if i > 0 else 0
            for i in range(len(prices))
        ]

        # ✅ Moving average (7-day)
        moving_avg = [
            round(x, 2) if not np.isnan(x) else None
            for x in prices.rolling(7).mean()
        ]

        # ✅ Drawdown
        running_max = prices.cummax()
        drawdown = [
            round(((p - m) / m) * 100, 2)
            for p, m in zip(prices, running_max)
        ]

        # ✅ Summary stats
        change_percent = round(
            ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100, 2
        )

        volatility = round(np.std(returns), 2)

        data = {
            "dates": dates,
            "prices": prices_list,
            "returns": returns,
            "moving_avg_7": moving_avg,
            "drawdown": drawdown,
            "summary": {
                "latest_price": prices_list[-1],
                "change_percent": change_percent,
                "volatility": volatility
            }
        }

        # ✅ Cache valid data
        try:
            redis_client.setex(cache_key, 3600, json.dumps(data))
        except:
            pass

        return {"source": "api", "data": data}

    except Exception as e:
        return {"error": str(e)}        
# ------------------ LLM ------------------

def build_prompt(user, question):
    return f"""
You are an Indian financial advisor.

User:
Age: {user['age']}
Monthly Expense: ₹{user['monthly_expenses']}
Savings: ₹{user['savings']}
Use only 150 or fewer tokens.
Give short, practical advice.

Question: {question}
"""

def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ better model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful Indian financial advisor."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=250,
            temperature=0.7
        )

        content = response.choices[0].message.content

        if not content:
            return "⚠️ No response from AI"

        return content

    except Exception as e:
        return f"LLM error: {str(e)}"
# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"msg": "FIRE Backend Running 🇮🇳"}

@app.get("/kaithhealthcheck")
def kaithhealthcheck():
    return {"msg": "Kaith Health Check Passed"}

@app.get("/kaithheathcheck")
def kaithheathcheck():
    return {"msg": "Kaith Heath Check Passed"}

@app.post("/fire")
def fire(data: FireInput):
    return calculate_fire(data)

@app.post("/monte-carlo")
def monte(data: FireInput):
    return monte_carlo_fire(data)

@app.post("/graph/growth")
def growth(data: FireInput):
    return generate_growth_chart(data)

@app.post("/graph/monte")
def monte_graph(data: FireInput):
    return monte_carlo_distribution(data)

@app.post("/graph/scenario")
def scenario(data: FireInput):
    return scenario_comparison(data)

@app.get("/market")
def market():
    return get_nifty_data()

@app.post("/chat")
def chat(data: ChatInput):
    user = {
        "age": data.age,
        "monthly_expenses": data.monthly_expenses,
        "savings": data.savings
    }

    prompt = build_prompt(user, data.question)
    return {"answer": ask_llm(prompt)}

@app.get("/tax")
def tax(gains: float):
    return {"tax": equity_tax(gains)}