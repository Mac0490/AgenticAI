from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ----------------------------
# State
# ----------------------------
class PendingOrder(TypedDict):
    symbol: str
    quantity: int
    price: float
    total: float


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    pending_order: Optional[PendingOrder]


# ----------------------------
# Tools (demo prices)
# ----------------------------
PRICES = {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}

MAX_QTY = 1000
MAX_NOTIONAL = 50_000


@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol."""
    return float(PRICES.get(symbol.upper(), 0.0))


@tool
def request_buy(symbol: str, quantity: int) -> dict:
    """
    Prepare a buy order and request human approval (HITL).
    Returns structured output.
    """
    sym = symbol.upper()
    qty = int(quantity)

    if qty <= 0:
        return {"ok": False, "error": "Quantity must be > 0."}
    if qty > MAX_QTY:
        return {"ok": False, "error": f"Quantity too large (max {MAX_QTY})."}

    price = float(PRICES.get(sym, 0.0))
    if price <= 0:
        return {"ok": False, "error": f"Unknown symbol: {sym}."}

    total = qty * price
    if total > MAX_NOTIONAL:
        return {"ok": False, "error": f"Notional too large (max ${MAX_NOTIONAL:.2f})."}

    # HITL interrupt
    prompt = (
        f"Approve BUY order?\n"
        f"- Symbol: {sym}\n"
        f"- Quantity: {qty}\n"
        f"- Price: ${price:.2f}\n"
        f"- Total: ${total:.2f}\n"
        f"Type: approve / reject"
    )
    decision = interrupt(prompt)

    decision_norm = str(decision).strip().lower()
    approved = decision_norm in {"approve", "approved", "yes", "y", "ok"}

    if approved:
        return {"ok": True, "status": "filled", "symbol": sym, "quantity": qty, "price": price, "total": total}
    else:
        return {"ok": True, "status": "rejected", "symbol": sym, "quantity": qty, "price": price, "total": total}


tools = [get_stock_price, request_buy]


# ----------------------------
# LLM + system instruction to force tool use
# ----------------------------
SYSTEM = {
    "role": "system",
    "content": (
        "You are a trading assistant.\n"
        "Rules:\n"
        "1) If the user asks for a stock price, call get_stock_price(symbol).\n"
        "2) If the user asks to buy, you MUST call request_buy(symbol, quantity).\n"
        "3) Never claim a trade happened unless request_buy returns status='filled'.\n"
        "4) Keep responses short and factual."
    ),
}

# Use a currently available Gemini model
llm = init_chat_model("google_genai:gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: State):
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}


# ----------------------------
# Build graph
# ----------------------------
memory = MemorySaver()
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)  # routes to ToolNode if tool calls exist
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "buy_thread"}}


# ----------------------------
# Demo run
# ----------------------------
# Step 1: ask price
state = graph.invoke(
    {"messages": [SYSTEM, {"role": "user", "content": "What is the current price of MSFT?"}]},
    config=config,
)
print(state["messages"][-1].content)

# Step 2: ask to buy (will interrupt)
state = graph.invoke(
    {"messages": [SYSTEM, {"role": "user", "content": "Buy 10 MSFT at the current price."}]},
    config=config,
)

intr = state.get("__interrupt__")
if intr:
    print(intr)
    decision = input("Type approve or reject: ").strip()
    state = graph.invoke(Command(resume=decision), config=config)
    print(state["messages"][-1].content)
else:
    print("No approval requested (model did not call request_buy).")
    print(state["messages"][-1].content)