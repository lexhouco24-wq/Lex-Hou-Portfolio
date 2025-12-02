import os
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# -------------------- Setup -------------------- #
nltk.download("vader_lexicon", quiet=True)
load_dotenv()
API_KEY = os.getenv("GNEWS_API_KEY") or os.getenv("NEWS_API_KEY")
stock_data = []

# -------------------- Helper Functions -------------------- #
def sanitize_query(q: str):
    if not q:
        return None
    q = re.sub(r"[^A-Za-z0-9\s]", " ", q)
    q = " ".join(q.split())
    return q or None

def search_tickers_autocomplete(query: str, limit=5):
    query = sanitize_query(query)
    if not query:
        return []
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        quotes = data.get("quotes", [])
        equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]
        results = [(q.get("symbol"), q.get("shortname", q.get("symbol"))) for q in equities]
        return results[:limit]
    except Exception:
        return []

def search_ticker_by_name(query: str):
    suggestions = search_tickers_autocomplete(query, limit=1)
    if suggestions:
        return suggestions[0][0]
    return None

def format_market_cap(value):
    try:
        value = float(value)
        if value >= 1e12:
            return f"{value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"{value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"{value / 1e6:.2f}M"
        else:
            return f"{value:.2f}"
    except:
        return "N/A"

def calculate_fundamental_score(stock):
    weights = {"pe": 0.3, "market_cap": 0.2, "div_yield": 0.15, "trend": 0.25, "volatility": 0.1}
    info = stock.info
    pe = info.get("trailingPE")
    market_cap = info.get("marketCap")
    div_yield = info.get("dividendYield", 0) or 0
    end = datetime.now()
    start = end - timedelta(days=180)
    hist = stock.history(start=start, end=end)
    if hist.empty:
        return 5.0
    try:
        trend = (hist["Close"][-1] - hist["Close"][0]) / hist["Close"][0]
    except:
        trend = 0
    try:
        volatility = np.std(hist["Close"]) / np.mean(hist["Close"])
    except:
        volatility = 1
    pe_score = 1 - min(pe, 100) / 100 if pe and pe > 0 else 0.5
    mc_score = min((market_cap / 1e9) / 500, 1) if market_cap and market_cap > 0 else 0
    dy_score = min(div_yield / 0.1, 1)
    trend_score = min(max(trend, 0), 0.5) / 0.5
    vol_score = 1 - min(volatility, 0.5) / 0.5
    total_score = (
        pe_score * weights["pe"]
        + mc_score * weights["market_cap"]
        + dy_score * weights["div_yield"]
        + trend_score * weights["trend"]
        + vol_score * weights["volatility"]
    ) * 10
    return round(total_score, 2)

def get_news(query: str, start_date: str, end_date: str):
    if not API_KEY:
        return []
    params = {
        "q": query,
        "lang": "en",
        "from": start_date,
        "to": end_date,
        "max": 50,
        "token": API_KEY,
    }
    try:
        resp = requests.get("https://gnews.io/api/v4/search", params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json() or {}
        return data.get("articles", [])
    except:
        return []

def aggregate_sentiment(articles):
    if not articles:
        return "Neutral", 5
    sia = SentimentIntensityAnalyzer()
    scores = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        content = f"{title}. {description}"
        comp = sia.polarity_scores(content)["compound"]
        scores.append(comp)
    avg_score = sum(scores) / len(scores) if scores else 0
    normalized = (avg_score + 1) * 5
    if avg_score >= 0.75:
        sentiment_label = "Extremely Positive"
    elif avg_score >= 0.5:
        sentiment_label = "Very Positive"
    elif avg_score >= 0.25:
        sentiment_label = "Positive"
    elif avg_score >= 0.05:
        sentiment_label = "Slightly Positive"
    elif avg_score > -0.05:
        sentiment_label = "Neutral"
    elif avg_score > -0.25:
        sentiment_label = "Slightly Negative"
    elif avg_score > -0.5:
        sentiment_label = "Negative"
    elif avg_score > -0.75:
        sentiment_label = "Very Negative"
    else:
        sentiment_label = "Extremely Negative"
    return sentiment_label, round(normalized, 2)

def combine_scores(fundamental_score, sentiment_score):
    return round((fundamental_score * 0.6 + sentiment_score * 0.4), 2)

def investment_label(overall):
    if overall >= 8:
        return "Excellent Investment"
    elif overall >= 6:
        return "Good Investment"
    elif overall >= 4:
        return "Neutral"
    else:
        return "Weak Investment"

# -------------------- Autocomplete Entry -------------------- #
class AutocompleteEntry(tk.Entry):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.listbox_up = False
        self.var = self["textvariable"] = tk.StringVar()
        self.var.trace("w", self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Down>", self.move_down)
        self.bind("<Up>", self.move_up)
        self.lb = None

    def changed(self, *args):
        if self.var.get() == "":
            self.close_listbox()
            return
        words = search_tickers_autocomplete(self.var.get())
        if words:
            if not self.listbox_up:
                self.lb = tk.Listbox(root, width=50, height=min(5, len(words)))
                self.lb.bind("<Double-Button-1>", self.selection)
                self.lb.bind("<Right>", self.selection)
                self.lb.place(x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                self.listbox_up = True
            self.lb.delete(0, tk.END)
            for w in words:
                self.lb.insert(tk.END, f"{w[1]} ({w[0]})")
        else:
            self.close_listbox()

    def selection(self, event=None):
        if self.listbox_up:
            index = self.lb.curselection()
            if index:
                text = self.lb.get(index)
                symbol = re.search(r"\((.*?)\)$", text).group(1)
                self.var.set(symbol)
            self.close_listbox()
            self.icursor(tk.END)

    def move_up(self, event):
        if self.listbox_up:
            if self.lb.curselection() == ():
                index = tk.END
            else:
                index = self.lb.curselection()[0]
            if index > 0:
                self.lb.selection_clear(index)
                index -= 1
                self.lb.selection_set(index)
                self.lb.activate(index)

    def move_down(self, event):
        if self.listbox_up:
            if self.lb.curselection() == ():
                index = -1
            else:
                index = self.lb.curselection()[0]
            if index < self.lb.size() - 1:
                self.lb.selection_clear(index)
                index += 1
                self.lb.selection_set(index)
                self.lb.activate(index)

    def close_listbox(self):
        if self.listbox_up:
            self.lb.destroy()
            self.listbox_up = False

# -------------------- GUI Functions -------------------- #
def fetch_stock_info(symbol_input):
    query = symbol_input.strip()
    if not query:
        return
    symbol = search_ticker_by_name(query)
    if not symbol:
        invalid_label.config(text=f"No stock found for '{query}'")
        return
    for s in stock_data:
        if s["symbol"] == symbol:
            invalid_label.config(text="Stock already in list")
            return
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get("shortName", symbol)
        price = info.get("regularMarketPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w = info.get("fiftyTwoWeekLow", "N/A")
        if any(x == "N/A" or x is None for x in [price, market_cap, pe_ratio]):
            invalid_label.config(text="Incomplete stock data")
            return
        invalid_label.config(text="")
        fundamentals = calculate_fundamental_score(stock)
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")


        search_query = company_name.split()[0] if company_name else symbol
        articles = get_news(search_query, start_date, end_date) or get_news(symbol, start_date, end_date)

        sentiment_label, sentiment_score = aggregate_sentiment(articles)
        overall = combine_scores(fundamentals, sentiment_score)
        stock_data.append({
            "symbol": symbol,
            "company": company_name,
            "price": float(price),
            "market_cap": float(market_cap),
            "pe_ratio": float(pe_ratio),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "overall": overall,
            "fundamentals": fundamentals,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "articles": articles
        })
        update_tree()
    except Exception as e:
        print(e)
        invalid_label.config(text="Failed to fetch stock info")

def update_tree():
    for row in tree.get_children():
        tree.delete(row)
    ranked = sorted(stock_data, key=lambda x: x.get("overall", 0), reverse=True)
    for s in ranked:
        tree.insert("", "end", values=(
            s["symbol"],
            s["company"],
            f"${s['price']:.2f}",
            format_market_cap(s["market_cap"]),
            f"{s['pe_ratio']:.2f}",
            f"{s['low_52w']:.2f} / {s['high_52w']:.2f}"
            if isinstance(s["low_52w"], (int,float)) and isinstance(s["high_52w"], (int,float))
            else "N/A",
            f"{s['overall']:.2f}"
        ))

def on_row_double_click(event):
    selected = tree.selection()
    if not selected:
        return
    item = tree.item(selected[0])
    symbol = item["values"][0]
    stock_entry = next((s for s in stock_data if s["symbol"] == symbol), None)
    if not stock_entry:
        return

    popup = tk.Toplevel(root)
    popup.title(f"{symbol} Details")
    popup.geometry("900x700")

    # Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    stock = yf.Ticker(symbol)
    end = datetime.now()
    start = end - timedelta(weeks=156)
    hist = stock.history(start=start, end=end)
    if not hist.empty:
        ax.plot(hist.index, hist["Close"], label="Close Price", color="dodgerblue")
        ax.set_title(f"{symbol} - 3 Year Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=False)
    toolbar = NavigationToolbar2Tk(canvas, popup)
    toolbar.update()
    toolbar.pack()

    # Sentiment & news
    text = scrolledtext.ScrolledText(popup, width=100, height=20, wrap=tk.WORD)
    text.pack(fill="both", expand=True)
    text.tag_configure("positive", foreground="green")
    text.tag_configure("negative", foreground="red")
    text.tag_configure("neutral", foreground="#00bfff")
    text.tag_configure("bold", font=("Segoe UI", 10, "bold"))

    text.insert(tk.END, f"Company: {stock_entry['company']} ({symbol})\n", "bold")
    text.insert(tk.END, f"Fundamentals Score: {stock_entry['fundamentals']:.2f} / 10\n", "bold")
    text.insert(tk.END, f"Overall Sentiment Score: {stock_entry['sentiment_score']:.2f} / 10 ({stock_entry['sentiment_label']})\n\n", "bold")

    sia = SentimentIntensityAnalyzer()
    if stock_entry["articles"]:
        for article in stock_entry["articles"]:
            title = article.get("title", "No title")
            description = article.get("description", "")
            content = f"{title} - {description}"
            comp = sia.polarity_scores(content)["compound"]
            if comp >= 0.05:
                tag = "positive"
            elif comp <= -0.05:
                tag = "negative"
            else:
                tag = "neutral"
            text.insert(tk.END, f"[{comp:.2f}] {title}\n", tag)
    else:
        text.insert(tk.END, "No news found for this stock.\n", "neutral")

    text.insert(tk.END, f"\nOverall Investment Score: {stock_entry['overall']:.2f} / 10\n", "bold")
    text.insert(tk.END, f"Recommendation: {investment_label(stock_entry['overall'])}\n", "bold")
    text.config(state=tk.DISABLED)

# -------------------- GUI -------------------- #
def main():
    global root, entry_symbol, invalid_label, tree

    root = tk.Tk()
    root.title("📈 Stock Info & Sentiment Analyzer")
    root.configure(bg="#1e1e1e")
    root.geometry("1280x720")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background="#2e2e2e", foreground="white", fieldbackground="#2e2e2e", rowheight=28)
    style.configure("Treeview.Heading", background="black", foreground="white", font=("Segoe UI", 10, "bold"))
    style.configure("Black.TButton", background="black", foreground="white", font=("Segoe UI", 10, "bold"), padding=6)
    style.map("Black.TButton", background=[("active", "#333333")], foreground=[("active", "white")])

    entry_label = tk.Label(root, text="Enter a stock name or ticker:", fg="white", bg="#1e1e1e", font=("Segoe UI", 10, "bold"))
    entry_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,0))

    entry_symbol = AutocompleteEntry(root, font=("Segoe UI", 12), bg="#333", fg="white", insertbackground="white", width=30)
    entry_symbol.grid(row=1, column=0, padx=10, pady=5)

    button_add = ttk.Button(root, text="Add Stock", command=lambda: fetch_stock_info(entry_symbol.get()), style="Black.TButton")
    button_add.grid(row=1, column=1, padx=10, pady=5)

    invalid_label = tk.Label(root, text="", fg="red", bg="#1e1e1e")
    invalid_label.grid(row=1, column=2, padx=10)

    instruction_label = tk.Label(root, text="💡 Double-click a row to view chart & sentiment", fg="lightgray", bg="#1e1e1e", font=("Segoe UI", 10))
    instruction_label.grid(row=2, column=0, columnspan=4, sticky="w", padx=10)

    columns = ("Symbol", "Company", "Price", "Market Cap", "P/E Ratio", "52-Week Range", "Overall Score")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=12)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", stretch=True, width=140)
    tree.grid(row=3, column=0, columnspan=4, padx=10, pady=10)
    tree.bind("<Double-1>", on_row_double_click)
    tree.configure(cursor="hand2")

    root.mainloop()


if __name__ == "__main__":
    main()

