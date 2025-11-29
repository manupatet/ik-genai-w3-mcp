import finnhub
import os
from mcp.server.fastmcp import FastMCP 
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

# --- Configuration ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Please set the FINNHUB_API_KEY environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- MCP Server Definition ---
mcp = FastMCP("finnhub-MCP-server")

# --- PROMPTS (The "Brain" of the operation) ---
@mcp.prompt()
def analyze_stock(ticker: str) -> str:
    """
    Returns the system prompt and instructions for analyzing a specific stock.
    This dictates the agent's behavior and workflow.
    """
    return f"""
    You are a Senior Financial Analyst AI. Your goal is to provide a comprehensive report on the stock ticker: {ticker.upper()}.
    
    You have access to a set of tools to fetch data. You also have access to local visualization tools (if provided by the client environment).
    
    ### EXECUTION PLAN:
    
    1. **Data Gathering**:
       - Call `get_stock_history` to retrieve price action for the last 7 days.
       - Call `get_recommendation_trends` to see what analysts are saying.
       - Call `get_company_news` to find recent context.
       - Call `get_earnings_reports` to check for recent transcripts.
       
    2. **Visualization**:
       - If you successfully retrieved history data, IMMEDIATEY call `plot_historical_price_chart` with that exact data.
       - If you successfully retrieved recommendation trends, IMMEDIATELY call `plot_analyst_recommendations_chart` with that data.
       
    3. **Analysis & Summary**:
       - Review all the data gathered.
       - Generate a "Financial Summary" section summarizing the current state.
       - Generate an "Overall Outlook" section based on the sentiment of the news and analyst ratings.
       - Be professional, neutral, and data-driven. 
       - If any data is missing (returns error), mention it in the report but continue the analysis with what you have.
    """

# --- TOOLS (The "Hands" of the operation) ---

@mcp.tool()
def get_stock_history(ticker: str) -> str:
    """
    Gets last 7 days of historical price for a given stock ticker.
    Returns JSON string with 'dates' and 'prices'.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="7d")
        if history.empty:
            return '{"error": "no history found"}'

        history.reset_index(inplace=True)
        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        prices = history['Close'].tolist()
        import json
        return json.dumps({"dates": dates, "prices": prices})
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'
        
@mcp.tool()
def get_latest_quote(ticker: str) -> dict:
    """Fetches the latest quote for a given stock ticker."""
    try:
        return finnhub_client.quote(ticker)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_company_news(ticker: str) -> list:
    """Fetches recent company news (last 30 days)."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        news = finnhub_client.company_news(ticker, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
        return news[:3]
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_recommendation_trends(ticker: str) -> list:
    """Fetches the latest analyst recommendation trends."""
    try:
        trends = finnhub_client.recommendation_trends(ticker)
        return trends
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_earnings_reports(ticker: str) -> list:
    """Fetches the top 3 most recent earnings call transcripts."""
    try:
        earnings = finnhub_client.earnings_calendar(_from="2025-01-01", to=datetime.now().strftime("%Y-%m-%d"), symbol=ticker, international=False)
        if earnings and 'earningsCalendar' in earnings:
            return sorted(earnings['earningsCalendar'], key=lambda x: x['date'], reverse=True)[:3]
        else:
            return []
    except Exception as e:
        return [{"error": str(e)}]

if __name__ == "__main__":
    mcp.run(transport="stdio")