import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from contextlib import AsyncExitStack
import json
import os
from datetime import datetime
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

# --- Load configuration ---
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise ValueError("Please set the OPENROUTER_KEY environment variable.")

llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",## Need tool-calling abilities in the model you choose
    openai_api_key=OPENROUTER_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
)

# --- Local Visualization Tools (Plotly for rich interactivity) ---
def plot_price_history(data_json: str) -> go.Figure | None:
    """Creates an interactive candlestick-style price chart."""
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        if "error" in data or not data.get("dates"):
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["dates"], y=data["prices"],
            mode="lines+markers",
            name="Close Price",
            line=dict(color="#00d4aa", width=3),
            marker=dict(size=8, color="#00d4aa"),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)"
        ))
        fig.update_layout(
            title=dict(text="7-Day Price History", font=dict(size=18, color="#e0e0e0")),
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_dark",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            hovermode="x unified",
            margin=dict(l=60, r=40, t=60, b=40)
        )
        return fig
    except Exception:
        return None

def plot_analyst_recommendations(data_json: str) -> go.Figure | None:
    """Creates a radial/bar chart for analyst recommendations."""
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        trend = data[0] if isinstance(data, list) and data else data
        
        labels = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
        values = [trend.get(k, 0) for k in ["strongSell", "sell", "hold", "buy", "strongBuy"]]
        colors = ["#ff4757", "#ff6b81", "#ffa502", "#7bed9f", "#2ed573"]
        
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
            text=values, textposition="outside"
        ))
        fig.update_layout(
            title=dict(text="Analyst Recommendations", font=dict(size=18, color="#e0e0e0")),
            xaxis_title="Rating", yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            margin=dict(l=60, r=40, t=60, b=40)
        )
        return fig
    except Exception:
        return None

LOCAL_TOOLS = {
    "plot_historical_price_chart": plot_price_history,
    "plot_analyst_recommendations_chart": plot_analyst_recommendations,
}

# --- MCP Tool Discovery & Execution ---
def mcp_tools_to_langchain_schema(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to LangChain/OpenAI function schema."""
    schemas = []
    for tool in mcp_tools:
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
            }
        }
        schemas.append(schema)
    
    # Add local visualization tools
    schemas.extend([
        {
            "type": "function",
            "function": {
                "name": "plot_historical_price_chart",
                "description": "Plots a 7-day price history chart from JSON data with dates and prices.",
                "parameters": {"type": "object", "properties": {"data_json": {"type": "string"}}, "required": ["data_json"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "plot_analyst_recommendations_chart", 
                "description": "Plots analyst recommendations from JSON data with strongSell, sell, hold, buy, strongBuy.",
                "parameters": {"type": "object", "properties": {"data_json": {"type": "string"}}, "required": ["data_json"]}
            }
        }
    ])
    return schemas

async def analyze_stock(ticker: str):
    """Main analysis loop - discovers capabilities, executes tools, generates report."""
    if not ticker:
        return None, "Please enter a ticker symbol.", ""
    
    async with AsyncExitStack() as stack:
        try:
            # 1. Connect to MCP Server
            server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            # 2. Discover Server Capabilities
            tools_response = await session.list_tools()
            prompts_response = await session.list_prompts()
            
            print(f"Discovered {len(tools_response.tools)} server tools:")
            for t in tools_response.tools:
                print(f"   - {t.name}: {t.description[:50] if t.description else 'No description'}...")
            
            # 3. Get System Prompt from Server
            prompt_result = await session.get_prompt("analyze_stock", arguments={"ticker": ticker})
            system_instruction = prompt_result.messages[0].content.text

            # 4. Prepare LLM with Tools
            tool_schemas = mcp_tools_to_langchain_schema(tools_response.tools)
            llm_with_tools = llm.bind_tools(tool_schemas)
            
            # 5. Agentic Execution Loop
            messages = [HumanMessage(content=system_instruction)]
            captured_figures = []
            collected_data = {}  # Store fetched data for visualization
            max_iterations = 15
            
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}")
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                # Debug: Show response info
                print(f"   Response type: {type(response).__name__}")
                print(f"   Content preview: {str(response.content)[:200]}...")
                print(f"   Tool calls: {response.tool_calls}")
                
                # Check for tool calls
                if not response.tool_calls:
                    print("   No tool calls, ending loop")
                    break
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    fn_name = tool_call["name"]
                    fn_args = tool_call["args"]
                    print(f"   Calling: {fn_name}({fn_args})")
                    
                    # Route to local or remote tool
                    if fn_name in LOCAL_TOOLS:
                        data_arg = fn_args.get("data_json", "{}")
                        print(f"   Local tool with data: {data_arg[:100]}...")
                        result = LOCAL_TOOLS[fn_name](data_arg)
                        if isinstance(result, go.Figure):
                            captured_figures.append(result)
                            tool_result = "Chart generated successfully."
                            print(f"   Chart captured! Total figures: {len(captured_figures)}")
                        else:
                            tool_result = "Failed to generate chart."
                            print(f"   Chart generation failed")
                    else:
                        # Call MCP server tool
                        mcp_result = await session.call_tool(name=fn_name, arguments=fn_args)
                        tool_result = "\n".join(c.text for c in mcp_result.content if hasattr(c, "text"))
                        print(f"   MCP result: {tool_result[:200]}...")
                        
                        # Cache data for potential manual visualization
                        if fn_name == "get_stock_history":
                            collected_data["history"] = tool_result
                        elif fn_name == "get_recommendation_trends":
                            collected_data["recommendations"] = tool_result
                    
                    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
            
            # Fallback: If LLM didn't call visualization tools, do it ourselves
            if not captured_figures and collected_data:
                print("\n Fallback: Generating charts from collected data...")
                if "history" in collected_data:
                    fig = plot_price_history(collected_data["history"])
                    if fig:
                        captured_figures.append(fig)
                        print("   Price history chart generated")
                if "recommendations" in collected_data:
                    fig = plot_analyst_recommendations(collected_data["recommendations"])
                    if fig:
                        captured_figures.append(fig)
                        print("   Recommendations chart generated")

            # 6. Extract Final Report
            final_text = response.content if isinstance(response.content, str) else str(response.content)
            
            # 7. Combine Figures
            if len(captured_figures) > 1:
                combined = make_subplots(rows=len(captured_figures), cols=1, subplot_titles=[
                    f.layout.title.text if f.layout.title else "" for f in captured_figures
                ], vertical_spacing=0.08)
                for i, fig in enumerate(captured_figures, 1):
                    for trace in fig.data:
                        combined.add_trace(trace, row=i, col=1)
                combined.update_layout(
                    height=400 * len(captured_figures),
                    template="plotly_dark",
                    paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                    showlegend=False
                )
                final_plot = combined
            elif captured_figures:
                final_plot = captured_figures[0]
            else:
                final_plot = None
            
            return final_plot, f" Analysis complete for {ticker.upper()}", final_text

        except Exception as e:
            traceback.print_exc()
            return None, f"Error: {str(e)}", ""

CUSTOM_CSS = """
.gradio-container { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
.gr-button-primary { background: linear-gradient(90deg, #00d4aa, #00b894) !important; border: none !important; }
.gr-button-primary:hover { transform: scale(1.02); box-shadow: 0 4px 20px rgba(0, 212, 170, 0.4); }
footer { display: none !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="teal")) as demo:
    gr.Markdown("""
    # Investment Advisor
    ### *OpenRouter + LangChain Edition*
    > Dynamically discovers server capabilities • Interactive Plotly visualizations • Decoupled architecture
    """)
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="NVDA, AAPL, TSLA...", scale=3)
        analyze_button = gr.Button("🔍 Analyze", variant="primary", scale=1)
    
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Visual Report")
    summary_output = gr.Markdown(label="Financial Report")

    analyze_button.click(
        analyze_stock,
        inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()
