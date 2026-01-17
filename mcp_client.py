import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from contextlib import AsyncExitStack
import json
import os
from typing import Callable

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

load_dotenv()

# --- Configuration ---
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise ValueError("Please set the OPENROUTER_KEY environment variable.")

llm = ChatOpenAI(model="mistralai/devstral-2512:free", openai_api_key=OPENROUTER_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.0)

# VISUALIZATION REGISTRY - Self-describing local tools
def render_price_chart(data: dict) -> go.Figure | None:
    """Renders a price history line chart."""
    try:
        if "error" in data or not data.get("dates"):
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["dates"], y=data["prices"],
            mode="lines+markers", name="Close Price",
            line=dict(color="#00d4aa", width=3),
            marker=dict(size=8, color="#00d4aa"),
            fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)"
        ))
        fig.update_layout(
            title=dict(text="Price History", font=dict(size=18, color="#e0e0e0")),
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_dark",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            hovermode="x unified", margin=dict(l=60, r=40, t=60, b=40)
        )
        return fig
    except Exception:
        return None


def render_recommendations_chart(data: dict) -> go.Figure | None:
    """Renders an analyst recommendations bar chart."""
    try:
        # Handle list format from server
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

# Registry: Maps visualization types to their requirements and renderers
VISUALIZATION_REGISTRY: dict[str, dict] = {
    "price_chart": {
        "description": "Line chart showing price history over time",
        "required_fields": ["dates", "prices"],
        "field_types": {"dates": "list[str]", "prices": "list[float]"},
        "renderer": render_price_chart,
    },
    "recommendations_chart": {
        "description": "Bar chart showing analyst recommendation distribution",
        "required_fields": ["strongBuy", "buy", "hold", "sell", "strongSell"],
        "field_types": {k: "int" for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]},
        "renderer": render_recommendations_chart,
    },
}

# CAPABILITY DISCOVERY & MAPPING
def discover_visualization_capabilities() -> str:
    """Introspects local visualization registry and returns description for the agent."""
    descriptions = []
    for viz_name, viz_info in VISUALIZATION_REGISTRY.items():
        fields = ", ".join(f"`{f}` ({viz_info['field_types'][f]})" for f in viz_info["required_fields"])
        descriptions.append(f"- **{viz_name}**: {viz_info['description']}. Requires: {fields}")
    return "\n".join(descriptions)


def describe_server_tools(tools: list) -> str:
    """Creates a description of discovered server tools for the agent."""
    descriptions = []
    for tool in tools:
        params = ""
        if tool.inputSchema and tool.inputSchema.get("properties"):
            params = ", ".join(f"`{p}`" for p in tool.inputSchema["properties"].keys())
        descriptions.append(f"- **{tool.name}**({params}): {tool.description or 'No description'}")
    return "\n".join(descriptions)


def infer_capability_mapping(server_tools: list) -> str:
    """
    Analyzes server tools and suggests which can feed which visualizations.
    This is based on description/name heuristics since we can't inspect output schemas.
    """
    mappings = []
    for tool in server_tools:
        name_lower = tool.name.lower()
        desc_lower = (tool.description or "").lower()
        
        # Match based on semantic hints
        if "history" in name_lower or "price" in desc_lower:
            mappings.append(f"- `{tool.name}` → `price_chart` (provides dates/prices)")
        if "recommendation" in name_lower or "analyst" in desc_lower:
            mappings.append(f"- `{tool.name}` → `recommendations_chart` (provides buy/hold/sell counts)")
    
    return "\n".join(mappings) if mappings else "No automatic mappings detected. Inspect tool outputs manually."

# CLIENT-SIDE AGENT PROMPT (The "Plan" lives here, not on server)
def build_analyst_prompt(ticker: str, server_tools_desc: str, viz_desc: str, mapping_desc: str) -> str:
    """Builds the agent's system prompt with discovered capabilities."""
    return f"""You are a Senior Finance & Investment Analyst. Your task is to analyze **{ticker.upper()}**.

## Available Data Sources (MCP Server)
{server_tools_desc}

## Available Visualizations (Local)
{viz_desc}

## Suggested Data → Visualization Mappings
{mapping_desc}

## Your Workflow

1. **Gather Data**: Call relevant server tools to fetch financial data for {ticker}.
   
2. **Visualize**: For each successful data fetch, call the matching visualization tool:
   - `render_price_chart(dates, prices)` - Pass the dates and prices arrays directly
   - `render_recommendations_chart(strongBuy, buy, hold, sell, strongSell)` - Pass the counts directly
   
3. **Analyze**: After gathering data, provide a professional investment report with:
   - **Financial Summary**: Current price, recent trend, analyst sentiment
   - **Investment Outlook**: Bull/bear case based on data
   - **Key Risks**: Any concerns from news or earnings

Be data-driven and professional. If a tool returns an error, note it and continue with available data.
"""

# Tool Creation
def create_server_tool(tool_def, session) -> StructuredTool:
    """Creates a LangChain StructuredTool from an MCP tool definition."""
    
    async def call_mcp_tool(**kwargs) -> str:
        result = await session.call_tool(name=tool_def.name, arguments=kwargs)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))
    
    # Build Pydantic schema from MCP inputSchema
    fields = {}
    if tool_def.inputSchema and tool_def.inputSchema.get("properties"):
        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
        for prop, details in tool_def.inputSchema["properties"].items():
            py_type = type_map.get(details.get("type", "string"), str)
            is_required = prop in tool_def.inputSchema.get("required", [])
            if is_required:
                fields[prop] = (py_type, Field(description=details.get("description", "")))
            else:
                fields[prop] = (py_type | None, Field(default=None, description=details.get("description", "")))
    
    ArgsSchema = create_model(f"{tool_def.name.title().replace('_', '')}Args", **fields) if fields else None
    
    return StructuredTool.from_function(
        coroutine=call_mcp_tool,
        name=tool_def.name,
        description=tool_def.description or f"MCP tool: {tool_def.name}",
        args_schema=ArgsSchema,
    )

# Global figure storage for visualization tools
captured_figures: list[go.Figure] = []

def create_visualization_tools() -> list[StructuredTool]:
    """Creates LangChain tools for local visualizations."""
    global captured_figures
    
    # Price chart tool
    class PriceChartInput(BaseModel):
        dates: list[str] = Field(description="List of dates in YYYY-MM-DD format")
        prices: list[float] = Field(description="List of closing prices")
    
    def render_price(dates: list[str], prices: list[float]) -> str:
        fig = render_price_chart({"dates": dates, "prices": prices})
        if fig:
            captured_figures.append(fig)
            return "Price chart rendered successfully"
        return "Failed to render price chart"
    
    # Recommendations chart tool
    class RecommendationsInput(BaseModel):
        strongBuy: int = Field(default=0, description="Strong buy count")
        buy: int = Field(default=0, description="Buy count")
        hold: int = Field(default=0, description="Hold count")
        sell: int = Field(default=0, description="Sell count")
        strongSell: int = Field(default=0, description="Strong sell count")
    
    def render_recs(strongBuy: int = 0, buy: int = 0, hold: int = 0, sell: int = 0, strongSell: int = 0) -> str:
        fig = render_recommendations_chart({
            "strongBuy": strongBuy, "buy": buy, "hold": hold, "sell": sell, "strongSell": strongSell
        })
        if fig:
            captured_figures.append(fig)
            return "Recommendations chart rendered successfully"
        return "Failed to render recommendations chart"
    
    return [
        StructuredTool.from_function(func=render_price, name="render_price_chart",
            description="Renders a price history chart. Pass dates and prices arrays.",
            args_schema=PriceChartInput),
        StructuredTool.from_function(func=render_recs, name="render_recommendations_chart",
            description="Renders analyst recommendations chart. Pass buy/hold/sell counts.",
            args_schema=RecommendationsInput),
    ]

# Main Analysis
async def analyze_stock(ticker: str):
    """Main analysis workflow with dynamic capability discovery."""
    if not ticker:
        return None, "Please enter a ticker symbol.", ""
    
    global captured_figures
    captured_figures = []
    
    async with AsyncExitStack() as stack:
        try:
            # 1. Connect to MCP Server
            server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()
            
            # 2. Discover Server Capabilities
            tools_response = await session.list_tools()
            print(f"\nDiscovered {len(tools_response.tools)} server tools:")
            for t in tools_response.tools:
                print(f"   • {t.name}")
            
            # 3. Introspect Local Capabilities
            viz_description = discover_visualization_capabilities()
            print(f"\nLocal visualization capabilities:\n{viz_description}")
            
            # 4. Map Capabilities
            server_tools_desc = describe_server_tools(tools_response.tools)
            mapping_desc = infer_capability_mapping(tools_response.tools)
            print(f"\nCapability mappings:\n{mapping_desc}")
            
            # 5. Build Agent Prompt
            system_prompt = build_analyst_prompt(ticker, server_tools_desc, viz_description, mapping_desc)
            
            # 6. Create Tools (Server + Local)
            server_tools = [create_server_tool(t, session) for t in tools_response.tools]
            local_tools = create_visualization_tools()
            all_tools = server_tools + local_tools
            
            # 7. Create Agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(llm, all_tools, prompt)
            executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, max_iterations=15)
            
            # 8. Run Agent
            result = await executor.ainvoke({"input": ""})
            report = result.get("output", "No analysis generated.")
            
            # 9. Combine Figures
            if len(captured_figures) > 1:
                combined = make_subplots(rows=len(captured_figures), cols=1,
                    subplot_titles=[f.layout.title.text if f.layout.title else "" for f in captured_figures],
                    vertical_spacing=0.1)
                for i, fig in enumerate(captured_figures, 1):
                    for trace in fig.data:
                        combined.add_trace(trace, row=i, col=1)
                combined.update_layout(height=400 * len(captured_figures), template="plotly_dark",
                    paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", showlegend=False)
                final_plot = combined
            elif captured_figures:
                final_plot = captured_figures[0]
            else:
                final_plot = None
            
            return final_plot, f"Analysis complete for {ticker.upper()}", report
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", ""

# Gradio UI
with gr.Blocks(theme=gr.themes.Base(primary_hue="teal")) as demo:
    gr.Markdown("""
    # Investment Analyst
    > Enter ticker symbol to generate analysis report
    """)
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="NVDA, AAPL, TSLA...", scale=3)
        analyze_button = gr.Button("🔍 Analyze", variant="primary", scale=1)
    
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Visual Report")
    summary_output = gr.Markdown(label="Financial Report")

    analyze_button.click(analyze_stock, inputs=[ticker_input], outputs=[plot_output, status_output, summary_output])

if __name__ == "__main__":
    demo.launch()