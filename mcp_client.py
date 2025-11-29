import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import asyncio
from contextlib import AsyncExitStack
import json
import os
from datetime import datetime
import traceback
import inspect
from typing import Any, Callable, Dict

# --- GenAI and MCP Imports ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

# Import protobuf types for manual function response construction
from google.ai.generativelanguage_v1beta.types import Part, FunctionResponse

from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# --- Local Client Capabilities (Visualizations) ---

def plot_historical_price_chart(data_json: str) -> plt.Figure:
    """Plots a 7-day price history chart."""
    try:
        if isinstance(data_json, dict): history_data = data_json
        else: history_data = json.loads(data_json)
        
        if "error" in history_data or not history_data.get('dates') or not history_data.get('prices'):
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in history_data['dates']]
        ax.plot(dates, history_data['prices'], 'o-', label='Closing Price', color='royalblue', markersize=6)
        ax.set_title('7-Day Price History', fontsize=14)
        ax.set_ylabel('Price (USD)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout(pad=2.0)
        return fig
    except Exception:
        return None

def plot_analyst_recommendations_chart(data_json: str) -> plt.Figure:
    """Plots analyst recommendation trends."""
    try:
        if isinstance(data_json, (dict, list)): trends_data = data_json
        else: trends_data = json.loads(data_json)
        
        trend = trends_data[0] if isinstance(trends_data, list) and trends_data else trends_data
        
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        counts = [trend.get(k, 0) for k in ['strongSell', 'sell', 'hold', 'buy', 'strongBuy']]
        colors = ['darkred', 'red', 'orange', 'skyblue', 'darkgreen']
        ax.bar(labels, counts, color=colors)
        ax.set_title(f'Analyst Recommendations', fontsize=14)
        ax.set_ylabel('Count')
        plt.tight_layout(pad=2.0)
        return fig
    except Exception:
        return None

# --- Dynamic Tool Bridge ---

def create_tool_wrapper(session: ClientSession, tool_name: str) -> Callable:
    """
    Creates a dynamic python function that wraps the MCP 'call_tool' method.
    """
    async def tool_wrapper(**kwargs):
        # Call the remote MCP server tool
        result = await session.call_tool(name=tool_name, arguments=kwargs)
        if result.content:
            text_content = [c.text for c in result.content if hasattr(c, 'text')]
            return "\n".join(text_content)
        return "No content returned."
    
    tool_wrapper.__name__ = tool_name
    tool_wrapper.__doc__ = f"Dynamic MCP tool: {tool_name}"
    return tool_wrapper


async def analyze_and_plot(ticker: str):
    if not ticker:
        return None, "Please enter a ticker symbol.", ""

    async with AsyncExitStack() as stack:
        try:
            # 1. Connect to Server
            server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
            read_pipe, write_pipe = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_pipe, write_pipe))
            await session.initialize()

            # 2. Dynamic Discovery
            available_tools_list = await session.list_tools()
            available_prompts_list = await session.list_prompts()
            
            # 3. Create Wrappers & Map
            tool_map = {}
            server_tool_functions = []
            
            # Map Server Tools
            for tool in available_tools_list.tools:
                wrapper = create_tool_wrapper(session, tool.name)
                tool_map[tool.name] = wrapper
                server_tool_functions.append(wrapper)

            # Map Local Tools
            async def local_history_wrapper(data_json): return plot_historical_price_chart(data_json)
            async def local_rec_wrapper(data_json): return plot_analyst_recommendations_chart(data_json)
            
            tool_map["plot_historical_price_chart"] = local_history_wrapper
            tool_map["plot_analyst_recommendations_chart"] = local_rec_wrapper

            # 4. Fetch Instruction
            target_prompt_name = "analyze_stock"
            found_prompt = next((p for p in available_prompts_list.prompts if p.name == target_prompt_name), None)
            if not found_prompt:
                return None, f"Server does not support prompt '{target_prompt_name}'", ""

            prompt_result = await session.get_prompt(target_prompt_name, arguments={"ticker": ticker})
            system_instruction = prompt_result.messages[0].content.text

            # 5. Initialize Model (Manual Mode)
            local_tools = [plot_historical_price_chart, plot_analyst_recommendations_chart]
            
            # Safety: Ensure no duplicate tool names in list
            # We filter server tools if they clash with local tool names (unlikely but safe)
            local_tool_names = {t.__name__ for t in local_tools}
            unique_server_tools = [t for t in server_tool_functions if t.__name__ not in local_tool_names]
            
            all_tools = unique_server_tools + local_tools
            
            model = genai.GenerativeModel(
                'gemini-2.5-flash-lite',
                tools=all_tools,
                generation_config=GenerationConfig(temperature=0.0)
            )
            chat = model.start_chat()
            
            # 6. Manual Execution Loop
            response = await chat.send_message_async(system_instruction)
            
            captured_figures = []

            # Loop as long as the model wants to call functions
            while response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                # Check if the response contains a function call
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    fn_name = fc.name
                    fn_args = dict(fc.args)
                    
                    # Execute the tool
                    if fn_name in tool_map:
                        try:
                            # Await the tool execution
                            fn_result = await tool_map[fn_name](**fn_args)
                            
                            # Capture Plots Side-effect
                            if isinstance(fn_result, plt.Figure):
                                captured_figures.append(fn_result)
                                api_response = {"result": "Chart generated successfully."}
                            else:
                                api_response = fn_result
                                
                        except Exception as e:
                            api_response = {"error": str(e)}
                    else:
                        api_response = {"error": f"Tool {fn_name} not found"}

                    # Send the result back to Gemini
                    response_part = Part(
                        function_response=FunctionResponse(
                            name=fn_name,
                            response={"result": api_response} 
                        )
                    )
                    
                    response = await chat.send_message_async([response_part])
                else:
                    break

            final_text = response.text

            # 7. Combine Figures
            final_plot = None
            if len(captured_figures) > 1:
                combined_fig, axes = plt.subplots(len(captured_figures), 1, figsize=(10, 5 * len(captured_figures)))
                if len(captured_figures) == 1: axes = [axes]
                for i, fig in enumerate(captured_figures):
                    ax_new = axes[i]
                    if fig.axes:
                        ax_old = fig.axes[0]
                        ax_new.set_title(ax_old.get_title())
                        ax_new.set_ylabel(ax_old.get_ylabel())
                        
                        # Copy lines
                        for line in ax_old.get_lines():
                            ax_new.plot(line.get_xdata(), line.get_ydata(), 
                                        color=line.get_color(), marker=line.get_marker(), 
                                        label=line.get_label())
                        # Copy bars
                        for patch in ax_old.patches:
                             ax_new.add_patch(
                                plt.Rectangle((patch.get_x(), patch.get_y()), 
                                              patch.get_width(), patch.get_height(),
                                              facecolor=patch.get_facecolor())
                             )
                        ax_new.autoscale()
                    plt.close(fig)
                combined_fig.tight_layout()
                final_plot = combined_fig
            elif len(captured_figures) == 1:
                final_plot = captured_figures[0]

            return final_plot, f"Analysis complete for {ticker.upper()}.", final_text

        except Exception as e:
            traceback.print_exc()
            return None, f"Error: {str(e)}", ""

# --- Gradio Interface ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# Agnostic MCP Investment Advisor")
    gr.Markdown("The Client does not know how to analyze stocks. It connects to the Server, asks for instructions, and follows them.")
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="NVDA")
        analyze_button = gr.Button("Analyze")
        
    status_output = gr.Textbox(label="Status", interactive=False)
    plot_output = gr.Plot(label="Visual Report")
    summary_output = gr.Markdown(label="Report")

    analyze_button.click(
        analyze_and_plot,
        inputs=[ticker_input],
        outputs=[plot_output, status_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()