"""
Main CLI Application Entrypoint for Corporate Climate Data API & 2°C Carbon Tax ML Engine.
"""

import argparse
import json
import os
import sys

# Force UTF-8 encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data_fetcher import KNOWN_COMPANY_METRICS, ClimateDataFetcher
from src.github_pusher import GitHubPusher
from src.ml_model import CarbonTaxMLPipeline
from src.tax_calculator import CarbonTaxCalculator
from src.visualizer import ClimateTaxVisualizer

console = Console()


def display_header():
    """Renders app header banner."""
    header_text = (
        "[bold cyan]Corporate Climate Data API & 2°C Carbon Tax ML Engine[/bold cyan]\n"
        "[dim]NGFS/IPCC Climate Scenario Modeling & Scikit-Learn Predictive Analytics[/dim]"
    )
    console.print(Panel(header_text, border_style="cyan"))


def run_company_analysis(ticker: str, scenario_code: str = "2C", decarbonization_rate: float = 3.0):
    """Runs complete end-to-end pull, tax calculation, ML prediction, and chart generation for a company."""
    console.print(f"\n[bold yellow]1. Fetching Climate & Financial Data for Ticker:[/bold yellow] [bold white]{ticker}[/bold white]")
    fetcher = ClimateDataFetcher()
    profile = fetcher.get_company_profile(ticker)

    # Print Company Info Table
    info_table = Table(title=f"Corporate Profile: {profile['name']} ({ticker})", header_style="bold magenta")
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Sector", profile["sector"])
    info_table.add_row("Annual Revenue", f"${profile['revenue']/1e9:.2f} Billion USD")
    info_table.add_row("EBITDA", f"${profile['ebitda']/1e9:.2f} Billion USD ({profile['ebitda_margin']*100:.1f}%)")
    info_table.add_row("Scope 1 Emissions", f"{profile['scope1_emissions']:,.0f} tCO2e")
    info_table.add_row("Scope 2 Emissions", f"{profile['scope2_emissions']:,.0f} tCO2e")
    info_table.add_row("Scope 3 Emissions", f"{profile['scope3_emissions']:,.0f} tCO2e")
    info_table.add_row("Total Emissions", f"{profile['total_emissions']:,.0f} tCO2e")
    info_table.add_row("Carbon Intensity", f"{profile['carbon_intensity_tCO2e_per_M']:.1f} tCO2e / $1M Rev")

    console.print(info_table)

    # Calculate Tax Trajectory
    console.print(f"\n[bold yellow]2. Modeling {scenario_code} Scenario Trajectory (2025–2050):[/bold yellow]")
    calc = CarbonTaxCalculator()
    df_2c = calc.generate_trajectory(profile, scenario_code="2C", decarbonization_rate_pct=decarbonization_rate)
    df_15c = calc.generate_trajectory(profile, scenario_code="1.5C", decarbonization_rate_pct=decarbonization_rate)
    df_3c = calc.generate_trajectory(profile, scenario_code="3C", decarbonization_rate_pct=decarbonization_rate)

    # Benchmark 2030 tax table
    res_2030 = calc.compute_tax_liability(
        scope1_mt=profile["scope1_emissions"],
        scope2_mt=profile["scope2_emissions"],
        scope3_mt=profile["scope3_emissions"],
        revenue=profile["revenue"],
        ebitda=profile["ebitda"],
        scenario_code=scenario_code,
        year=2030,
        decarbonization_rate_pct=decarbonization_rate,
    )

    tax_table = Table(title=f"2030 Carbon Taxation Exposure ({scenario_code} Scenario)", header_style="bold green")
    tax_table.add_column("Financial Metric", style="cyan")
    tax_table.add_column("Scenario Output", style="white")

    tax_table.add_row("2030 Projected Carbon Price", f"${res_2030['carbon_price_usd_per_ton']:.2f} / tCO2e")
    tax_table.add_row("Direct Scope 1+2 Emissions (2030)", f"{res_2030['direct_emissions_mt']:,.0f} tCO2e")
    tax_table.add_row("Direct Annual Carbon Tax Liability", f"${res_2030['direct_tax_liability_usd']/1e6:.2f} Million USD")
    tax_table.add_row("Total Value Chain Tax Exposure (Scope 1+2+3)", f"${res_2030['total_tax_liability_usd']/1e6:.2f} Million USD")
    tax_table.add_row("Revenue at Risk (%)", f"[bold red]{res_2030['revenue_at_risk_pct']:.2f}%[/bold red]")
    tax_table.add_row("EBITDA Margin Impact (%)", f"[bold red]{res_2030['ebitda_impact_pct']:.2f}%[/bold red]")

    console.print(tax_table)

    # Machine Learning Prediction
    console.print("\n[bold yellow]3. Applying Machine Learning Model (RandomForest/GradientBoosting):[/bold yellow]")
    ml_pipeline = CarbonTaxMLPipeline()
    ml_pipeline.load_models()
    ml_pred = ml_pipeline.predict_company(profile, scenario_code=scenario_code, year=2030)

    console.print(f"  * ML Predicted Direct Tax Liability (2030): [bold green]${ml_pred['ml_predicted_tax_usd']/1e6:.2f}M USD[/bold green]")
    console.print(f"  * ML Predicted Revenue at Risk (2030): [bold red]{ml_pred['ml_predicted_revenue_at_risk_pct']:.2f}%[/bold red]")

    # Visualization Generation
    console.print("\n[bold yellow]4. Generating Analytics Charts:[/bold yellow]")
    vis = ClimateTaxVisualizer()
    chart_path = vis.plot_tax_trajectory(
        df_trajectory_2c=df_2c,
        df_trajectory_15c=df_15c,
        df_trajectory_3c=df_3c,
        company_name=profile["name"],
    )
    console.print(f"  [green][OK][/green] Saved Trajectory Chart to: [bold white]{chart_path}[/bold white]")

    return profile, res_2030, ml_pred


def run_batch_benchmark():
    """Runs carbon tax and risk evaluation across top 10 corporate tickers."""
    console.print("\n[bold yellow]Running Multi-Company Corporate Benchmark (10 Top Global Tickers)...[/bold yellow]")
    fetcher = ClimateDataFetcher()
    calc = CarbonTaxCalculator()
    ml_pipeline = CarbonTaxMLPipeline()
    ml_pipeline.load_models()

    tickers = list(KNOWN_COMPANY_METRICS.keys())
    results = []

    for t in tickers:
        p = fetcher.get_company_profile(t)
        tax_res = calc.compute_tax_liability(
            scope1_mt=p["scope1_emissions"],
            scope2_mt=p["scope2_emissions"],
            scope3_mt=p["scope3_emissions"],
            revenue=p["revenue"],
            ebitda=p["ebitda"],
            scenario_code="2C",
            year=2030,
        )
        ml_res = ml_pipeline.predict_company(p, scenario_code="2C", year=2030)
        results.append({
            "ticker": t,
            "name": p["name"],
            "sector": p["sector"],
            "revenue_b": p["revenue"] / 1e9,
            "scope1_2_mt": (p["scope1_emissions"] + p["scope2_emissions"]) / 1e6,
            "tax_2030_m": tax_res["direct_tax_liability_usd"] / 1e6,
            "revenue_at_risk_pct": tax_res["revenue_at_risk_pct"],
            "ml_risk_pct": ml_res["ml_predicted_revenue_at_risk_pct"],
        })

    df_res = pd.DataFrame(results)

    table = Table(title="2030 Corporate Carbon Tax & Risk Benchmark (NGFS 2°C Scenario)", header_style="bold blue")
    table.add_column("Ticker", style="cyan")
    table.add_column("Sector", style="white")
    table.add_column("Rev ($B)", style="white")
    table.add_column("S1+2 (M t)", style="white")
    table.add_column("2030 Tax ($M)", style="green")
    table.add_column("Rev Risk %", style="bold yellow")
    table.add_column("ML Pred Risk %", style="bold red")

    for _, r in df_res.iterrows():
        table.add_row(
            r["ticker"],
            r["sector"],
            f"${r['revenue_b']:.1f}B",
            f"{r['scope1_2_mt']:.2f}M",
            f"${r['tax_2030_m']:.1f}M",
            f"{r['revenue_at_risk_pct']:.2f}%",
            f"{r['ml_risk_pct']:.2f}%",
        )

    console.print(table)

    vis = ClimateTaxVisualizer()
    cross_path = vis.plot_cross_company_risk(df_res)
    console.print(f"  [green][OK][/green] Saved Cross-Company Benchmark Chart to: [bold white]{cross_path}[/bold white]")

    # Feature Importance plot
    df_imp = ml_pipeline.get_feature_importances()
    imp_path = vis.plot_feature_importances(df_imp)
    console.print(f"  [green][OK][/green] Saved ML Feature Importance Chart to: [bold white]{imp_path}[/bold white]")


def cli_entrypoint():
    """CLI Argument Parser and Execution Dispatcher."""
    parser = argparse.ArgumentParser(description="Corporate Climate Data API & 2°C Carbon Tax ML Engine")
    parser.add_argument("--company", type=str, default="AAPL", help="Company ticker (e.g. AAPL, MSFT, TSLA, XOM)")
    parser.add_argument("--scenario", type=str, default="2C", choices=["2C", "1.5C", "3C"], help="Climate scenario (2C, 1.5C, 3C)")
    parser.add_argument("--train-ml", action="store_true", help="Force retrain Machine Learning model pipeline")
    parser.add_argument("--batch", action="store_true", help="Run benchmark across 10 top corporations")
    parser.add_argument("--push-github", action="store_true", help="Commit and push changes to GitHub repository")
    parser.add_argument("--remote", type=str, default=None, help="GitHub repository URL for push")
    parser.add_argument("--no-interactive", action="store_true", help="Run in non-interactive batch mode")

    args = parser.parse_args()
    display_header()

    if args.train_ml:
        console.print("[bold yellow]Training ML Pipeline...[/bold yellow]")
        ml_pipeline = CarbonTaxMLPipeline()
        metrics = ml_pipeline.train()
        console.print(f"[green][OK][/green] Training complete! Tax Model R2: [bold]{metrics['tax_model']['r2']:.4f}[/bold], MAE: ${metrics['tax_model']['mae']/1e6:.2f}M")

    if args.batch:
        run_batch_benchmark()
    else:
        run_company_analysis(args.company, scenario_code=args.scenario)

    if args.push_github or args.remote:
        console.print("\n[bold yellow]Executing Git & GitHub Push Automation...[/bold yellow]")
        pusher = GitHubPusher()
        pusher.stage_and_commit()
        res = pusher.push_to_github(remote_url=args.remote)
        if res["success"]:
            console.print(f"[bold green][OK] Successfully pushed to GitHub remote repository: {res['remote_url']}[/bold green]")
        else:
            console.print(f"[bold red]! Git push note: {res['error']}[/bold red]")


if __name__ == "__main__":
    cli_entrypoint()
