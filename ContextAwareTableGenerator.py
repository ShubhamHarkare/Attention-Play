"""
Generate publication-ready metrics tables for your report.
Creates LaTeX, Markdown, and CSV formats.
"""

import json
import pandas as pd
from pathlib import Path
from TraningConfig import TrainingConfig


def load_all_results(config):
    """Load all results from saved files"""
    results = {}
    
    # Load baseline results
    try:
        with open(config.OUTPUT_DIR / "metrics" / "summary.json", 'r') as f:
            summary = json.load(f)
            results['baseline'] = summary.get('baseline_results', {})
    except:
        print("Warning: Could not load baseline results")
        results['baseline'] = {}
    
    # Load deep learning results
    try:
        with open(config.OUTPUT_DIR / "metrics" / "deep_learning_results.json", 'r') as f:
            dl_results = json.load(f)
            results['deep_learning'] = dl_results.get('test_results', {})
    except:
        print("Warning: Could not load deep learning results")
        results['deep_learning'] = {}
    
    # Load context-aware results
    try:
        with open(config.OUTPUT_DIR / "metrics" / "context_aware_results.json", 'r') as f:
            context_results = json.load(f)
            results['context_aware'] = context_results['context_aware_results']['overall']['accuracies']
            results['context_by_type'] = context_results['context_aware_results']['by_context']
    except:
        print("Warning: Could not load context-aware results")
        results['context_aware'] = {}
        results['context_by_type'] = {}
    
    return results


def create_main_comparison_table(results, config):
    """Create main comparison table of all models"""
    
    models = ['KNN', 'Cosine', 'GRU4Rec', 'Transformer', 'Context-Aware']
    k_values = [1, 5, 10, 20]
    
    # Build data
    data = []
    for model in models:
        row = {'Model': model}
        
        if model == 'Context-Aware':
            accs = results.get('context_aware', {})
        elif model in results.get('baseline', {}):
            accs = results['baseline'][model]
        elif model in results.get('deep_learning', {}):
            accs = results['deep_learning'][model]
        else:
            accs = {}
        
        for k in k_values:
            row[f'Top-{k}'] = accs.get(k, 0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_context_breakdown_table(results):
    """Create table showing performance by context"""
    
    if 'context_by_type' not in results:
        return None
    
    context_data = results['context_by_type']
    accuracies = context_data.get('accuracies', {})
    counts = context_data.get('counts', {})
    
    data = []
    for context in sorted(accuracies.keys()):
        row = {
            'Context': context.capitalize(),
            'Samples': counts.get(context, 0),
            'Top-1': accuracies[context].get(1, 0),
            'Top-5': accuracies[context].get(5, 0),
            'Top-10': accuracies[context].get(10, 0),
            'Top-20': accuracies[context].get(20, 0)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Top-10', ascending=False)
    return df


def create_improvement_table(results):
    """Create table showing improvement over baseline"""
    
    if 'Transformer' not in results.get('deep_learning', {}) or \
       'context_aware' not in results:
        return None
    
    baseline = results['deep_learning']['Transformer']
    context = results['context_aware']
    
    data = []
    for k in [1, 5, 10, 20]:
        base_acc = baseline.get(k, 0)
        context_acc = context.get(k, 0)
        
        if base_acc > 0:
            improvement = ((context_acc - base_acc) / base_acc) * 100
            abs_improvement = context_acc - base_acc
        else:
            improvement = 0
            abs_improvement = 0
        
        data.append({
            'Metric': f'Top-{k}',
            'Baseline': base_acc,
            'Context-Aware': context_acc,
            'Abs. Improvement': abs_improvement,
            'Rel. Improvement (%)': improvement
        })
    
    df = pd.DataFrame(data)
    return df


def format_latex_table(df, caption, label):
    """Convert DataFrame to LaTeX table"""
    
    # Format numbers
    for col in df.columns:
        if col not in ['Model', 'Context', 'Metric']:
            if 'Samples' in col:
                df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
            elif 'Improvement' in col and '%' in col:
                df[col] = df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            elif 'Improvement' in col:
                df[col] = df[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
            else:
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x))
    
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'r'*(len(df.columns)-1))
    
    # Wrap in table environment
    full_latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""
    return full_latex


def format_markdown_table(df):
    """Convert DataFrame to Markdown table"""
    
    # Format numbers
    for col in df.columns:
        if col not in ['Model', 'Context', 'Metric']:
            if 'Samples' in col:
                df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
            elif 'Improvement' in col and '%' in col:
                df[col] = df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            elif 'Improvement' in col:
                df[col] = df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
            else:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else str(x))
    
    return df.to_markdown(index=False)


def main():
    """Generate all metrics tables"""
    
    config = TrainingConfig()
    
    print("="*80)
    print("GENERATING METRICS TABLES")
    print("="*80)
    
    # Load results
    print("\n[1/4] Loading results...")
    results = load_all_results(config)
    
    output_dir = config.OUTPUT_DIR / "metrics" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Main comparison
    print("\n[2/4] Creating main comparison table...")
    main_table = create_main_comparison_table(results, config)
    
    print("\nMAIN MODEL COMPARISON")
    print("="*80)
    print(main_table.to_string(index=False))
    
    # Save in multiple formats
    main_table.to_csv(output_dir / "main_comparison.csv", index=False)
    
    with open(output_dir / "main_comparison.md", 'w') as f:
        f.write("# Main Model Comparison\n\n")
        f.write(format_markdown_table(main_table.copy()))
    
    with open(output_dir / "main_comparison.tex", 'w') as f:
        f.write(format_latex_table(
            main_table.copy(),
            "Performance comparison of all models on test set. "
            "Accuracies shown as percentages.",
            "tab:main_comparison"
        ))
    
    print(f"✓ Saved: main_comparison.[csv|md|tex]")
    
    # Table 2: Context breakdown
    print("\n[3/4] Creating context breakdown table...")
    context_table = create_context_breakdown_table(results)
    
    if context_table is not None:
        print("\nCONTEXT-SPECIFIC PERFORMANCE")
        print("="*80)
        print(context_table.to_string(index=False))
        
        context_table.to_csv(output_dir / "context_breakdown.csv", index=False)
        
        with open(output_dir / "context_breakdown.md", 'w') as f:
            f.write("# Context-Specific Performance\n\n")
            f.write(format_markdown_table(context_table.copy()))
        
        with open(output_dir / "context_breakdown.tex", 'w') as f:
            f.write(format_latex_table(
                context_table.copy(),
                "Context-aware model performance by playlist context. "
                "Contexts sorted by Top-10 accuracy.",
                "tab:context_breakdown"
            ))
        
        print(f"✓ Saved: context_breakdown.[csv|md|tex]")
    else:
        print("⚠ Context breakdown not available")
    
    # Table 3: Improvement analysis
    print("\n[4/4] Creating improvement analysis table...")
    improvement_table = create_improvement_table(results)
    
    if improvement_table is not None:
        print("\nIMPROVEMENT OVER BASELINE")
        print("="*80)
        print(improvement_table.to_string(index=False))
        
        improvement_table.to_csv(output_dir / "improvement_analysis.csv", index=False)
        
        with open(output_dir / "improvement_analysis.md", 'w') as f:
            f.write("# Improvement Over Baseline Transformer\n\n")
            f.write(format_markdown_table(improvement_table.copy()))
        
        with open(output_dir / "improvement_analysis.tex", 'w') as f:
            f.write(format_latex_table(
                improvement_table.copy(),
                "Context-aware model improvement over baseline Transformer. "
                "Both absolute and relative improvements shown.",
                "tab:improvement"
            ))
        
        print(f"✓ Saved: improvement_analysis.[csv|md|tex]")
    else:
        print("⚠ Improvement analysis not available")
    
    # Create summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary = {}
    
    if 'context_aware' in results:
        summary['Context-Aware Top-10'] = results['context_aware'].get(10, 0)
    
    if 'Transformer' in results.get('deep_learning', {}):
        summary['Baseline Transformer Top-10'] = results['deep_learning']['Transformer'].get(10, 0)
        
        if 'context_aware' in results:
            baseline = results['deep_learning']['Transformer'].get(10, 0)
            context = results['context_aware'].get(10, 0)
            if baseline > 0:
                summary['Relative Improvement'] = ((context - baseline) / baseline * 100)
    
    if 'KNN' in results.get('baseline', {}):
        summary['KNN Baseline Top-10'] = results['baseline']['KNN'].get(10, 0)
    
    print("\nKey Metrics:")
    for key, value in summary.items():
        if 'Improvement' in key:
            print(f"  {key}: {value:+.1f}%")
        else:
            print(f"  {key}: {value:.2f}%")
    
    # Save summary
    with open(output_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL TABLES GENERATED!")
    print("="*80)
    print(f"\n📁 Tables saved to: {output_dir}/")
    print("\nFormats available:")
    print("  • CSV - For Excel/spreadsheets")
    print("  • Markdown (.md) - For GitHub/documentation")
    print("  • LaTeX (.tex) - For academic papers/reports")
    print("\n💡 For your report, you can:")
    print("  1. Copy-paste Markdown tables directly")
    print("  2. Import CSV into Word/Google Docs")
    print("  3. Use LaTeX for technical reports")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()