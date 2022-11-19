import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans serif"

def plot_variation_summary(gene_symbol,      
                           variants,
                           patients_table,
                           refseq,
                           subfigs):

    # histogram: number of variants per sample
    axL = subfigs[0].subplots()
    r = subfigs[0].canvas.get_renderer()

    # total number of variant samples
    heights, xs = np.histogram(a = patients_table['n_vars'], 
                               bins = np.arange(0, patients_table['n_vars'].max() + 3,1))
    n_var_samples = heights[1:].sum()

    axL.bar(xs[:-1], 
            heights, 
            width = 1, 
            color = 'tab:grey',
            alpha = 0.6)
    axL.set_yscale('log')
    axL.set_ylabel("# samples")
    axL.set_xlabel("# variants in sample")
    axL._children[0]._facecolor = (0,0,0,1)
    axL._children[0]._alpha = 1
    txt = axL.text(axL.dataLim._points[1, 0],
                   axL.dataLim._points[1, 1],
                   f"{n_var_samples:,}", 
                   ha='right', va='top', 
                   fontsize = 11, 
                   weight = 'bold', 
                   color = 'tab:grey')

    txtbox = axL.transData.inverted()\
                        .transform(txt.get_window_extent(renderer = r).get_points())\
                        .data

    t = axL.text(txtbox[1, 0],
             txtbox[0,1],
             "variant samples", 
             ha='right', va='top', 
             fontsize = 8, 
             color = 'tab:grey')

    # sequence polymorphism plot
    axR = subfigs[1].subplots(1, 2, 
                              gridspec_kw={"width_ratios":[3,1], 
                                                 "wspace":0.005})
    r = subfigs[1].canvas.get_renderer()

    nvars_per_pos = variants.groupby("AA_POS").aggregate({'AC': np.sum})
    seq_variation = np.zeros(len(refseq.seq))                    
    seq_variation[nvars_per_pos.index.to_numpy()-1] = nvars_per_pos['AC'].to_numpy()

    ax = axR[0]
    ax.plot(seq_variation, lw = 0.5)
    ax.set_xticks(np.linspace(0, len(refseq.seq), 3).astype(int))
    ax.set_yscale('log')
    ax.set_ylabel('# variants')
    ax.set_xlabel('res. position')

    txt = ax.text(ax.dataLim._points[0, 0],
                  ax.dataLim._points[1, 1],
                  f"{len(variants['AA_POS'].unique()):,}", 
                  ha='left', va='top', 
                  fontsize = 11, 
                  weight = 'bold', 
                  color = 'tab:blue')

    txtbox = ax.transData.inverted().transform(txt.get_window_extent(renderer = r).get_points()).data

    ax.text(txtbox[0, 0],
            txtbox[0,1],
            "variant positions", 
            ha='left', va='top', fontsize = 8, color = 'tab:blue')

    # histogram of variant counts per position 
    ax = axR[1]           
    hist = ax.hist(np.log10(nvars_per_pos['AC'].to_numpy()), 
                   bins = 20, 
                   orientation='horizontal', 
                   align = 'left', 
                   alpha = 0.6);

    # add bar for 0s
    hist_w = hist[2].patches[0]._height
    ax.barh(y =-hist_w, 
            width = (seq_variation == 0).sum(), 
            height = hist_w, 
            color = "black")
    # match the two plots' y scales
    axR[1].set_ylim(np.log10(axR[0].get_ylim()))

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('# positions')
    ax.set_xscale('log')

    subfigs[0].suptitle(gene_symbol, 
                        fontsize = 12, 
                        x=0.25, ha='left', 
                        weight = 'bold')
    subfigs[1].suptitle(' ', fontsize = 12)


def plot_sequence_variation_content(gene_symbol, 
                                    seq_table,
                                    ax):

    unique_n_vars = sorted(seq_table.loc[seq_table['n_vars'] > 0, 'n_vars'].unique())
    n_unique_var_seqs = len(seq_table)-1
    counts_per_nvars = [seq_table[seq_table['n_vars']==n]['seq_count'].value_counts() for n in unique_n_vars]
    counts_per_nvars = [c.sort_index() for c in counts_per_nvars]
    
    cmap = plt.get_cmap('RdBu')
    
    for i,counts in enumerate(counts_per_nvars):
        ax.step(counts.index, 
                np.cumsum(counts.values[::-1])[::-1],
                color = cmap(i/len(counts_per_nvars)))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid('on', alpha = 0.3)
    ax.set_xlabel('allele count (AC)')
    ax.set_ylabel(r'# sequences $\geq$ AC')
    ax.set_title(f"{gene_symbol} ({n_unique_var_seqs})", weight="bold")
    ax.legend(unique_n_vars, title = "# SNPs")
